#!/usr/bin/env python3
"""
train_rb2d_pbfm_conditioned.py
================================================================================
Train the (Ra,Pr)-CONDITIONED PBFM velocity field v_theta(x_t, t, Ra, Pr) for
2-D steady Rayleigh-Benard convection.

PBFM is trained CONDITIONED ONLY, by design. Without conditioning the model has
no handle on the input parameters at all: physics enters PBFM through the
training-time residual loss, and sampling is a plain ODE integration, so there is
no inference-time mechanism that could steer an unconditioned model towards a
requested (Ra,Pr). (PCFM is different -- its physics projection at sampling time
supplies exactly that control, which is why an unconditioned PCFM variant was
worth training and checking.)

PBFM = flow matching + a PDE-residual loss on the UNROLLED clean-sample
prediction x~1, combined CONFLICT-FREE via ConFIG (Baldan et al. 2025, Alg. 1).

--------------------------------------------------------------------------
MULTI-GPU (Kaggle 2 x T4)
--------------------------------------------------------------------------
Both T4s are used by default (--gpus defaults to all visible GPUs).

    # simplest -- works in a VS Code remote terminal AND in a notebook cell
    python train_rb2d_pbfm_conditioned.py --bank ./rb2d/split/train_bank.pt --gpus 2

    # or via torchrun (auto-detected, --gpus then ignored)
    torchrun --nproc_per_node=2 train_rb2d_pbfm_conditioned.py \
        --bank ./rb2d/split/train_bank.pt

`--batch` is PER GPU, so 2 x T4 with --batch 32 gives a global batch of 64.
Each rank draws different samples; the two per-loss gradients are averaged with
one NCCL all_reduce each, so the update equals the exact global-batch update.
The whole bank is resident on each GPU (it is small), so there is no DataLoader,
no worker processes and no host->device copies inside the loop -- the usual
sources of multi-GPU stalling simply do not exist here.

If NCCL misbehaves on a particular Kaggle image, fall back with --backend gloo
(slower but reliable), or just run --gpus 1.

Resumes automatically from <ckpt>_resume.pt; --no-resume forces a fresh run.
"""
import argparse
import math
import torch

import rb2d_pbfm_common as C


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', default='./rb2d/split/train_bank.pt',
                   help='TRAIN bank written by prepare_rb2d_pbfm_data.py')
    p.add_argument('--out', default='ckpt_rb2d_pbfm_cond.pt')
    # multi-GPU
    p.add_argument('--gpus', type=int, default=None,
                   help='number of GPUs (default: all visible). Ignored under torchrun.')
    p.add_argument('--backend', default=None, choices=['nccl', 'gloo'],
                   help='distributed backend (default nccl on GPU, gloo on CPU)')
    p.add_argument('--port', default='29500', help='rendezvous port for --gpus>1')
    # model
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--modes-x', dest='modes_x', type=int, default=20)
    p.add_argument('--modes-z', dest='modes_z', type=int, default=12)
    p.add_argument('--n-layers', dest='n_layers', type=int, default=4)
    p.add_argument('--time-emb', dest='time_emb', type=int, default=32)
    # PBFM training
    p.add_argument('--iters', type=int, default=40000)
    p.add_argument('--batch', type=int, default=64,
                   help='GLOBAL batch size (paper App.G keeps this constant '
                        'regardless of GPU count; each GPU takes batch//gpus)')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--unroll', type=int, default=4,
                   help='PBFM unrolling steps used to predict x~1 (paper uses a '
                        'curriculum ending at 2-4; 4 gave the best residuals)')
    p.add_argument('--unroll-start', dest='unroll_start', type=int, default=1,
                   help='curriculum start (paper S3.3: unrolling is ramped up '
                        'during training). Set equal to --unroll to disable.')
    p.add_argument('--curriculum-end', dest='curriculum_end', type=int, default=None,
                   help='iteration at which the curriculum reaches --unroll '
                        '(default: half of --iters)')
    p.add_argument('--lr-schedule', dest='lr_schedule', default='constant',
                   choices=['constant', 'cosine'],
                   help='paper App.G uses a FIXED learning rate')
    p.add_argument('--adam-b1', dest='adam_b1', type=float, default=0.5,
                   help='paper App.G: beta1 = 0.5')
    p.add_argument('--adam-b2', dest='adam_b2', type=float, default=0.999)
    p.add_argument('--resid-p', dest='resid_p', type=float, default=1.0,
                   help='power-law weight t^p on the residual loss (paper optimum 1)')
    p.add_argument('--backprop', choices=['last', 'all'], default='last',
                   help='backprop residual through last unroll step (default) or all')
    p.add_argument('--config', dest='use_config', action='store_true', default=True,
                   help='combine FM + residual gradients conflict-free (ConFIG)')
    p.add_argument('--no-config', dest='use_config', action='store_false')
    p.add_argument('--resid-weight', dest='resid_weight', type=float, default=1.0,
                   help='fixed residual weight when --no-config is used')
    p.add_argument('--warmup-iters', dest='warmup_iters', type=int, default=1000,
                   help='iters of pure FM before the residual loss is switched on')
    p.add_argument('--ema-decay', dest='ema_decay', type=float, default=0.999)
    p.add_argument('--grad-clip', dest='grad_clip', type=float, default=1.0)
    p.add_argument('--val-frac', dest='val_frac', type=float, default=0.10)
    p.add_argument('--val-every', dest='val_every', type=int, default=500)
    p.add_argument('--patience', type=int, default=8,
                   help='stop after this many consecutive validations without '
                        'improvement (so patience x --val-every iterations). '
                        'Set 0 to DISABLE early stopping and run all --iters.')
    p.add_argument('--early-stop-metric', dest='early_stop_metric',
                   default='fm', choices=['fm', 'gen_res'],
                   help="what to monitor: 'fm' = generative val loss (default); "
                        "'gen_res' = physics residual of samples generated from "
                        "noise (the deployment metric -- better for PBFM, since "
                        "the FM loss often flattens while physics still improves)")
    p.add_argument('--log-every', dest='log_every', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--no-physics', dest='physics', action='store_false',
                   default=True,
                   help='train a plain flow-matching model (no residual loss, no '
                        'ConFIG). NOT NEEDED for the paper: the eval spec says '
                        'vanilla and PCFM share one checkpoint, so the vanilla '
                        'baseline is the PCFM plain-FM checkpoint you already '
                        'have (ckpt_rb2d_cond.pt). Kept only for ablations.')
    p.add_argument('--val-gen-samples', dest='val_gen_samples', type=int, default=8,
                   help='samples generated from pure noise at each validation to '
                        'measure the residual the SAMPLER actually produces')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if not args.physics and args.out == 'ckpt_rb2d_pbfm_cond.pt':
        args.out = 'ckpt_rb2d_fm_baseline.pt'
    if args.quick:
        args.iters = 300
        args.hidden = 24
        args.modes_x = 10
        args.modes_z = 8
        args.n_layers = 2
        args.batch = 8
        args.unroll = 2
        args.unroll_start = 1
        args.warmup_iters = 50
        args.val_every = 100
        args.log_every = 50
        args.patience = 99
    return args


def sanity_sample(ckpt, data, device):
    m, ck = C.load_model(ckpt, device=device)
    rc = math.sqrt(data.Ra_range[0] * data.Ra_range[1])
    pc = math.sqrt(data.Pr_range[0] * data.Pr_range[1])
    f, info = C.pbfm_sample(m, ck, data, rc, pc, K=5, n_step=50, seed=0,
                            device=device)
    print(f'[sanity] samples at Ra={rc:.0f} Pr={pc:.2f}: '
          f'rolls={info["roll"].tolist()} '
          f'resid={[f"{x:.2e}" for x in info["residual"].tolist()]}')


def worker(rank, world_size, args):
    device = C.ddp_setup(rank, world_size, backend=args.backend, port=args.port)
    main = C.is_main(rank)
    if main:
        tag = 'CONDITIONED PBFM' if args.physics else 'CONDITIONED FM BASELINE (no physics)'
        print(f'[train] {tag} | world_size={world_size} '
              f'| device={device} | global batch={args.batch} '
              f'| per-GPU {args.batch // max(1, world_size)}')
        if torch.cuda.is_available():
            for g in range(min(world_size, torch.cuda.device_count())):
                print(f'         GPU{g}: {torch.cuda.get_device_name(g)}')
    try:
        data = C.RB2DData(C.find_file(args.bank) or args.bank, device=device)
        if main:
            print(f'[train] {data.N} fields  grid {data.Nx}x{data.Nz}  '
                  f'branches {data.n_list}  '
                  f'sigma={[round(s, 4) for s in data.sigma.tolist()]}')
        ckpt = C.train_pbfm(
            data, cond=True, iters=args.iters, batch=args.batch, lr=args.lr,
            hidden=args.hidden, modes_x=args.modes_x, modes_z=args.modes_z,
            n_layers=args.n_layers, time_emb=args.time_emb,
            unroll=args.unroll, resid_p=args.resid_p, backprop=args.backprop,
            use_config=args.use_config, resid_weight=args.resid_weight,
            warmup_iters=args.warmup_iters, ema_decay=args.ema_decay,
            grad_clip=args.grad_clip, val_frac=args.val_frac,
            val_every=args.val_every, patience=args.patience,
            log_every=args.log_every, device=device, ckpt_out=args.out,
            resume=args.resume, seed=args.seed,
            rank=rank, world_size=world_size,
            physics=args.physics, val_gen_samples=args.val_gen_samples,
            adam_b1=args.adam_b1, adam_b2=args.adam_b2,
            lr_schedule=args.lr_schedule, unroll_start=args.unroll_start,
            curriculum_end=args.curriculum_end,
            early_stop_metric=args.early_stop_metric)
        if main:
            sanity_sample(ckpt, data, device)
    finally:
        if world_size > 1:
            C.ddp_cleanup()


def main():
    args = parse_args()
    gpus = args.gpus
    if gpus is None:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    C.launch(worker, gpus, args)


if __name__ == '__main__':
    main()