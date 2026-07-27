#!/usr/bin/env python3
"""
train_rb3d_pbfm_conditioned.py
================================================================================
Train the (Ra,Pr)-CONDITIONED PBFM velocity field v_theta(x_t, t, Ra, Pr) for
3-D steady Rayleigh-Benard convection. 3-D analog of
train_rb2d_pbfm_conditioned.py, built on top of rb3d_pbfm_common.py (which
reuses rb3d_pcfm_common.FNO3d / RB3DData unmodified -- same architecture as
3-D PCFM, so its plain-FM checkpoint IS the "vanilla" baseline; no separate
vanilla run needed).

PBFM is trained CONDITIONED ONLY, by design: physics enters through the
training-time residual loss and sampling is a plain ODE integration, so there
is no inference-time mechanism that could steer an unconditioned model toward
a requested (Ra,Pr) -- unlike PCFM, whose sampling-time projection supplies
exactly that control.

--------------------------------------------------------------------------
MULTI-GPU (dual T4)
--------------------------------------------------------------------------
Both GPUs are used by default (--gpus defaults to all visible GPUs):

    python train_rb3d_pbfm_conditioned.py \
        --bank ./datasets/rb3d_multisolution/splits/train_bank.pt --gpus 2

    # or via torchrun (auto-detected, --gpus then ignored)
    torchrun --nproc_per_node=2 train_rb3d_pbfm_conditioned.py --bank ...

`--batch` is the GLOBAL batch (PBFM paper App. G keeps this constant
regardless of GPU count); each rank takes batch//gpus and draws a different
data slice. ConFIG needs the flow-matching and residual gradients kept
separate, so there is deliberately no DDP wrapper -- see rb3d_pbfm_common.py's
multi-GPU section for why.

Checkpointing survives a session timeout: `<out>_last.pt` (full resume state)
is overwritten every --eval-every iters regardless of validation outcome;
`<out>.pt` (best EMA weights, same payload shape as PCFM's checkpoints -- one
loader for vanilla/PBFM/PCFM) only on improvement. Resumes automatically;
--no-resume forces a fresh run.
"""

import argparse
import os

import numpy as np
import torch

import rb3d_pbfm_common as C


def parse_args():
    cfg = C.default_cfg_pbfm()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank',
                   default='./datasets/rb3d_multisolution/splits/train_bank.pt')
    p.add_argument('--out', default=None, help='output directory')
    p.add_argument('--ckpt-name', default='ckpt_rb3d_pbfm_cond.pt')
    # multi-GPU
    p.add_argument('--gpus', type=int, default=None,
                   help='number of GPUs (default: all visible). Ignored under torchrun.')
    p.add_argument('--backend', default=None, choices=['nccl', 'gloo'])
    p.add_argument('--port', default='29500')
    # model (kept identical in spirit to PCFM's defaults for comparability)
    p.add_argument('--hidden', type=int, default=cfg['hidden'])
    p.add_argument('--modes-x', dest='modes_x', type=int, default=cfg['modes_x'])
    p.add_argument('--modes-y', dest='modes_y', type=int, default=cfg['modes_y'])
    p.add_argument('--modes-z', dest='modes_z', type=int, default=cfg['modes_z'])
    p.add_argument('--n-layers', dest='n_layers', type=int, default=cfg['n_layers'])
    p.add_argument('--time-emb', dest='time_emb', type=int, default=cfg['time_emb'])
    # PBFM training
    p.add_argument('--iters', type=int, default=cfg['iters'])
    p.add_argument('--batch', type=int, default=cfg['batch'],
                   help='GLOBAL batch size (constant regardless of GPU count; '
                        'each GPU takes batch//gpus)')
    p.add_argument('--lr', type=float, default=cfg['lr'])
    p.add_argument('--unroll', type=int, default=cfg['unroll'],
                   help='PBFM unrolling steps used to predict x~1. Start small '
                        '(default 2) -- the 3-D Leray-projected residual is far '
                        'costlier per call than 2-D\'s Poisson solve.')
    p.add_argument('--unroll-start', dest='unroll_start', type=int,
                   default=cfg['unroll_start'],
                   help='curriculum start; None/>=--unroll disables the curriculum '
                        '(default: ramp from 1, matching 2-D\'s actual practice)')
    p.add_argument('--curriculum-end', dest='curriculum_end', type=int,
                   default=cfg['curriculum_end'],
                   help='iteration the curriculum reaches --unroll (default: '
                        'half of --iters)')
    p.add_argument('--lr-schedule', dest='lr_schedule', default=cfg['lr_schedule'],
                   choices=['constant', 'cosine'])
    p.add_argument('--adam-b1', dest='adam_b1', type=float, default=cfg['adam_b1'])
    p.add_argument('--adam-b2', dest='adam_b2', type=float, default=cfg['adam_b2'])
    p.add_argument('--resid-p', dest='resid_p', type=float, default=cfg['resid_p'])
    p.add_argument('--backprop', choices=['last', 'all'], default=cfg['backprop'])
    p.add_argument('--config', dest='use_config', action='store_true', default=True)
    p.add_argument('--no-config', dest='use_config', action='store_false')
    p.add_argument('--resid-weight', dest='resid_weight', type=float,
                   default=cfg['resid_weight'])
    p.add_argument('--warmup-iters', dest='warmup_iters', type=int,
                   default=cfg['warmup_iters'])
    p.add_argument('--ema-decay', dest='ema', type=float, default=cfg['ema'])
    p.add_argument('--grad-clip', dest='grad_clip', type=float, default=cfg['grad_clip'])
    p.add_argument('--val-frac', dest='val_frac', type=float, default=cfg['val_frac'])
    p.add_argument('--eval-every', dest='val_every', type=int, default=cfg['val_every'])
    p.add_argument('--patience', type=int, default=cfg['patience'],
                   help='stop after this many consecutive evals without '
                        'improvement. 0 disables early stopping.')
    p.add_argument('--min-iters', dest='min_iters', type=int,
                   default=cfg['min_iters'],
                   help='early stopping cannot fire before this iteration '
                        'regardless of --patience (default: 0.4 * --iters)')
    p.add_argument('--early-stop-metric', dest='early_stop_metric',
                   default=cfg['early_stop_metric'], choices=['fm', 'gen_res'])
    p.add_argument('--val-gen-samples', dest='val_gen_samples', type=int,
                   default=cfg['val_gen_samples'])
    p.add_argument('--log-every', dest='log_every', type=int, default=100)
    p.add_argument('--seed', type=int, default=cfg['seed'])
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--no-physics', dest='physics', action='store_false',
                   default=True,
                   help='train a plain flow-matching model (no residual loss, '
                        'no ConFIG). NOT NEEDED for the paper: vanilla == the '
                        'PCFM plain-FM checkpoint (same architecture). Kept '
                        'only for ablations.')
    p.add_argument('--no-sanity', dest='sanity', action='store_false')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if not args.physics and args.ckpt_name == 'ckpt_rb3d_pbfm_cond.pt':
        args.ckpt_name = 'ckpt_rb3d_fm_baseline.pt'
    if args.quick:
        args.iters = 300
        args.hidden, args.modes_x, args.modes_y, args.modes_z = 16, 8, 4, 4
        args.n_layers = 2
        args.batch = 4
        args.unroll = 1
        args.warmup_iters = 50
        args.val_every = 100
        args.log_every = 20
        args.patience = 0
    return args


def sanity_sample(ckpt, data, device):
    model, ck = C.load_flow_model(ckpt, device)
    Ra0 = float(np.exp(0.5 * (np.log(data.Ra_rng[0]) + np.log(data.Ra_rng[1]))))
    Pr0 = float(np.exp(0.5 * (np.log(data.Pr_rng[0]) + np.log(data.Pr_rng[1]))))
    fields, diag = C.pbfm_sample_3d(model, ck, data, [Ra0] * 5, [Pr0] * 5,
                                    n_step=50, seed=0)
    print(f'[sanity] Ra={Ra0:.0f} Pr={Pr0:.2f} | planforms={diag["planform"]} | '
          f'residual={[f"{r:.2e}" for r in diag["residual"].tolist()]} | '
          f'{diag["seconds"]:.1f}s for 5 samples (plain ODE, no projection)')


def worker(rank, world_size, args, data=None):
    device = C.ddp_setup(rank, world_size, backend=args.backend, port=args.port)
    main = C.is_main(rank)
    if main:
        tag = 'CONDITIONED PBFM' if args.physics else 'CONDITIONED FM BASELINE (no physics)'
        print(f'[train] {tag} | world_size={world_size} | device={device} '
              f'| global batch={args.batch} | per-GPU {args.batch // max(1, world_size)}')
        if torch.cuda.is_available():
            for g in range(min(world_size, torch.cuda.device_count())):
                print(f'         GPU{g}: {torch.cuda.get_device_name(g)}')
    try:
        cfg = C.default_cfg_pbfm()
        cfg.update(
            hidden=args.hidden, modes_x=args.modes_x, modes_y=args.modes_y,
            modes_z=args.modes_z, n_layers=args.n_layers, time_emb=args.time_emb,
            iters=args.iters, batch=args.batch, lr=args.lr,
            unroll=args.unroll, unroll_start=args.unroll_start,
            curriculum_end=args.curriculum_end, lr_schedule=args.lr_schedule,
            adam_b1=args.adam_b1, adam_b2=args.adam_b2, resid_p=args.resid_p,
            backprop=args.backprop, use_config=args.use_config,
            resid_weight=args.resid_weight, warmup_iters=args.warmup_iters,
            ema=args.ema, grad_clip=args.grad_clip, val_frac=args.val_frac,
            val_every=args.val_every, patience=args.patience,
            min_iters=args.min_iters,
            early_stop_metric=args.early_stop_metric,
            val_gen_samples=args.val_gen_samples, log_every=args.log_every,
            seed=args.seed, augment=args.augment, physics=args.physics)

        if data is None:
            # torchrun path: each rank is an independently-launched OS process
            # (no common parent to share memory from), so it must load its own
            # copy. Under --gpus N (mp.spawn), main() below loads ONE copy and
            # shares it here instead -- see main()'s comment for why that
            # matters on a multi-GB bank.
            data = C.RB3DData(args.bank, device=device, val_frac=cfg['val_frac'],
                              seed=args.seed)
        else:
            data.device = device        # each rank targets its own GPU; the
                                        # underlying field storage is shared
        out = args.out or ('/kaggle/working' if os.path.isdir('/kaggle/working')
                           else '.')
        os.makedirs(out, exist_ok=True)
        ckpt_path = os.path.join(out, args.ckpt_name)

        ckpt = C.train_pbfm_3d(data, cfg, cond=True, device=device,
                               ckpt_path=ckpt_path, eval_every=args.val_every,
                               patience=args.patience, rank=rank,
                               world_size=world_size, resume=args.resume)
        if main and args.sanity:
            sanity_sample(ckpt, data, device)
    finally:
        if world_size > 1:
            C.ddp_cleanup()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gpus = args.gpus
    if gpus is None:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpus = max(1, int(gpus))

    # Load the (multi-GB) data bank ONCE and share its storage across every
    # spawned rank via share_memory_(), instead of each rank independently
    # torch.load-ing + densely re-copying the whole bank. RB3DData.__init__
    # first loads the full raw bank dict, then copies it into a freshly
    # allocated dense tensor while releasing entries -- so each rank
    # transiently needs close to 2x the bank's on-disk size at the peak of
    # that copy. Two ranks hitting that peak simultaneously under mp.spawn
    # (an 8.6 GB bank -> ~15-17 GB/rank peak) is what SIGKILLed a 31 GB-RAM
    # box here; sharing one copy keeps total RAM cost independent of GPU count.
    # torchrun launches genuinely separate OS processes with no common parent
    # to share memory from, so that path still loads independently per rank.
    if C.ddp_is_torchrun():
        data = None
    else:
        data = C.RB3DData(args.bank, device='cpu', val_frac=args.val_frac,
                          seed=args.seed)
        data.fields.share_memory_()
        data.params.share_memory_()
        data.Ra.share_memory_()
        data.Pr.share_memory_()
    C.launch(worker, gpus, args, data)


if __name__ == '__main__':
    main()
