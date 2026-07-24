#!/usr/bin/env python3
"""
evaluate_rb2d_groundtruth.py
================================================================================
Ground-truth comparison for the CONDITIONED RB2D PBFM model. For bank (Ra,Pr)
points that carry several reference branches, draw K PBFM samples, align each to
the reference of matching roll number (symmetry-aware: circular x-shift + the
reflection u_x -> -u_x), and report:

  * Data MSE, Data NRMSE (%)   -- fidelity to the reference after alignment
  * Phys L2                    -- buoyancy-scaled physics residual of the sample
  * GT Phys L2                 -- the reference's own Phys L2 (the O(0.1) floor)

Reported per branch n and overall (AVG). Lead with Data NRMSE and the
floor-relative view: for RB the reference's own Phys L2 is O(0.1) (not 1e-6 as in
Bratu), so a sample sitting at that floor is as physically good as ground truth.

MULTI-GPU (Kaggle 2 x T4): points are sharded across GPUs with --gpus.

Example:
    python evaluate_rb2d_groundtruth.py --bank ./rb2d/split/train_bank.pt \
        --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2 \
        --n-points 4 --point-seed 0 --k-samples 40
"""
import argparse
import glob
import os
import numpy as np
import torch

import rb2d_pbfm_common as C

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


def resolve_device(rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return f'cuda:{rank % torch.cuda.device_count()}'
    return 'cpu'


def pick_points(data, n_points, min_branches, seed, explicit=None):
    uniq = data.unique_points()
    if explicit:
        chosen = []
        for (Ra, Pr) in explicit:
            best = min(uniq.keys(), key=lambda k: abs(k[0] - Ra) + abs(k[1] - Pr))
            chosen.append((best, uniq[best]))
        return chosen
    good = [(k, v) for k, v in uniq.items() if len(v) >= min_branches]
    if not good:
        good = list(uniq.items())
    good.sort(key=lambda kv: (kv[0][0], kv[0][1]))          # deterministic order
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(good), size=min(n_points, len(good)), replace=False)
    return [good[i] for i in sorted(idx)]


def worker(rank, world_size, args):
    device = resolve_device(rank)
    data = C.RB2DData(C.find_file(args.bank) or args.bank, device=device)
    E = C.build_eig(data.Nx, data.Nz, data.dx, data.dz, device, torch.float64)
    m, ck = C.load_model(C.find_file(args.ckpt) or args.ckpt, device=device)
    thr, _floor = C.calibrate_thresholds(data)

    explicit = None
    if args.points:
        explicit = []
        for tok in args.points.split(','):
            ra, pr = tok.split(':')
            explicit.append((float(ra), float(pr)))
    points = pick_points(data, args.n_points, args.min_branches,
                         args.point_seed, explicit)
    if rank == 0:
        print('[gt] points (Ra,Pr, #branches):',
              [(round(k[0]), round(k[1], 2), len(v)) for k, v in points])
        print(f'[gt] sharding {len(points)} points over {world_size} GPU(s)')

    mine = [(i, p) for i, p in enumerate(points) if i % world_size == rank]
    zc = (1.0 - data.z)[None, :]
    per_branch, gallery = {}, []
    for pi, ((Ra, Pr), idxs) in mine:
        refs = {data.rolls[i]: data.fields[i].cpu() for i in idxs}
        f, info = C.pbfm_sample(m, ck, data, Ra, Pr, K=args.K,
                                n_step=args.n_step, seed=args.seed + pi,
                                device=device, E=E,
                                project=args.with_project,
                                hard_cleanup=args.with_cleanup)
        rolls = info['roll'].tolist()
        for k in range(len(rolls)):
            rn = int(rolls[k])
            if rn not in refs:
                continue
            g, r = f[k].cpu(), refs[rn]
            ga = C.align_to(g, r)
            mse, nrmse = C.data_errors(ga, r)
            pl2 = C.physics_L2(ga[0].double(), ga[1].double(),
                               ga[2].double() + zc.double(), Ra, Pr,
                               data.dx, data.dz)
            gl2 = C.physics_L2(r[0].double(), r[1].double(),
                               r[2].double() + zc.double(), Ra, Pr,
                               data.dx, data.dz)
            pb = per_branch.setdefault(rn, {'mse': [], 'nrmse': [],
                                            'phys': [], 'gt': [], 'valid': []})
            pb['mse'].append(mse); pb['nrmse'].append(100 * nrmse)
            pb['phys'].append(pl2); pb['gt'].append(gl2)
            pb['valid'].append(1.0 if float(info['residual'][k]) < thr.get(rn, 1e9)
                               else 0.0)
            if len(gallery) < 3:
                gallery.append((rn, Ra, Pr, ga, r))
        print(f'[gt][gpu{rank}] Ra={Ra:.0f} Pr={Pr:.2f} matched '
              f'{sum(1 for x in rolls if int(x) in refs)}/{len(rolls)} samples',
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'per_branch': per_branch, 'gallery': gallery},
               os.path.join(args.out_dir, f'_gtpart{rank}.pt'))


def merge_and_report(args):
    per_branch, gallery = {}, []
    for p in sorted(glob.glob(os.path.join(args.out_dir, '_gtpart*.pt'))):
        d = torch.load(p, weights_only=False)
        for n, pb in d['per_branch'].items():
            tgt = per_branch.setdefault(n, {'mse': [], 'nrmse': [],
                                            'phys': [], 'gt': [], 'valid': []})
            for k in tgt:
                tgt[k] += pb[k]
        gallery += d['gallery']
        os.remove(p)

    print('\n===== GROUND-TRUTH (conditioned PBFM) =====')
    print(f'{"n":>3} {"count":>6} {"DataMSE":>10} {"NRMSE%":>9} '
          f'{"PhysL2":>10} {"GT PhysL2":>10} {"valid%":>7s}')
    allm, alln, allp, allg, allv = [], [], [], [], []
    csv_rows = []
    for n in sorted(per_branch):
        pb = per_branch[n]
        vp = 100 * np.mean(pb['valid']) if pb.get('valid') else float('nan')
        print(f'{n:>3} {len(pb["nrmse"]):>6} {np.mean(pb["mse"]):>10.3e} '
              f'{np.mean(pb["nrmse"]):>9.3f} {np.mean(pb["phys"]):>10.3e} '
              f'{np.mean(pb["gt"]):>10.3e} {vp:>7.1f}')
        csv_rows.append(dict(branch=f'n={n}', n=n, **{
            'Data MSE': np.mean(pb['mse']), 'Data NRMSE': np.mean(pb['nrmse']),
            'Phys L2': np.mean(pb['phys']), 'GT Phys L2': np.mean(pb['gt']),
            'valid%': vp}))
        allm += pb['mse']; alln += pb['nrmse']; allp += pb['phys']; allg += pb['gt']
        allv += pb.get('valid', [])
    if alln:
        vpa = 100 * np.mean(allv) if allv else float('nan')
        print(f'{"AVG":>3} {len(alln):>6} {np.mean(allm):>10.3e} '
              f'{np.mean(alln):>9.3f} {np.mean(allp):>10.3e} '
              f'{np.mean(allg):>10.3e} {vpa:>7.1f}')
        csv_rows.append(dict(branch='AVG', n='', **{
            'Data MSE': np.mean(allm), 'Data NRMSE': np.mean(alln),
            'Phys L2': np.mean(allp), 'GT Phys L2': np.mean(allg), 'valid%': vpa}))
        try:
            import csv as _csv
            p = os.path.join(args.out_dir, 'gt_metrics.csv')
            with open(p, 'w', newline='') as fh:
                w = _csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
                w.writeheader()
                [w.writerow(r) for r in csv_rows]
            print(f'[gt] table -> {p}')
        except Exception as e:
            print(f'[gt] CSV skipped ({e})')
        print(f'\n[gt] sample Phys L2 / reference floor = '
              f'{np.mean(allp) / max(np.mean(allg), 1e-12):.3f}x')
    else:
        print('  (no samples matched a reference roll number)')

    if HAVE_MPL and gallery:
        os.makedirs(args.out_dir, exist_ok=True)
        data = C.RB2DData(C.find_file(args.bank) or args.bank, device='cpu')
        n = len(gallery)
        fig, axes = plt.subplots(2, n, figsize=(2.4 * n, 4.4), squeeze=False)
        for j, (rn, Ra, Pr, ga, r) in enumerate(gallery):
            for row, (fld, name) in enumerate([(ga, 'gen'), (r, 'ref')]):
                axes[row][j].imshow(fld[1].numpy().T, origin='lower',
                                    aspect='auto', cmap='RdBu_r',
                                    extent=[0, data.aspect, 0, 1])
                axes[row][j].set_yticks([])
                if row == 0:
                    axes[row][j].set_title(f'n={rn}\nRa={Ra:.0f}', fontsize=8)
                if j == 0:
                    axes[row][j].set_ylabel(name)
        fig.suptitle('PBFM generated vs ground-truth u_z', fontsize=10)
        fig.tight_layout()
        fig.savefig(f'{args.out_dir}/fig_gt_gallery.png', dpi=120)
        plt.close(fig)
        print(f'[gt] figure -> {args.out_dir}/fig_gt_gallery.png')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', default='./rb2d/split/train_bank.pt',
                   help='TRAIN bank (in-distribution GT) or '
                        './rb2d/split/heldout_reference.pt for held-out GT')
    p.add_argument('--ckpt', default='ckpt_rb2d_pbfm_cond.pt')
    p.add_argument('--out-dir', dest='out_dir', default='rb2d_pbfm_gt')
    p.add_argument('--gpus', type=int, default=None)
    p.add_argument('--n-points', dest='n_points', type=int, default=4)
    p.add_argument('--min-branches', dest='min_branches', type=int, default=3)
    p.add_argument('--point-seed', dest='point_seed', type=int, default=0)
    p.add_argument('--points', default=None,
                   help='explicit "Ra:Pr,Ra:Pr" list overriding random pick')
    p.add_argument('--k-samples', dest='K', type=int, default=40)
    p.add_argument('--n-step', dest='n_step', type=int, default=50)
    p.add_argument('--with-cleanup', dest='with_cleanup', action='store_true',
                   default=False,
                   help='ABLATION: one Poisson clean-up at sampling time. '
                        'Default is the pure projection-free PBFM path.')
    p.add_argument('--with-project', dest='with_project', action='store_true',
                   default=False, help='ABLATION: PCFM-style IMEX projection')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.out_dir, '_gtpart*.pt')):
        os.remove(f)

    if C.ddp_is_torchrun():
        rank = int(os.environ['RANK']); world = int(os.environ['WORLD_SIZE'])
        C.ddp_setup(rank, world)
        worker(rank, world, args)
        import torch.distributed as dist
        dist.barrier()
        if rank == 0:
            merge_and_report(args)
        C.ddp_cleanup()
        return

    gpus = args.gpus
    if gpus is None:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpus = max(1, int(gpus))
    if gpus > 1:
        import torch.multiprocessing as mp
        mp.spawn(worker, args=(gpus, args), nprocs=gpus, join=True)
    else:
        worker(0, 1, args)
    merge_and_report(args)


if __name__ == '__main__':
    main()