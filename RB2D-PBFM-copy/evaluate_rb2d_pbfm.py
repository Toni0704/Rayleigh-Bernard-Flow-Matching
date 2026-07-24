#!/usr/bin/env python3
"""
evaluate_rb2d_pbfm.py
================================================================================
Main sweep evaluation for the CONDITIONED RB2D PBFM model. Reports exactly the
metrics the paper needs:

  * physics validity %      (combined residual <= per-branch CALIBRATED threshold)
  * mean residual and residual / reference-floor ratio
  * mode coverage           (fraction of {2,3,4,5,6} reached as valid samples)
  * roll-number entropy      (branch diversity, bits)
  * per-point breakdown

PBFM HAS ONE SAMPLING PATH. Physics is imposed during TRAINING, so sampling is
a plain ODE integration -- there is nothing to switch off at inference and hence
no "vanilla mode" of a PBFM model. A default run therefore produces exactly ONE
row: the PBFM result.

The vanilla and PCFM rows of the paper table come from the PCFM pipeline
(rb2d_eval/metrics_summary.csv), where 'vanilla' means that same plain-FM
checkpoint sampled WITHOUT the projection. This script uses the identical sweep
points, K, thresholds and metric definitions as evaluate_rb2d_pcfm.py, so its
CSV concatenates directly onto that one.

Optional ablations, both OFF by default and never part of the headline number:
  --with-cleanup : one Poisson clean-up after sampling
  --with-project : PCFM-style IMEX projection applied to the PBFM model

Per-branch acceptance thresholds are CALIBRATED from the bank by default
(1.5 x q99 of per-branch reference residuals) -- a fixed absolute threshold is
meaningless because the RB residual has an intrinsic, Ra-dependent floor.

--------------------------------------------------------------------------
MULTI-GPU (Kaggle 2 x T4)
--------------------------------------------------------------------------
Sweep points and held-out points are sharded across GPUs (embarrassingly
parallel -- each point is an independent batch of samples). Each rank writes a
partial result file; the parent merges them and prints one report.

    python evaluate_rb2d_pbfm.py --bank ./rb2d/split/train_bank.pt \
        --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2

This matters most with --with-project, where the IMEX relaxation dominates
runtime; the projection-free default path is already fast.

Examples:
    python evaluate_rb2d_pbfm.py --bank ./rb2d/split/train_bank.pt \
        --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2

    python evaluate_rb2d_pbfm.py --bank ./rb2d/split/train_bank.pt \
        --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2 --cal-ra-max 8000 \
        --heldout-spec ./rb2d/split/heldout_test_spec.pt

Then score the export with your existing harness:
    python heldout_branch_test.py --mode score --bank <refs_bank.pt> \
        --out-dir ./rb2d/split --samples rb2d_pbfm_eval/heldout_model_samples.pt -v
"""
import argparse
import glob
import math
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

TAG = 'pbfm-cond'


# ---------------------------------------------------------------- sweep points
def logspace_points(data, n=6):
    """VERBATIM from evaluate_rb2d_pcfm.pick_eval_params -- log-spaced diagonal
    through the (Ra,Pr) box interior (10-90%). Must match exactly, otherwise the
    PBFM row and the vanilla/PCFM rows in the paper table are measured at
    different parameters and cannot be compared."""
    lra = np.linspace(math.log(data.Ra_range[0]), math.log(data.Ra_range[1]), n + 2)[1:-1]
    lpr = np.linspace(math.log(data.Pr_range[0]), math.log(data.Pr_range[1]), n + 2)[1:-1]
    return [(float(np.exp(a)), float(np.exp(b))) for a, b in zip(lra, lpr[::-1])]


def resolve_device(rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return f'cuda:{rank % torch.cuda.device_count()}'
    return 'cpu'


def sampling_kwargs(mode):
    """'pure' is the headline PBFM path: plain ODE, NO projection, NO clean-up.
    '+clean' and '+imex' are inference-time ABLATIONS, not the PBFM result."""
    if mode == '+clean':
        return dict(project=False, hard_cleanup=True)
    if mode == '+imex':
        return dict(project=True, hard_cleanup=False)
    return dict(project=False, hard_cleanup=False)      # 'pure'


# ---------------------------------------------------------------- worker
def worker(rank, world_size, args):
    device = resolve_device(rank)
    data = C.RB2DData(C.find_file(args.bank) or args.bank, device=device)
    E = C.build_eig(data.Nx, data.Nz, data.dx, data.dz, device, torch.float64)
    models = [('pbfm', C.find_file(args.ckpt) or args.ckpt)]
    if args.baseline_ckpt:      # optional, off by default
        bp = C.find_file(args.baseline_ckpt) or args.baseline_ckpt
        if os.path.exists(bp):
            models.append(('fm-baseline', bp))
        elif rank == 0:
            print(f'[eval] vanilla checkpoint {args.baseline_ckpt} not found; '
                  f'skipping the vanilla row (point --vanilla-ckpt at the PCFM '
                  f'plain-FM checkpoint, e.g. ckpt_rb2d_cond.pt)')

    modes = ['pure']
    if args.with_cleanup:
        modes.append('+clean')
    if args.with_project:
        modes.append('+imex')

    all_points = logspace_points(data, args.n_points)
    if args.snap_to_bank:
        uniq = list(data.unique_points().keys())
        snapped = []
        for (Ra, Pr) in all_points:
            b = min(uniq, key=lambda k: (math.log(k[0]/Ra))**2 + (math.log(k[1]/Pr))**2)
            snapped.append(b)
        if rank == 0:
            print('[eval] sweep snapped to nearest bank points so NRMSE has an '
                  'exact reference (A.8); disable with --no-snap-to-bank')
        all_points = snapped
    my_points = [(i, p) for i, p in enumerate(all_points)
                 if i % world_size == rank]
    if rank == 0:
        print(f'[eval] sweep points (Ra,Pr): '
              f'{[(round(a), round(b, 2)) for a, b in all_points]}')
        print(f'[eval] sharding {len(all_points)} points over {world_size} GPU(s)')

    part = {'sweep': {}, 'heldout': {}}
    for mtag, mpath in models:
        m, ck = C.load_model(mpath, device=device)
        for mode in modes:
            if mtag == 'fm-baseline' and mode != 'pure':
                continue                       # baseline only needs the pure row
            kw = sampling_kwargs(mode)
            for pi, (Ra, Pr) in my_points:
                f, info = C.pbfm_sample(m, ck, data, Ra, Pr, K=args.K,
                                        n_step=args.n_step, seed=args.seed + pi,
                                        device=device, E=E, **kw)
                # symmetry-aware NRMSE vs the reference of matching roll (A.8)
                nrmse = []
                _, refs = data.refs_at(Ra, Pr)
                zc1 = (1.0 - data.z)[None, :]
                for k, rn in enumerate(info['roll'].tolist()):
                    r = refs.get(int(rn))
                    if r is None:
                        nrmse.append(float('nan')); continue
                    ga = C.align_to(f[k].cpu(), r.cpu())
                    nrmse.append(100.0 * C.data_errors(ga, r.cpu())[1])
                part['sweep'][(mtag, mode, pi)] = dict(
                    Ra=Ra, Pr=Pr,
                    rolls=[int(x) for x in info['roll'].tolist()],
                    res=[float(x) for x in info['residual'].tolist()],
                    nrmse=nrmse, secs=float(info['seconds']),
                    secs_ode=float(info['seconds_ode']))
                print(f'[eval][gpu{rank}] {mtag:11s} {mode:7s} '
                      f'Ra={Ra:.0f} Pr={Pr:.2f} done', flush=True)

    # held-out export shard
    if args.heldout_spec:
        spec = torch.load(C.find_file(args.heldout_spec) or args.heldout_spec,
                          weights_only=False)
        params = spec['params']
        hmode = args.heldout_mode
        kw = sampling_kwargs(hmode)
        m, ck = C.load_model(models[0][1], device=device)
        mine = [(i, s) for i, s in enumerate(params) if i % world_size == rank]
        for i, s in mine:
            Ra, Pr = float(s['Ra']), float(s['Pr'])
            f, info = C.pbfm_sample(m, ck, data, Ra, Pr, K=args.heldout_k,
                                    n_step=args.n_step, seed=args.seed + i,
                                    device=device, E=E, **kw)
            # (K,3,Nx,Nz) -> (K,Nx,Nz,3) channels (u,w,T') as the scorer expects
            part['heldout'][(Ra, Pr)] = f.permute(0, 2, 3, 1).contiguous().cpu()
            print(f'[eval][gpu{rank}] heldout Ra={Ra:.0f} Pr={Pr:.2f} '
                  f'rolls={sorted(set(info["roll"].tolist()))}', flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(part, os.path.join(args.out_dir, f'_part{rank}.pt'))


# ---------------------------------------------------------------- merge/report
def merge_and_report(args):
    device = resolve_device(0)
    data = C.RB2DData(C.find_file(args.bank) or args.bank, device=device)
    E = C.build_eig(data.Nx, data.Nz, data.dx, data.dz, device, torch.float64)

    thr, floor = C.calibrate_thresholds(
        data, mult=args.cal_mult, q=args.cal_q,
        ra_min=args.cal_ra_min, ra_max=args.cal_ra_max)
    if args.resid_tol is not None:
        print(f'[eval] WARNING: overriding calibrated thresholds with fixed '
              f'resid-tol={args.resid_tol}. The RB residual floor is '
              f'Ra-dependent; a fixed tol is usually too strict at high Ra.')
        thr = {n: args.resid_tol for n in data.n_list}
    print('[eval] calibrated per-branch thresholds (mult x quantile):')
    for n in sorted(thr):
        print(f'    n={n}  floor={floor.get(n, float("nan")):.3f}  thr={thr[n]:.3f}')

    sweep, heldout = {}, {}
    for p in sorted(glob.glob(os.path.join(args.out_dir, '_part*.pt'))):
        d = torch.load(p, weights_only=False)
        sweep.update(d['sweep']); heldout.update(d['heldout'])
        os.remove(p)
    if not sweep:
        raise SystemExit('[eval] no partial results found.')

    combos = sorted({(mt, md) for (mt, md, _) in sweep},
                    key=lambda x: ({'pbfm': 0, 'fm-baseline': 1}.get(x[0], 9),
                                   {'pure': 0, '+clean': 1, '+imex': 2}.get(x[1], 9)))
    rows, per_point = [], {}
    fref = floor.get(min(floor), 0.15) if floor else 0.15
    for mtag, mode in combos:
        rolls_all, res_all, ratio_all, valids, covers = [], [], [], [], []
        valid_rolls, valid_nrmse, secs_pp, secs_ode_pp = [], [], [], []
        for (mt, md, pi), d in sorted(sweep.items()):
            if (mt, md) != (mtag, mode):
                continue
            branches, nvalid = set(), 0
            nr_list = d.get('nrmse', [float('nan')] * len(d['rolls']))
            for rn, rv, nr in zip(d['rolls'], d['res'], nr_list):
                t = thr.get(int(rn))
                if t is not None and rv < t:            # A.5: strict <
                    nvalid += 1
                    branches.add(int(rn))
                    valid_rolls.append(int(rn))         # A.7: VALID samples only
                    if nr == nr:
                        valid_nrmse.append(nr)
            K = len(d['rolls'])
            ratios = [rv / max(floor.get(int(rn), fref), 1e-9)
                      for rn, rv in zip(d['rolls'], d['res'])]
            rolls_all += d['rolls']; res_all += d['res']; ratio_all += ratios
            secs_pp.append(d.get('secs', float('nan')) / max(K, 1))
            secs_ode_pp.append(d.get('secs_ode', float('nan')) / max(K, 1))
            valids.append(nvalid / max(K, 1))
            cover = len(branches & set(data.n_list)) / len(data.n_list)
            covers.append(cover)
            per_point[(mtag, mode, pi)] = dict(Ra=d['Ra'], Pr=d['Pr'], nvalid=nvalid,
                                         K=K, branches=sorted(branches),
                                         cover=cover, rolls=d['rolls'],
                                         res=d['res'])
        # coverage: mean of PER-POINT coverage over valid samples -- identical to
        # evaluate_rb2d_pcfm.py (cov_frac), so the rows concatenate with PCFM's CSV
        pooled_cov = len(set(valid_rolls) & set(data.n_list)) / len(data.n_list)
        all_nr = [x for x in
                  (n for (mt, md, pi), dd in sweep.items() if (mt, md) == (mtag, mode)
                   for n in dd.get('nrmse', [])) if x == x]
        rows.append(dict(
            model='cond',
            sampling={'pbfm': 'PBFM', 'fm-baseline': 'vanilla',
                      'vanilla': 'vanilla'}.get(mtag, mtag),
            path=mode,
            # --- columns identical to evaluate_rb2d_pcfm.py ---
            valid_pct=100 * float(np.mean(valids)),
            mean_resid=float(np.mean(res_all)),
            resid_over_floor=float(np.median(ratio_all)),
            mode_coverage=float(np.mean(covers)),
            roll_entropy=C.roll_entropy(valid_rolls),
            sec_per_sample=float(np.nanmean(secs_pp)) if secs_pp else float('nan'),
            sec_per_sample_ode=float(np.nanmean(secs_ode_pp)) if secs_ode_pp else float('nan'),
            # --- additional columns required by the eval spec A.9 ---
            rho_median=float(np.median(res_all)),
            coverage_pooled=pooled_cov,
            nrmse_valid=float(np.mean(valid_nrmse)) if valid_nrmse else float('nan'),
            nrmse_all=float(np.mean(all_nr)) if all_nr else float('nan')))

    print('\n=========================== SWEEP SUMMARY (spec A.9) '
          '===========================')
    print(f'{"model":6s} {"sampling":8s} {"valid%":>7s} {"mean_res":>9s} '
          f'{"rho(med)":>9s} {"res/floor":>9s} {"cover":>6s} {"H(nats)":>8s} '
          f'{"s/sample":>9s} {"NRMSE%":>8s}')
    for r in rows:
        nr = r['nrmse_valid'] if r['nrmse_valid'] == r['nrmse_valid'] else r['nrmse_all']
        nrs = f'{nr:8.2f}' if nr == nr else '     n/a'
        print(f'{r["model"]:6s} {r["sampling"]:8s} {r["valid_pct"]:7.2f} '
              f'{r["mean_resid"]:9.4f} {r["rho_median"]:9.4f} '
              f'{r["resid_over_floor"]:9.3f} {r["mode_coverage"]:6.2f} '
              f'{r["roll_entropy"]:8.4f} {r["sec_per_sample"]:9.4f} {nrs}')
    print('===================================================================='
          '=========')
    print(f'  rho = max_j rms(R_j)/rms(D_j)   |   H max = ln(5) = '
          f'{math.log(len(data.n_list)):.4f} nats')
    print('  NRMSE% is over VALID samples (falls back to all samples if none valid)')

    print('\nPer-point (PBFM, pure ODE sampling):')
    for (mtag, mode, pi), d in sorted(per_point.items()):
        if (mtag, mode) != ('pbfm', 'pure'):
            continue
        print(f'  Ra={d["Ra"]:.0f} Pr={d["Pr"]:.2f}: {d["nvalid"]}/{d["K"]} '
              f'valid, branches={d["branches"]}, coverage={d["cover"]:.2f}')

    try:
        import json, csv
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as fh:
            json.dump(rows, fh, indent=2)
        csv_path = os.path.join(args.out_dir, 'rb2d_summary.csv')
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'[eval] table -> {csv_path}')
    except Exception as e:
        print(f'[eval] CSV write skipped ({e})')

    if heldout:
        out = os.path.join(args.out_dir, 'heldout_model_samples.pt')
        torch.save(heldout, out)
        print(f'\n[heldout] {len(heldout)} points -> {out}\n'
              f'  score with: python heldout_branch_test.py --mode score '
              f'--bank <refs_bank.pt> --out-dir ./rb2d/split --samples {out} -v')

    make_figures(args, rows, per_point, data, floor, device, E)


# ---------------------------------------------------------------- figures
def make_figures(args, rows, per_point, data, floor, device, E):
    if not HAVE_MPL:
        print('[figures] matplotlib unavailable; skipping')
        return
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    labels = [f"{r['model']}\n{r['sampling']}" for r in rows]
    ax[0].bar(labels, [r['valid_pct'] for r in rows], color='tab:green')
    ax[0].set_ylabel('valid %'); ax[0].set_title('Physics validity'); ax[0].set_ylim(0, 105)
    ax[1].bar(labels, [100 * r['mode_coverage'] for r in rows], color='tab:blue')
    ax[1].set_ylabel('mode coverage %'); ax[1].set_title('Branch coverage'); ax[1].set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(f'{out_dir}/fig_validity_coverage.png', dpi=120)
    plt.close(fig)

    fref = floor.get(min(floor), 0.15) if floor else 0.15
    box, box_lbl = [], []
    for (mtag, mode, pi), d in sorted(per_point.items()):
        if (mtag, mode) != ('pbfm', 'pure'):
            continue
        box.append([rv / max(floor.get(int(rn), fref), 1e-9)
                    for rn, rv in zip(d['rolls'], d['res'])])
        box_lbl.append(f'Ra={d["Ra"]:.0f}')
    if box:
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(box)), 4))
        ax.boxplot(box, showfliers=False)
        ax.set_xticks(range(1, len(box_lbl) + 1))
        ax.set_xticklabels(box_lbl, fontsize=8, rotation=30)
        ax.axhline(1.0, color='k', ls='--', lw=1, label='reference floor')
        ax.set_ylabel('residual / floor')
        ax.set_title('PBFM floor-relative residual'); ax.legend()
        fig.tight_layout(); fig.savefig(f'{out_dir}/fig_resid_over_floor.png', dpi=120)
        plt.close(fig)

    from collections import Counter
    c = Counter()
    for (mtag, mode, pi), d in per_point.items():
        if (mtag, mode) == ('pbfm', 'pure'):
            c.update(d['rolls'])
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.array(data.n_list)
    ax.bar(xs, [c.get(int(n), 0) for n in xs], color='tab:purple')
    ax.set_xticks(xs); ax.set_xlabel('roll number n'); ax.set_ylabel('count')
    ax.set_title('Roll-number distribution (PBFM)')
    fig.tight_layout(); fig.savefig(f'{out_dir}/fig_roll_hist.png', dpi=120); plt.close(fig)

    # field gallery
    try:
        m, ck = C.load_model(C.find_file(args.ckpt) or args.ckpt, device=device)
        pts = logspace_points(data, args.n_points)
        Ra, Pr = pts[len(pts) // 2]
        f, info = C.pbfm_sample(m, ck, data, Ra, Pr, K=40, n_step=args.n_step,
                                seed=123, device=device, E=E,
                                project=False, hard_cleanup=False)
        picks = {}
        for i, rn in enumerate(info['roll'].tolist()):
            if int(rn) in data.n_list and int(rn) not in picks:
                picks[int(rn)] = i
        if picks:
            ncol = len(picks)
            fig, axes = plt.subplots(1, ncol, figsize=(2.6 * ncol, 2.4))
            if ncol == 1:
                axes = [axes]
            for ax_, n in zip(axes, sorted(picks)):
                uz = f[picks[n], 1].cpu().numpy().T
                ax_.imshow(uz, origin='lower', aspect='auto', cmap='RdBu_r',
                           extent=[0, data.aspect, 0, 1])
                ax_.set_title(f'n={n}', fontsize=9); ax_.set_yticks([])
            fig.suptitle(f'PBFM u_z samples  Ra={Ra:.0f} Pr={Pr:.2f}', fontsize=10)
            fig.tight_layout(); fig.savefig(f'{out_dir}/fig_field_gallery.png', dpi=120)
            plt.close(fig)
    except Exception as e:
        print(f'[figures] gallery skipped ({e})')

    rp = C.find_file(os.path.splitext(os.path.basename(args.ckpt))[0] + '_resume.pt')
    if rp and os.path.exists(rp):
        try:
            h = torch.load(rp, map_location='cpu', weights_only=False)['history']
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(h['it'], h['fm'], label='train FM', lw=1)
            v = [e for e in h.get('val', []) if len(e) >= 2]
            if v:
                ax[0].plot([e[0] for e in v], [e[1] for e in v],
                           'o-', ms=3, label='val FM (EMA, fixed set)')
            ax[0].set_yscale('log'); ax[0].set_xlabel('iter')
            ax[0].set_ylabel('flow-matching loss'); ax[0].legend()
            ax[0].set_title('Generative objective')
            ax[1].plot(h['it'], h['res'], label='train residual', lw=1)
            vr = [e for e in v if len(e) >= 3 and e[2] == e[2]]
            if vr:
                ax[1].plot([e[0] for e in vr], [e[2] for e in vr],
                           'o-', ms=3, label='val residual (EMA)')
            ax[1].set_yscale('log'); ax[1].set_xlabel('iter')
            ax[1].set_ylabel('combined steady residual'); ax[1].legend()
            ax[1].set_title('Physics objective (PBFM)')
            fig.suptitle('Conditioned PBFM training', fontsize=11)
            fig.tight_layout(); fig.savefig(f'{out_dir}/fig_loss.png', dpi=120)
            plt.close(fig)
        except Exception:
            pass
    print(f'[figures] written to {out_dir}/')


# ---------------------------------------------------------------- cli
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', default='./rb2d/split/train_bank.pt',
                   help='TRAIN bank written by prepare_rb2d_pbfm_data.py')
    p.add_argument('--ckpt', default='ckpt_rb2d_pbfm_cond.pt',
                   help='conditioned PBFM checkpoint')
    p.add_argument('--out-dir', dest='out_dir', default='rb2d_pbfm_eval')
    p.add_argument('--gpus', type=int, default=None,
                   help='number of GPUs to shard the sweep over (default: all)')
    p.add_argument('--k-samples', dest='K', type=int, default=40)
    p.add_argument('--n-step', dest='n_step', type=int, default=50)
    p.add_argument('--n-points', dest='n_points', type=int, default=6)
    p.add_argument('--vanilla-ckpt', dest='baseline_ckpt', default=None,
                   help='OPTIONAL. Not needed: the vanilla row already exists in '
                        'rb2d_eval/metrics_summary.csv from the PCFM run, and this '
                        'script now uses the identical sweep points, K and metric '
                        'definitions, so the two CSVs concatenate directly. Only '
                        'pass this if you want vanilla re-measured in one go.')
    p.add_argument('--with-cleanup', dest='with_cleanup', action='store_true',
                   default=False,
                   help='ABLATION: add one Poisson clean-up at sampling time')
    p.add_argument('--with-project', dest='with_project', action='store_true',
                   default=False,
                   help='ABLATION: PCFM-style IMEX projection (slow)')
    p.add_argument('--snap-to-bank', dest='snap_to_bank', action='store_true',
                   default=False,
                   help='snap sweep points to the nearest bank point so the NRMSE '
                        'column has an exact reference. OFF by default because it '
                        'moves the evaluation parameters away from the PCFM sweep '
                        'and would break comparability with the vanilla/PCFM rows.')
    p.add_argument('--heldout-mode', dest='heldout_mode', default='pure',
                   choices=['pure', '+clean', '+imex'],
                   help='sampling path used for the held-out export')
    p.add_argument('--cal-mult', dest='cal_mult', type=float, default=1.5)
    p.add_argument('--cal-q', dest='cal_q', type=float, default=0.99)
    p.add_argument('--cal-ra-min', dest='cal_ra_min', type=float, default=0.0)
    p.add_argument('--cal-ra-max', dest='cal_ra_max', type=float, default=float('inf'))
    p.add_argument('--resid-tol', dest='resid_tol', type=float, default=None,
                   help='OVERRIDE calibrated thresholds with a fixed value')
    p.add_argument('--heldout-spec', dest='heldout_spec', default=None)
    p.add_argument('--heldout-k', dest='heldout_k', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(args.out_dir, '_part*.pt')):
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