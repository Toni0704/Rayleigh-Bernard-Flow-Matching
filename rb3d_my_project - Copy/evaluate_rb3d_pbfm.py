#!/usr/bin/env python3
"""
evaluate_rb3d_pbfm.py
================================================================================
Standalone evaluation for the CONDITIONED RB3D PBFM model. Reports exactly the
metric suite of evaluation_and_paper_spec.md Part A for the PBFM row -- no
dependency on any PCFM/vanilla checkpoint (that comparison lives in
evaluate_rb3d.py once those are trained; this script never loads them).

PBFM has ONE sampling path. Per spec A.1/A.11: physics is enforced at TRAINING
time via the residual loss (rb3d_pbfm_common.py), so sampling is a plain ODE
integration with NO projection -- "vanilla and PBFM are sampled identically
(no projection); they differ only in the trained checkpoint." There is
nothing to switch off at inference and hence nothing to ablate for the
headline number.

Reuses evaluate_rb3d.py's verifier-matching `Residual` class,
`calibrate_thresholds` and `median_floor_by_planform` UNCHANGED, so if/when a
PCFM row is added later the two are scored by the identical evaluator and the
CSVs concatenate directly, per spec A.9's "one schema, all configs" rule.

Suites (from prepare_rb3d_splits.py; same test-bank convention evaluate_rb3d.py
uses, so PBFM is evaluated at the IDENTICAL points a PCFM run would be):
    gt      -- Test-GT: references exist -> physics + validity + NRMSE
    drift   -- Test-drift: reorganized-branch references -> physics + NRMSE
    heldout -- Test-heldout: no references -> physics + coverage only

Per suite, writes into <out>/<suite>/:
    metrics.csv / metrics.json   -- per-planform + AVG breakdown
    paper_table_a9.csv           -- spec A.9 schema, ONE row (model=cond,
                                     method=PBFM): rho_median, rho_over_floor,
                                     valid_pct, coverage, entropy_nats, nrmse_pct
    fig_validity.png, fig_resid_over_floor.png, fig_roll_hist.png

USAGE
    python evaluate_rb3d_pbfm.py --splits /kaggle/working/rb3d_splits \
        --ckpt ckpt_rb3d_pbfm_cond.pt --suite all --k-samples 24
    python evaluate_rb3d_pbfm.py --suite gt --quick        # fast smoke

MULTI-GPU (optional, --gpus): each (Ra,Pr) point is independent, so points are
sharded round-robin across ranks (mp.spawn), matching the worker/merge pattern
already used by evaluate_rb2d_pbfm.py and evaluate_rb3d.py's --dual-gpu.
"""

import argparse
import glob
import math
import os
import sys
import time
from collections import defaultdict, Counter

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rb3d_pcfm_common import RB3DData, load_flow_model
from rb3d_pcfm_sampler import PTCProjector, pcfm_sample
from rb3d_classify import classify_purity

from evaluate_rb3d import (Residual, calibrate_thresholds,
                          median_floor_by_planform, align_to, data_errors,
                          _ckpt_load, _ckpt_append, _ckpt_clear, physics_L2)

# Purity gate: labels below PURITY_MIN become ('unknown',0,0) instead of a
# forced guess -- matches evaluate_rb3d.py's own convention exactly so a
# sample is classified the same way regardless of which script scores it.
PURITY_MIN = 0.90


def _classify1(field, aspect):
    lab, _pur, _ = classify_purity(field[2], aspect[0], aspect[1],
                                   purity_min=PURITY_MIN)
    return lab


def resolve_device(rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return f'cuda:{rank % torch.cuda.device_count()}'
    return 'cpu'


# ============================================================================
#  Sampling: the ONE PBFM path (plain ODE flow, no projection)
# ============================================================================
def draw_pbfm(model, ck, data, projector, resid, Ra, Pr, k, n_step, batch,
              seed0):
    """Draw k samples via pcfm_sample(..., vanilla=True) -- exactly the PBFM
    headline path (spec A.1). `projector` is passed through only because
    pcfm_sample's signature needs one for a cheap post-hoc residual read
    (`projector.res(...)`); with vanilla=True it never calls `.project()`."""
    out = []
    done = 0
    while done < k:
        b = min(batch, k - done)
        Ra_l, Pr_l = [Ra] * b, [Pr] * b
        f, _ = pcfm_sample(model, ck, data, Ra_l, Pr_l, projector,
                           n_step=n_step, seed=seed0 + done, vanilla=True)
        for j in range(b):
            field = f[j]
            rm, rt, rc = resid(field, Ra, Pr)
            out.append(dict(field=field, pf=tuple(_classify1(field, data.aspect)),
                            res=(rm, rt, rc)))
        done += b
    return out


def _score(rec, pf, rm, rt, rc, thr):
    # key must END IN '_pf' -- evaluate_rb3d.py's _ckpt_load only restores
    # tuple-vs-list for keys matching that suffix when replaying a resumed
    # JSONL checkpoint; a bare 'pf' key would silently stay a list after
    # resume and crash Counter()/dict-keying downstream (unhashable list).
    rec['pbfm_pf'] = pf
    rec['mom'], rec['temp'], rec['cont'] = rm, rt, rc
    rec['rho'] = max(rm, rt, rc)                  # spec A.4: worst-of-equations
    rec['physL2'] = physics_L2(rm, rt, rc)
    rec['valid'] = bool(max(rm, rt) < thr)         # matches calibrate_thresholds' gate
    return rec


# ============================================================================
#  Reference banks (Test-GT / Test-drift)
# ============================================================================
def _refs_by_point(bank, data):
    refs = defaultdict(dict)
    for (Ra, Pr, mode), e in bank['entries'].items():
        pf = tuple(e['planform'])
        Tp = e['grid_T'].float() - (1.0 - data.z)[None, None, :]
        refs[(Ra, Pr)][pf] = torch.stack(
            [e['grid_u'].float(), e['grid_v'].float(), e['grid_w'].float(), Tp])
    return refs


def worker_reference(rank, world_size, args, tag, bank_name, data):
    """`data` is a single RB3DData built ONCE in main() with its big tensors
    share_memory_()'d, not re-loaded per rank -- see main()'s comment. Each
    rank still needs data.fields for calibrate_thresholds, and the ~8.6GB bank
    plus a ~15-17GB transient during RB3DData's own construction is exactly
    what SIGKILLed a 31GB-RAM box during PBFM training (rb3d_pbfm_action_plan.md);
    re-loading independently per eval worker would hit the identical ceiling."""
    device = resolve_device(rank)
    data.device = device
    projector = PTCProjector(data.Nx, data.Ny, data.Nz, data.aspect, device)
    resid = Residual(data.Nx, data.Ny, data.Nz, data.aspect)
    thresholds, floor = calibrate_thresholds(data, resid, mult=args.cal_mult,
                                             q=args.cal_q)
    model, ck = load_flow_model(args.ckpt, device)

    bank = torch.load(os.path.join(args.splits, bank_name), map_location='cpu',
                      weights_only=False)
    refs = _refs_by_point(bank, data)
    points = sorted(refs)
    done_keys, prior = _ckpt_load(args.out_dir, tag)
    mine = [(i, p) for i, p in enumerate(points)
           if i % world_size == rank and ('pbfm', *p) not in done_keys]
    if rank == 0:
        print(f'[{tag}] {len(points)} point(s), '
              f'{sum(len(r) for r in refs.values())} references')

    part = list(prior.get('pbfm', [])) if rank == 0 else []
    for pi, (Ra, Pr) in mine:
        samples = draw_pbfm(model, ck, data, projector, resid, Ra, Pr,
                            args.k_samples, args.n_step, args.batch, 1000 * pi)
        rr = refs[(Ra, Pr)]
        recs = []
        for s in samples:
            rec = dict(Ra=Ra, Pr=Pr)
            rm, rt, rc = s['res']
            thr = thresholds.get(s['pf'], floor)
            _score(rec, s['pf'], rm, rt, rc, thr)
            rec['nrmse'] = float('nan')
            if s['pf'] in rr:
                ga, _ = align_to(s['field'], rr[s['pf']])
                _, nr = data_errors(ga, rr[s['pf']])
                rec['nrmse'] = nr
            recs.append(rec)
        _ckpt_append(args.out_dir, tag, 'pbfm', Ra, Pr, recs)
        part += recs
        nvalid = sum(r['valid'] for r in recs)
        print(f'[eval][gpu{rank}] {tag} Ra={Ra:.0f} Pr={Pr:.2f}  '
              f'valid={nvalid}/{args.k_samples}', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(dict(records=part, thresholds=thresholds, floor=floor),
              os.path.join(args.out_dir, f'_part_{tag}_{rank}.pt'))


# ============================================================================
#  Test-heldout (no references -> physics + coverage only)
# ============================================================================
def worker_heldout(rank, world_size, args, data):
    """See worker_reference's docstring: `data` is shared, not re-loaded."""
    device = resolve_device(rank)
    data.device = device
    projector = PTCProjector(data.Nx, data.Ny, data.Nz, data.aspect, device)
    resid = Residual(data.Nx, data.Ny, data.Nz, data.aspect)
    thresholds, floor = calibrate_thresholds(data, resid, mult=args.cal_mult,
                                             q=args.cal_q)
    model, ck = load_flow_model(args.ckpt, device)

    bank = torch.load(os.path.join(args.splits, 'test_heldout_bank.pt'),
                      map_location='cpu', weights_only=False)
    points = sorted({(k[0], k[1]) for k in bank['entries']})
    if args.max_heldout_points:
        points = points[:args.max_heldout_points]
    done_keys, prior = _ckpt_load(args.out_dir, 'heldout')
    mine = [(i, p) for i, p in enumerate(points)
           if i % world_size == rank and ('pbfm', *p) not in done_keys]
    if rank == 0:
        print(f'[heldout] {len(points)} unconverged point(s) '
              f'(physics-residual scoring only)')

    part = list(prior.get('pbfm', [])) if rank == 0 else []
    for pi, (Ra, Pr) in mine:
        samples = draw_pbfm(model, ck, data, projector, resid, Ra, Pr,
                            args.k_samples, args.n_step, args.batch, 7000 * pi)
        recs = []
        for s in samples:
            rec = dict(Ra=Ra, Pr=Pr)
            rm, rt, rc = s['res']
            thr = thresholds.get(s['pf'], floor)
            _score(rec, s['pf'], rm, rt, rc, thr)
            recs.append(rec)
        _ckpt_append(args.out_dir, 'heldout', 'pbfm', Ra, Pr, recs)
        part += recs
        nvalid = sum(r['valid'] for r in recs)
        print(f'[eval][gpu{rank}] heldout Ra={Ra:.0f} Pr={Pr:.2f}  '
              f'valid={nvalid}/{args.k_samples}', flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(dict(records=part, thresholds=thresholds, floor=floor),
              os.path.join(args.out_dir, f'_part_heldout_{rank}.pt'))


# ============================================================================
#  Merge partial results + write metrics/figures for one suite
# ============================================================================
def merge_and_report(args, tag, n_branches, with_nrmse):
    parts = sorted(glob.glob(os.path.join(args.out_dir, f'_part_{tag}_*.pt')))
    if not parts:
        print(f'[{tag}] no partial results -- suite skipped or bank missing')
        return None
    records, thresholds, floor = [], {}, 0.3
    for p in parts:
        d = torch.load(p, map_location='cpu', weights_only=False)
        records += d['records']
        thresholds, floor = d['thresholds'], d['floor']
        os.remove(p)
    if not records:
        print(f'[{tag}] no samples drawn')
        return None

    outdir = os.path.join(args.out_dir, tag)
    os.makedirs(outdir, exist_ok=True)

    # -------- per-planform + AVG metrics.csv/json --------
    by_pf = defaultdict(list)
    for r in records:
        by_pf[r['pbfm_pf']].append(r)

    def _line(label, sub):
        nv = [r['nrmse'] for r in sub if not math.isnan(r.get('nrmse', float('nan')))]
        return dict(planform=label, n=len(sub),
                   physL2=round(float(np.mean([r['physL2'] for r in sub])), 4),
                   rho_median=round(float(np.median([r['rho'] for r in sub])), 4),
                   valid_pct=round(100 * float(np.mean([r['valid'] for r in sub])), 1),
                   mom=round(float(np.mean([r['mom'] for r in sub])), 4),
                   temp=round(float(np.mean([r['temp'] for r in sub])), 4),
                   cont=round(float(np.mean([r['cont'] for r in sub])), 4),
                   nrmse_pct=(round(100 * float(np.mean(nv)), 2) if nv else float('nan')))

    rows = [_line(str(pf), sub) for pf, sub in sorted(by_pf.items())]
    rows.append(_line('AVG', records))
    hdr = ['planform', 'n', 'physL2', 'rho_median', 'valid_pct', 'mom', 'temp',
          'cont', 'nrmse_pct']
    with open(os.path.join(outdir, 'metrics.csv'), 'w') as fh:
        fh.write(','.join(hdr) + '\n')
        for row in rows:
            fh.write(','.join(str(row[h]) for h in hdr) + '\n')
    import json as _json
    _json.dump(rows, open(os.path.join(outdir, 'metrics.json'), 'w'), indent=2)
    print(f'  [{tag}] wrote metrics.csv / metrics.json ({len(records)} samples)')

    # -------- spec A.9 table: ONE row (model=cond, method=PBFM) --------
    rhos = [r['rho'] for r in records]
    valid = [r['valid'] for r in records]
    # `floor` (from calibrate_thresholds) is the SCALAR fallback behind the
    # 1.5*q99 threshold, not spec A.5's per-branch reference floor -- for
    # rho/floor we need median_floor_by_planform's per-branch dict instead.
    floor_by_pf = median_floor_by_planform(_data_cache['data'], _data_cache['resid'])
    ratios = [r['rho'] / max(floor_by_pf.get(r['pbfm_pf'], 1e-9), 1e-9) for r in records]
    valid_pfs = [r['pbfm_pf'] for r in records if r['valid']]
    cnt = Counter(valid_pfs)
    tot = sum(cnt.values())
    ent = -sum((c / tot) * math.log(c / tot) for c in cnt.values()) if tot else 0.0
    nrmse_pct = float('nan')
    if with_nrmse:
        nv = [r['nrmse'] for r in records if not math.isnan(r.get('nrmse', float('nan')))]
        if nv:
            nrmse_pct = 100 * float(np.mean(nv))
    a9 = dict(model='cond', method='PBFM',
             rho_median=round(float(np.median(rhos)), 4),
             rho_over_floor=round(float(np.median(ratios)), 4),
             valid_pct=round(100 * float(np.mean(valid)), 2),
             coverage=round(len(set(valid_pfs)) / n_branches, 4) if n_branches else float('nan'),
             entropy_nats=round(ent, 4),
             nrmse_pct=round(nrmse_pct, 3) if nrmse_pct == nrmse_pct else nrmse_pct)
    a9_hdr = ['model', 'method', 'rho_median', 'rho_over_floor', 'valid_pct',
             'coverage', 'entropy_nats', 'nrmse_pct']
    with open(os.path.join(outdir, 'paper_table_a9.csv'), 'w') as fh:
        fh.write(','.join(a9_hdr) + '\n')
        fh.write(','.join(str(a9[h]) for h in a9_hdr) + '\n')
    print(f'  [{tag}] wrote paper_table_a9.csv  '
          f'(max H = ln({n_branches}) = {math.log(n_branches):.4f} nats)')
    print(f'  [{tag}] rho(med)={a9["rho_median"]:.4f}  rho/floor={a9["rho_over_floor"]:.3f}  '
          f'valid%={a9["valid_pct"]:.1f}  coverage={a9["coverage"]:.2f}  '
          f'H={a9["entropy_nats"]:.3f} nats  NRMSE%={a9["nrmse_pct"]}')

    # -------- figures --------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(['PBFM'], [a9['valid_pct']], color='tab:green')
    ax[0].set_ylabel('valid %'); ax[0].set_ylim(0, 105); ax[0].set_title('Physics validity')
    ax[1].bar(['PBFM'], [100 * a9['coverage']], color='tab:blue')
    ax[1].set_ylabel('coverage %'); ax[1].set_ylim(0, 105); ax[1].set_title('Branch coverage')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_validity.png'), dpi=120)
    plt.close(fig)

    by_ra = defaultdict(list)
    for r, ratio in zip(records, ratios):
        by_ra[r['Ra']].append(ratio)
    if by_ra:
        ras = sorted(by_ra)
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(ras)), 4))
        ax.boxplot([by_ra[ra] for ra in ras], showfliers=False)
        ax.set_xticks(range(1, len(ras) + 1))
        ax.set_xticklabels([f'{ra:.0f}' for ra in ras], fontsize=8, rotation=30)
        ax.axhline(1.0, color='k', ls='--', lw=1, label='reference floor')
        ax.set_ylabel(r'$\rho$ / floor'); ax.set_xlabel('Ra')
        ax.set_title('PBFM floor-relative residual'); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_resid_over_floor.png'), dpi=120)
        plt.close(fig)

    c = Counter(r['pbfm_pf'] for r in records)
    if c:
        pfs = sorted(c, key=str)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(pfs)), 4))
        ax.bar([str(p) for p in pfs], [c[p] for p in pfs], color='tab:purple')
        ax.set_xlabel('planform'); ax.set_ylabel('count')
        ax.set_xticklabels([str(p) for p in pfs], rotation=25, fontsize=7)
        ax.set_title('Planform distribution (PBFM, all samples)')
        fig.tight_layout(); fig.savefig(os.path.join(outdir, 'fig_roll_hist.png'), dpi=120)
        plt.close(fig)

    _ckpt_clear(args.out_dir, tag)
    return a9


# calibrate_thresholds/median_floor_by_planform both need a live (data, resid)
# pair; merge_and_report runs in the PARENT process (never spawned), once per
# suite, AFTER that suite's workers have exited -- populated once by main()
# from the SAME shared `data` object the workers use (see main()'s comment),
# not a fresh RB3DData load, so this never adds another copy of the bank.
_data_cache = {}


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--splits', default='./datasets/rb3d_multisolution/splits')
    p.add_argument('--ckpt', default='ckpt_rb3d_pbfm_cond.pt')
    p.add_argument('--out-dir', dest='out_dir', default='rb3d_pbfm_eval')
    p.add_argument('--gpus', type=int, default=None,
                   help='shard points across this many GPUs (default: all visible)')
    p.add_argument('--suite', default='all',
                   choices=['all', 'gt', 'drift', 'heldout'])
    p.add_argument('--k-samples', dest='k_samples', type=int, default=24)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--n-step', dest='n_step', type=int, default=50)
    p.add_argument('--cal-mult', dest='cal_mult', type=float, default=1.5)
    p.add_argument('--cal-q', dest='cal_q', type=float, default=0.95)
    p.add_argument('--max-heldout-points', dest='max_heldout_points', type=int,
                   default=0, help='cap heldout points for speed (0 = all)')
    p.add_argument('--fast', action='store_true',
                   help='fewer samples + capped heldout, for a quick full run')
    p.add_argument('--quick', action='store_true', help='fast smoke test')
    args = p.parse_args()
    if args.quick:
        args.k_samples = 6; args.n_step = 20; args.max_heldout_points = 3
    if args.fast:
        args.k_samples = 12
        if args.max_heldout_points == 0:
            args.max_heldout_points = 40
    return args


def _run_suite(suite, args, n_branches, data):
    gpus = args.gpus
    if gpus is None:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpus = max(1, int(gpus))

    if suite in ('gt', 'drift'):
        bank_name = 'test_gt_bank.pt' if suite == 'gt' else 'test_drift_bank.pt'
        if not os.path.exists(os.path.join(args.splits, bank_name)):
            print(f'[{suite}] {bank_name} not found -- skipping')
            return None
        if gpus > 1:
            import torch.multiprocessing as mp
            mp.spawn(worker_reference, args=(gpus, args, suite, bank_name, data),
                     nprocs=gpus, join=True)
        else:
            worker_reference(0, 1, args, suite, bank_name, data)
        return merge_and_report(args, suite, n_branches, with_nrmse=True)

    if suite == 'heldout':
        if not os.path.exists(os.path.join(args.splits, 'test_heldout_bank.pt')):
            print('[heldout] test_heldout_bank.pt not found -- skipping')
            return None
        if gpus > 1:
            import torch.multiprocessing as mp
            mp.spawn(worker_heldout, args=(gpus, args, data), nprocs=gpus, join=True)
        else:
            worker_heldout(0, 1, args, data)
        return merge_and_report(args, 'heldout', n_branches, with_nrmse=False)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert os.path.exists(args.ckpt) or os.path.exists(os.path.join('.', args.ckpt)), \
        f'PBFM checkpoint not found: {args.ckpt}'

    # Load the (multi-GB) bank ONCE and share its storage across every spawned
    # worker via share_memory_(), instead of each rank independently loading
    # its own copy. RB3DData's construction transiently needs close to 2x the
    # bank's on-disk size (~15-17GB here) while densifying it -- two ranks
    # hitting that peak simultaneously under mp.spawn is exactly what SIGKILLed
    # a 31GB-RAM box during PBFM training (see rb3d_pbfm_action_plan.md); an
    # eval run with --gpus>1 would hit the identical ceiling without this.
    #
    # ONLY do this when mp.spawn will actually be used (gpus>1): share_memory_()
    # allocates a NEW /dev/shm-backed copy of each tensor (transiently ~2x, so
    # ~17GB here) with no benefit in the single-process (--gpus 1) path, and
    # /dev/shm is a small, fixed-size tmpfs (14GB seen on this box) shared with
    # whatever else is using it -- e.g. a concurrently-running training job
    # that already claimed its own ~8.6GB there via the identical mechanism.
    gpus_resolved = args.gpus
    if gpus_resolved is None:
        gpus_resolved = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpus_resolved = max(1, int(gpus_resolved))

    data = RB3DData(os.path.join(args.splits, 'train_bank.pt'), device='cpu')
    if gpus_resolved > 1:
        data.fields.share_memory_()
        data.params.share_memory_()
        data.Ra.share_memory_()
        data.Pr.share_memory_()
    resid = Residual(data.Nx, data.Ny, data.Nz, data.aspect)
    n_branches = len({tuple(k[2]) for k in data.keys})
    _data_cache['data'] = data
    _data_cache['resid'] = resid
    print(f'[eval] {n_branches} distinct trained planform(s) in the bank '
          f'(A.9 coverage/entropy denominator, max H = ln({n_branches}) = '
          f'{math.log(n_branches):.4f} nats)')

    suites = ['gt', 'drift', 'heldout'] if args.suite == 'all' else [args.suite]
    summary = {}
    for suite in suites:
        row = _run_suite(suite, args, n_branches, data)
        if row is not None:
            summary[suite] = row

    import json as _json
    _json.dump(summary, open(os.path.join(args.out_dir, 'summary.json'), 'w'),
              indent=2)
    print(f'\n[eval] DONE. organized outputs under {args.out_dir}/')
    print(f'[eval] summary: {_json.dumps(summary, indent=2)}')


if __name__ == '__main__':
    main()
