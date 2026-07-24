#!/usr/bin/env python3
"""
prepare_rb2d_pbfm_data.py
================================================================================
Split the full RB2D reference bank into the parts every downstream PBFM script
needs. This is the PBFM-side equivalent of

    python heldout_branch_test.py --mode split     --bank refs_bank.pt --out-dir split/
    python heldout_branch_test.py --mode calibrate --bank refs_bank.pt --out-dir split/
    python heldout_branch_test.py --mode selftest  --bank refs_bank.pt --out-dir split/

rolled into one command, and it is deliberately self-contained so you do not need
`heldout_branch_test.py` sitting in the PBFM folder. The artifacts it writes are
byte-compatible with that harness: you can still score PBFM samples with your
existing `heldout_branch_test.py --mode score`.

--------------------------------------------------------------------------
THE SPLIT (identical rule to the PCFM pipeline)
--------------------------------------------------------------------------
Each (Ra,Pr) parameter point is classified by the fate of its seeded branches:

  * 'clean'        every branch converged AND locked onto its seeded roll number
                   -> goes to TRAIN
  * 'recoverable'  >=1 branch collapsed to a different roll number, but none
                   went unsteady -> goes to the HELD-OUT TEST. A steady branch
                   provably exists there (it is an exact solution), the
                   time-marching solver just could not hold it. This is the
                   credible capability test for a generative surrogate.
  * 'unsteady'     >=1 branch never converged (a steady state may not exist)
                   -> EXCLUDED from both.

--------------------------------------------------------------------------
OUTPUTS  (default --out-dir ./rb2d/split)
--------------------------------------------------------------------------
  train_bank.pt          clean points only -> what PBFM trains on
  heldout_test_spec.pt   held-out (Ra,Pr) + target/unreached branches
                         + per-branch acceptance THRESHOLDS (from calibrate)
  heldout_reference.pt   the reference fields at the held-out points (for
                         ground-truth comparison on held-out data)

--------------------------------------------------------------------------
CALIBRATION NOTE (differs slightly from the harness default, on purpose)
--------------------------------------------------------------------------
Thresholds are calibrated on the TRAIN references by default (--cal-on train),
so no held-out information leaks into the acceptance bar. Use --cal-on all to
reproduce the harness's behaviour of calibrating over the whole bank.

The RB combined residual has an intrinsic, Ra-dependent floor, so the bar is
per-branch `mult x quantile_q` of the reference residuals -- never a fixed
absolute number.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
    # from the RB2D-PBFM folder, with your data at ./rb2d/refs_bank.pt
    python prepare_rb2d_pbfm_data.py --bank ./rb2d/refs_bank.pt

    # narrower calibration window (matches --cal-ra-max in the evaluator)
    python prepare_rb2d_pbfm_data.py --bank ./rb2d/refs_bank.pt --cal-ra-max 8000
"""
import argparse
import os

import torch

import rb2d_pbfm_common as C


# ============================================================================
#  Classify the bank by parameter point
# ============================================================================
def classify(bank):
    """(Ra,Pr) -> {'entries':{n:entry}, 'status':clean|recoverable|unsteady,
                   'unreached':[n...], 'unsteady':[n...]}"""
    params = {}
    for (Ra, Pr, n), e in bank['entries'].items():
        params.setdefault((Ra, Pr), {})[n] = e
    out = {}
    for (Ra, Pr), br in params.items():
        collapsed = [n for n, e in br.items() if e.get('roll_mode') != n]
        unsteady = [n for n, e in br.items() if not e.get('converged')]
        status = 'unsteady' if unsteady else ('recoverable' if collapsed else 'clean')
        out[(Ra, Pr)] = {'entries': br, 'status': status,
                         'unreached': sorted(set(collapsed)),
                         'unsteady': sorted(set(unsteady))}
    return out


# ============================================================================
#  SPLIT
# ============================================================================
def do_split(bank, out_dir):
    cls = classify(bank)
    n_list = sorted(bank['n_list'])
    train = {'entries': {}}
    heldout_ref = {'entries': {}}
    spec_params = []
    counts = {'clean': 0, 'recoverable': 0, 'unsteady': 0}

    for (Ra, Pr), info in cls.items():
        counts[info['status']] += 1
        if info['status'] == 'clean':
            for n, e in info['entries'].items():
                train['entries'][(Ra, Pr, n)] = e
        elif info['status'] == 'recoverable':
            spec_params.append({'Ra': Ra, 'Pr': Pr,
                                'target_branches': n_list,
                                'unreached': info['unreached']})
            for n, e in info['entries'].items():
                heldout_ref['entries'][(Ra, Pr, n)] = e
        # 'unsteady' -> excluded from both

    for d in (train, heldout_ref):
        for k in ('grid', 'meshes', 'aspect', 'n_list'):
            if k in bank:
                d[k] = bank[k]
        for k in ('Ra_range', 'Pr_range', 'sampling', 'method', 'param_points'):
            if k in bank:
                d[k] = bank[k]

    spec = {'params': spec_params, 'grid': bank['grid'],
            'aspect': bank['aspect'], 'n_list': n_list,
            'test_shape': bank['meshes']['test_shape'], 'thresholds': None}

    os.makedirs(out_dir, exist_ok=True)
    torch.save(train, os.path.join(out_dir, 'train_bank.pt'))
    torch.save(spec, os.path.join(out_dir, 'heldout_test_spec.pt'))
    torch.save(heldout_ref, os.path.join(out_dir, 'heldout_reference.pt'))

    print(f'[split] params: {counts["clean"]} clean -> TRAIN, '
          f'{counts["recoverable"]} recoverable -> HELD-OUT TEST, '
          f'{counts["unsteady"]} unsteady -> EXCLUDED')
    print(f'[split] TRAIN: {len(train["entries"])} entries '
          f'({counts["clean"]} param points x {len(n_list)} branches)')
    print(f'[split] HELD-OUT TEST: {len(spec_params)} param points; the model '
          f'must reproduce the solver-unreached branches:')
    for s in spec_params[:12]:
        print(f'    Ra={s["Ra"]:.0f} Pr={s["Pr"]:.2f}  unreached={s["unreached"]}')
    if len(spec_params) > 12:
        print(f'    ... (+{len(spec_params) - 12} more)')
    print(f'[split] wrote train_bank.pt, heldout_test_spec.pt, '
          f'heldout_reference.pt -> {out_dir}/')
    if counts['clean'] == 0:
        print('[split] WARNING: no clean param points -- nothing to train on.')
    if not spec_params:
        print('[split] NOTE: no recoverable points; the held-out branch test '
              'is not applicable to this bank.')
    return train, spec


# ============================================================================
#  CALIBRATE  (per-branch acceptance thresholds, written into the spec)
# ============================================================================
def _entry_residual(e, x, z, Ra, Pr):
    """Combined relative steady residual of one reference entry."""
    dx = float(x[1] - x[0]); dz = float(z[1] - z[0])
    kx = C.kx_vec(x.numel(), dx, x.device, torch.float64)
    psi = e['grid_psi'].to(torch.float64)
    u = C._ddz1(psi, dz); u[:, 0] = 0; u[:, -1] = 0
    w = -C._dX1(psi, kx); w[:, 0] = 0; w[:, -1] = 0
    Tfull = e['grid_T'].to(torch.float64)
    r = C.steady_residual_rel(u[None], w[None], Tfull[None], [Ra], [Pr], dx, dz)
    return float(r['combined'][0])


def do_calibrate(bank, out_dir, cal_on='train', mult=1.5, q=0.99,
                 ra_min=0.0, ra_max=float('inf')):
    src_name = {'train': 'train_bank.pt', 'all': None}[cal_on]
    if src_name:
        src = torch.load(os.path.join(out_dir, src_name), weights_only=False)
    else:
        src = bank
    grid = src['grid']
    x = grid['x'].to(torch.float64); z = grid['z'].to(torch.float64)

    by_n = {}
    for (Ra, Pr, n), e in src['entries'].items():
        if 'grid_psi' not in e or 'grid_T' not in e:
            continue
        if e.get('roll_mode') != n or not e.get('converged'):
            continue
        if Ra < ra_min or Ra > ra_max:
            continue
        by_n.setdefault(n, []).append(_entry_residual(e, x, z, Ra, Pr))

    if not by_n:
        raise SystemExit('[calibrate] no reference branches in the chosen Ra '
                         'window; widen --cal-ra-min/--cal-ra-max.')

    print(f'[calibrate] source={cal_on}  (metric = spectral-x + FD-z on u,w,T, '
          f'the same one used to score samples)')
    print(f'    {"n":>3} {"count":>6} {"median":>9} {"p95":>9} {"max":>9} '
          f'{"-> threshold":>14}')
    thresholds, floors = {}, {}
    for n in sorted(by_n):
        v = torch.tensor(by_n[n], dtype=torch.float64)
        thr = mult * float(v.quantile(q))
        thresholds[n] = thr
        floors[n] = float(v.median())
        print(f'    {n:>3} {len(v):>6} {float(v.median()):>9.2e} '
              f'{float(v.quantile(0.95)):>9.2e} {float(v.max()):>9.2e} '
              f'{thr:>14.2e}')

    spec_path = os.path.join(out_dir, 'heldout_test_spec.pt')
    spec = torch.load(spec_path, weights_only=False)
    spec['thresholds'] = thresholds
    spec['threshold_mult'] = mult
    spec['threshold_q'] = q
    spec['calibrated_on'] = cal_on
    spec['reference_floor'] = floors
    torch.save(spec, spec_path)
    print(f'[calibrate] wrote thresholds into {spec_path}')
    print('[calibrate] a sample counts as branch n iff wavenumber==n AND '
          'combined residual <= threshold[n].')
    return thresholds


# ============================================================================
#  SELFTEST  (prove the acceptance check discriminates, no model needed)
# ============================================================================
def do_selftest(bank, out_dir):
    grid = bank['grid']
    x = grid['x'].to(torch.float64); z = grid['z'].to(torch.float64)
    spec_path = os.path.join(out_dir, 'heldout_test_spec.pt')
    thr = torch.load(spec_path, weights_only=False).get('thresholds') \
        if os.path.exists(spec_path) else None

    ok_valid = bad_valid = 0
    collapsed_caught = collapsed_total = 0
    worst_valid = 0.0
    passed = total = 0
    for (Ra, Pr, n), e in bank['entries'].items():
        if 'grid_psi' not in e or 'grid_T' not in e:
            continue
        dz = float(z[1] - z[0])
        kx = C.kx_vec(x.numel(), float(x[1] - x[0]), x.device, torch.float64)
        psi = e['grid_psi'].to(torch.float64)
        w = -C._dX1(psi, kx); w[:, 0] = 0; w[:, -1] = 0
        m = C.wavenumber(w)
        if e.get('roll_mode') == n and e.get('converged'):
            r = _entry_residual(e, x, z, Ra, Pr)
            if m == n:
                ok_valid += 1
            else:
                bad_valid += 1
            worst_valid = max(worst_valid, r)
            if thr:
                total += 1
                if n in thr and r <= thr[n]:
                    passed += 1
        elif e.get('roll_mode') != n:
            collapsed_total += 1
            if m != n:
                collapsed_caught += 1

    print('[selftest] acceptance check run on the REFERENCE fields:')
    print(f'  valid references classified to the correct wavenumber: '
          f'{ok_valid} ok, {bad_valid} mis-ID')
    print(f'  worst combined residual among valid references: {worst_valid:.2e}')
    if collapsed_total:
        print(f'  collapsed entries correctly identified as a DIFFERENT roll: '
              f'{collapsed_caught}/{collapsed_total}')
    if thr:
        print(f'  valid references passing their calibrated threshold: '
              f'{passed}/{total}')
    verdict = (bad_valid == 0 and
               (collapsed_total == 0 or collapsed_caught == collapsed_total))
    print(f'[selftest] {"PASS" if verdict else "CHECK"}: the harness accepts '
          f'valid branches and does not mistake a collapsed field for the '
          f'missed branch.')
    return verdict


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', default='./rb2d/refs_bank.pt',
                   help='full reference bank (default ./rb2d/refs_bank.pt)')
    p.add_argument('--out-dir', dest='out_dir', default='./rb2d/split')
    p.add_argument('--mode', default='all',
                   choices=['all', 'split', 'calibrate', 'selftest'])
    p.add_argument('--cal-on', dest='cal_on', default='train',
                   choices=['train', 'all'],
                   help="calibrate thresholds on the TRAIN references (default, "
                        "no held-out leakage) or on the whole bank")
    p.add_argument('--cal-mult', dest='cal_mult', type=float, default=1.5)
    p.add_argument('--cal-q', dest='cal_q', type=float, default=0.99)
    p.add_argument('--cal-ra-min', dest='cal_ra_min', type=float, default=0.0)
    p.add_argument('--cal-ra-max', dest='cal_ra_max', type=float, default=float('inf'))
    return p.parse_args()


def main():
    args = parse_args()
    bank_path = C.find_file(args.bank) or args.bank
    if not os.path.exists(bank_path):
        raise SystemExit(f'[prepare] bank not found: {args.bank}')
    bank = torch.load(bank_path, map_location='cpu', weights_only=False)
    print(f'[prepare] bank={bank_path}  entries={len(bank["entries"])}  '
          f'grid {bank["grid"]["Nx"]}x{bank["grid"]["Nz"]}  '
          f'branches={sorted(bank["n_list"])}')

    if args.mode in ('all', 'split'):
        do_split(bank, args.out_dir)
    if args.mode in ('all', 'calibrate'):
        print()
        do_calibrate(bank, args.out_dir, cal_on=args.cal_on,
                     mult=args.cal_mult, q=args.cal_q,
                     ra_min=args.cal_ra_min, ra_max=args.cal_ra_max)
    if args.mode in ('all', 'selftest'):
        print()
        do_selftest(bank, args.out_dir)

    od = args.out_dir
    print(f"""
[prepare] done. Next steps (paths for your layout):

  # train the conditioned PBFM model on both T4s
  python train_rb2d_pbfm_conditioned.py --bank {od}/train_bank.pt --gpus 2

  # sweep evaluation
  python evaluate_rb2d_pbfm.py --bank {od}/train_bank.pt \\
      --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2

  # ground truth (in-distribution); use heldout_reference.pt for held-out GT
  python evaluate_rb2d_groundtruth.py --bank {od}/train_bank.pt \\
      --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2

  # held-out branch-recovery export + score
  python evaluate_rb2d_pbfm.py --bank {od}/train_bank.pt \\
      --ckpt ckpt_rb2d_pbfm_cond.pt --gpus 2 \\
      --heldout-spec {od}/heldout_test_spec.pt
  python heldout_branch_test.py --mode score --bank {args.bank} \\
      --out-dir {od} --samples rb2d_pbfm_eval/heldout_model_samples.pt -v
""")


if __name__ == '__main__':
    main()