#!/usr/bin/env python3
"""
prepare_rb3d_splits.py
================================================================================
Turn a verified bank from generate_rb3d_multisolution.py into the train / test
splits for the multimodal flow-matching surrogate. The split is driven by the
TWO independent labels every entry already carries:

    converged : did the pseudo-time march reach a steady state?
    locked    : does the recomputed planform == the seeded mode?
                (i.e. entry['planform'] == the mode in its key)

giving three physically-distinct groups, plus two carved-out test sets:

  TRAIN            converged & locked
                   -> genuine seeded steady branches. The model trains ONLY on
                      these. (A few whole (Ra,Pr) points are held out entirely
                      for TEST-GT, so they never leak into training.)

  TEST-HELDOUT     NOT converged
                   -> time-dependent / oscillatory flow at the low-Pr/high-Ra
                      corner. NO trustworthy ground truth (a snapshot of an
                      oscillation is not a steady solution). The surrogate is
                      scored ONLY by physics residual here: "did it produce A
                      valid steady branch at a parameter the solver couldn't
                      settle?" -- the 3D analog of the Bratu held-out-branch
                      test. Stored WITHOUT fields being treated as references.

  TEST-DRIFT       converged & NOT locked
                   -> a valid steady state that reorganized to a DIFFERENT
                      planform than seeded (e.g. roll(2,0) -> rect(6,0) via the
                      Eckhaus/zigzag instability). These DO have ground truth
                      (they are converged), so they get full NRMSE scoring, but
                      against their ACTUAL (recomputed) planform, not the seed.
                      A bonus generalization probe: "recover the reorganized
                      branch."

  TEST-GT          k random (Ra,Pr) POINTS pulled ENTIRELY out of train
                   -> every converged+locked branch at these points becomes a
                      held-out reference with full ground truth. Because whole
                      points are removed, this measures interpolation in
                      (Ra,Pr) with no train leakage. (default k=4)

  RES-PROBE        the TEST-GT references, each RESAMPLED to several grids
                   (Fourier in x,y; the FNO is meant to be discretization
                   invariant). Lets the evaluator feed the model inputs at a
                   resolution it never trained on and check the error is ~flat.

Output: one .pt per split under --out, each a self-contained mini-bank with the
same schema as the input (so the existing evaluator/loader reads them directly),
plus split_manifest.json describing counts and the exact (Ra,Pr) points removed
for TEST-GT (reproducibility).

USAGE
    python prepare_rb3d_splits.py --bank ./datasets/rb3d_multisolution/refs_bank.pt
    python prepare_rb3d_splits.py --bank bank.pt --n-gt-points 4 --seed 0 \
        --res-factors 0.5 1.0 1.5 2.0
"""

import argparse
import json
import math
import os
import shutil

import torch


def safe_save(bank, path):
    """torch.save with a free-space check and read-back verification, so a
    disk-full write leaves NO file at `path` rather than a silently truncated
    one. (A truncated .pt loads fine at write time -- torch.save() does not
    error on ENOSPC until the reader later hits 'failed finding central
    directory'. This is exactly what happened when refs_bank.pt (9.6 GB) was
    still on disk while writing train_bank.pt (8.6 GB) pushed a ~20 GB Kaggle
    quota over the edge mid-write.)"""
    outdir = os.path.dirname(os.path.abspath(path)) or '.'
    os.makedirs(outdir, exist_ok=True)
    # rough size estimate: sum of tensor bytes in the bank (a good proxy --
    # torch.save's zip overhead is small relative to the float32 payload)
    est = 0
    for e in bank['entries'].values():
        for v in e.values():
            if torch.is_tensor(v):
                est += v.numel() * v.element_size()
            elif isinstance(v, dict):        # e.g. res_probe's 'resampled'
                for vv in v.values():
                    if torch.is_tensor(vv):
                        est += vv.numel() * vv.element_size()
    free = shutil.disk_usage(outdir).free
    if free < 1.15 * est:
        raise RuntimeError(
            f'not enough disk to safely write {path}: {free/1e9:.2f} GB free, '
            f'~{est/1e9:.2f} GB needed. Free some space (e.g. delete the '
            f'source bank with --delete-source, or remove old checkpoints) '
            f'before writing splits.')
    tmp = path + '.tmp'
    torch.save(bank, tmp)
    try:
        chk = torch.load(tmp, map_location='cpu', weights_only=False)
        assert len(chk['entries']) == len(bank['entries'])
        del chk
    except Exception as ex:
        os.remove(tmp)
        raise RuntimeError(f'write verification failed for {path}: {ex}. '
                           f'Removed the incomplete file (this is the disk-'
                           f'full failure mode -- check free space).')
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
def load_banks(paths):
    """Load one bank, or union several shard banks in memory (e.g. the two
    --dual-gpu shard files, avoiding an extra merged copy on disk)."""
    bank = torch.load(paths[0], map_location='cpu', weights_only=False)
    for p in paths[1:]:
        b = torch.load(p, map_location='cpu', weights_only=False)
        bank['entries'].update(b['entries'])
        del b
    if len(paths) > 1:
        print(f'[split] unioned {len(paths)} bank files -> '
              f'{len(bank["entries"])} entries')
    try:
        exp = len(bank['param_points']) * len(bank['mode_list'])
        if len(bank['entries']) != exp:
            print(f'[split] NOTE: {len(bank["entries"])} entries vs {exp} '
                  f'expected ({exp - len(bank["entries"])} configs absent)')
    except (KeyError, TypeError):
        pass
    return bank


def is_locked(key, entry):
    _, _, mode = key
    return tuple(entry['planform']) == tuple(mode)


def blank_bank(src):
    """A new bank carrying all top-level metadata of src but no entries."""
    return {k: src[k] for k in src if k != 'entries'} | {'entries': {}}


def resample_fields(entry, grid, factors):
    """Fourier-resample (u,v,w,T') in x and y to new horizontal resolutions.
    z is left untouched (walls -> a spectral z-resample would need the sine/
    cosine bases; horizontal Fourier resampling is the clean, exact operation
    for the periodic directions and is what probes the FNO's x/y mode handling).
    Returns {factor: {'grid_*':..., 'Nx':..,'Ny':..}}."""
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    out = {}
    for f in factors:
        nx = max(4, int(round(Nx * f)) // 2 * 2)
        ny = max(4, int(round(Ny * f)) // 2 * 2)
        fields = {}
        for name in ('grid_u', 'grid_v', 'grid_w', 'grid_T'):
            a = entry[name].double()                        # (Nx,Ny,Nz)
            if name == 'grid_T':                            # resample T' then re-add base
                z = torch.linspace(0, 1, Nz, dtype=a.dtype)
                a = a - (1.0 - z)[None, None, :]
            ah = torch.fft.rfft2(a, dim=(0, 1))
            # crop/pad spectral coeffs to the new size (Fourier interpolation)
            b = _resize_rfft2(ah, Nx, Ny, nx, ny) * (nx * ny) / (Nx * Ny)
            r = torch.fft.irfft2(b, s=(nx, ny), dim=(0, 1))
            if name == 'grid_T':
                z = torch.linspace(0, 1, Nz, dtype=r.dtype)
                r = r + (1.0 - z)[None, None, :]
            fields[name] = r.float()
        fields['Nx'] = nx; fields['Ny'] = ny; fields['Nz'] = Nz
        out[round(f, 3)] = fields
    return out


def _resize_rfft2(ah, Nx, Ny, nx, ny):
    """Resize an rfft2 spectrum (dims 0,1) from (Nx,Ny) to (nx,ny) by
    cropping/zero-padding around the Nyquist -- exact Fourier interpolation."""
    Nz = ah.shape[-1]
    kyc = ny // 2 + 1
    out = torch.zeros(nx, kyc, Nz, dtype=ah.dtype)
    # x axis is full fft freq order [0..+..-..]; copy the lowest |kx| modes
    kx_keep = min(Nx, nx)
    # positive & negative kx blocks
    pos = kx_keep // 2 + 1
    neg = kx_keep - pos
    ky_keep = min(ah.shape[1], kyc)
    out[:pos, :ky_keep] = ah[:pos, :ky_keep]
    if neg > 0:
        out[nx - neg:, :ky_keep] = ah[Nx - neg:, :ky_keep]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', required=True, nargs='+',
                   help='one or more bank files. Pass the shard files directly '
                        '(refs_bank.pt.shard0 refs_bank.pt.shard1) to skip the '
                        'merge step entirely -- the merged file is a full extra '
                        'copy of the dataset on disk and is not needed here.')
    p.add_argument('--out', default=None,
                   help='output dir (default: <bank_dir>/splits)')
    p.add_argument('--n-gt-points', type=int, default=4,
                   help='number of random (Ra,Pr) points fully held out for '
                        'TEST-GT (default 4)')
    p.add_argument('--gt-min-branches', type=int, default=4,
                   help='only pick TEST-GT points with >= this many '
                        'converged+locked branches (default 4)')
    p.add_argument('--res-factors', type=float, nargs='+',
                   default=[0.5, 0.75, 1.0, 1.5, 2.0],
                   help='horizontal resolution multipliers for RES-PROBE')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--delete-source', dest='delete_source', action='store_true',
                   help='remove the source bank(s) after the splits are written '
                        'AND verified. The five split files together hold ~all '
                        'the entries again (~9 GB at 128x64x49), so keeping the '
                        'original too roughly doubles disk; once the splits '
                        'exist the original is not needed by any later step.')
    args = p.parse_args()

    bank = load_banks(args.bank)
    ent = bank['entries']
    grid = bank['grid']
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.bank[0])),
                                   'splits')
    os.makedirs(out, exist_ok=True)

    # ---- categorize every entry -------------------------------------------
    conv_locked, conv_drift, unconv = [], [], []
    for key, e in ent.items():
        if not e['converged']:
            unconv.append(key)
        elif is_locked(key, e):
            conv_locked.append(key)
        else:
            conv_drift.append(key)
    print(f'[split] {len(ent)} entries: {len(conv_locked)} converged+locked, '
          f'{len(conv_drift)} converged+drifted, {len(unconv)} unconverged')

    # ---- choose TEST-GT points (whole (Ra,Pr) points, removed from train) --
    # group converged+locked keys by (Ra,Pr)
    by_point = {}
    for key in conv_locked:
        by_point.setdefault((key[0], key[1]), []).append(key)
    eligible = [pt for pt, ks in by_point.items()
                if len(ks) >= args.gt_min_branches]
    assert eligible, (f'no (Ra,Pr) point has >= {args.gt_min_branches} '
                      f'converged+locked branches; lower --gt-min-branches')
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(eligible), generator=rng)[:args.n_gt_points].tolist()
    gt_points = [eligible[i] for i in idx]
    gt_point_set = set(gt_points)
    print(f'[split] TEST-GT points ({len(gt_points)}): '
          + ', '.join(f'(Ra={r:.0f},Pr={p:.2f})' for r, p in gt_points))

    # ---- assemble the split banks -----------------------------------------
    train = blank_bank(bank)
    test_heldout = blank_bank(bank)
    test_drift = blank_bank(bank)
    test_gt = blank_bank(bank)

    for key in conv_locked:
        if (key[0], key[1]) in gt_point_set:
            test_gt['entries'][key] = ent[key]
        else:
            train['entries'][key] = ent[key]
    for key in conv_drift:
        # relabel to the ACTUAL planform so downstream code reads a truthful key
        e = ent[key]
        newkey = (key[0], key[1], tuple(e['planform']))
        test_drift['entries'].setdefault(newkey, e)
        test_drift['entries'][newkey] = {**e, 'seeded_mode': tuple(key[2])}
    for key in unconv:
        # store WITHOUT implying the field is a valid reference
        e = ent[key]
        test_heldout['entries'][key] = {**e, 'reference_valid': False}

    # tag each split
    train['split'] = 'train'
    test_heldout['split'] = 'test_heldout'
    test_drift['split'] = 'test_drift'
    test_gt['split'] = 'test_gt'
    test_gt['gt_points'] = gt_points

    safe_save(train, os.path.join(out, 'train_bank.pt'))
    safe_save(test_heldout, os.path.join(out, 'test_heldout_bank.pt'))
    safe_save(test_drift, os.path.join(out, 'test_drift_bank.pt'))
    safe_save(test_gt, os.path.join(out, 'test_gt_bank.pt'))

    # ---- RES-PROBE: resample the TEST-GT references ------------------------
    res_probe = blank_bank(bank)
    res_probe['split'] = 'res_probe'
    res_probe['res_factors'] = [round(f, 3) for f in args.res_factors]
    res_probe['base_grid'] = {'Nx': grid['Nx'], 'Ny': grid['Ny'], 'Nz': grid['Nz']}
    for key in test_gt['entries']:
        e = ent[key]
        res_probe['entries'][key] = {
            'planform': tuple(e['planform']),
            'resampled': resample_fields(e, grid, args.res_factors),
        }
    safe_save(res_probe, os.path.join(out, 'res_probe_bank.pt'))

    # ---- manifest ----------------------------------------------------------
    manifest = {
        'source_bank': [os.path.abspath(p) for p in args.bank],
        'grid': {k: int(grid[k]) for k in ('Nx', 'Ny', 'Nz')},
        'aspect': list(bank['aspect']),
        'mode_list': [list(m) for m in bank['mode_list']],
        'counts': {
            'train': len(train['entries']),
            'test_heldout': len(test_heldout['entries']),
            'test_drift': len(test_drift['entries']),
            'test_gt': len(test_gt['entries']),
        },
        'test_gt_points': [[float(r), float(p)] for r, p in gt_points],
        'res_factors': [round(f, 3) for f in args.res_factors],
        'seed': args.seed,
        'notes': {
            'train': 'converged & locked, minus TEST-GT points',
            'test_heldout': 'unconverged/time-dependent; physics-residual '
                            'scoring only, no valid reference field',
            'test_drift': 'converged but reorganized; NRMSE vs recomputed '
                          'planform; carries seeded_mode',
            'test_gt': 'whole (Ra,Pr) points held out; full ground-truth NRMSE',
            'res_probe': 'TEST-GT refs Fourier-resampled in x,y for '
                         'resolution-independence testing',
        },
    }
    with open(os.path.join(out, 'split_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=2)

    # ---- report ------------------------------------------------------------
    print(f'\n[split] wrote to {out}/')
    for name, b in (('train_bank.pt', train), ('test_gt_bank.pt', test_gt),
                    ('test_drift_bank.pt', test_drift),
                    ('test_heldout_bank.pt', test_heldout),
                    ('res_probe_bank.pt', res_probe)):
        print(f'    {name:24s}: {len(b["entries"]):5d} entries')
    # sanity: no TEST-GT point leaked into train
    tp = {(k[0], k[1]) for k in train['entries']}
    assert not (tp & gt_point_set), 'LEAK: a TEST-GT point is in train!'
    # branch-count distribution in train
    per_pt = {}
    for k in train['entries']:
        per_pt[(k[0], k[1])] = per_pt.get((k[0], k[1]), 0) + 1
    from collections import Counter
    dist = Counter(per_pt.values())
    print(f'[split] train branch-count/point: '
          + ', '.join(f'{n}br x {c}pts' for n, c in sorted(dist.items())))
    print(f'[split] manifest -> {out}/split_manifest.json')
    print('[split] no TEST-GT leakage into train (checked).')

    if args.delete_source:
        # verify all five split files are readable before removing the source
        ok = True
        for name in ('train_bank.pt', 'test_gt_bank.pt', 'test_drift_bank.pt',
                     'test_heldout_bank.pt', 'res_probe_bank.pt'):
            fp = os.path.join(out, name)
            try:
                chk = torch.load(fp, map_location='cpu', weights_only=False)
                assert 'entries' in chk
                del chk
            except Exception as ex:
                ok = False
                print(f'[split] verification of {name} FAILED ({ex}); '
                      f'keeping source bank(s).')
                break
        if ok:
            for p in args.bank:
                try:
                    os.remove(p)
                    print(f'[split] removed source {p}')
                except OSError as ex:
                    print(f'[split] could not remove {p}: {ex}')


if __name__ == '__main__':
    main()