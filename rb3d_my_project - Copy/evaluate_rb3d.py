#!/usr/bin/env python3
"""
evaluate_rb3d.py
================================================================================
Evaluate the trained RB3D flow-matching surrogates (conditioned & unconditioned)
across every test split produced by prepare_rb3d_splits.py, and write an
ORGANIZED tree of metrics + figures:

    <out>/
      gt/          -- Test-GT: NRMSE vs ground truth at 4 held-out (Ra,Pr) points
        metrics.csv, metrics.json
        figA_roll_coverage.png          per-point branch coverage (stacked bar)
        figB_gen_vs_gt_<model>.png      generated-vs-GT field gallery (u_z)
        figC_nrmse_by_branch.png        NRMSE per planform, both models
      drift/       -- Test-drift: recover the reorganized branch (has GT)
        metrics.csv, figB_gen_vs_gt_<model>.png
      heldout/     -- Test-heldout: unconverged points (NO valid GT)
        metrics.csv, figA_coverage.png, figD_residual_hist.png
      resolution/  -- FNO resolution-independence probe
        metrics.csv, figE_nrmse_vs_resolution.png
      summary.json -- top-line numbers for both models, all suites

Scoring philosophy
------------------------------------------------------------------------------
Every generated sample is (1) drawn from the flow model, (2) physics-RELAXED
(short IMEX+Leray pseudo-time at the target Ra,Pr -- the 3D analog of PCFM's
projection), then judged by:
  * Data NRMSE  -- generated-vs-reference, after SYMMETRY ALIGNMENT (a roll at
                   an arbitrary phase/reflection would otherwise show a huge,
                   meaningless error). Only where a valid reference exists
                   (GT, drift). Heldout has NO reference -> residual only.
  * Phys resid  -- bulk momentum + temperature + continuity, the SAME operators
                   the verifier uses (imported), so eval and verification agree.
  * valid %     -- physics-valid fraction against a threshold CALIBRATED on the
                   training bank (per planform).
  * coverage    -- which of the trained planforms the model produced.
HONESTY: we report the planform coverage and NRMSE BEFORE relaxation as well as
after, so "the model generated the right branch" is not confused with "the
relaxer rescued a bad sample".

USAGE
    python evaluate_rb3d.py \
        --splits ./datasets/rb3d_multisolution/splits \
        --cond-ckpt ckpt_rb3d_cond.pt --uncond-ckpt ckpt_rb3d_uncond.pt \
        --suite all --k-samples 24
    python evaluate_rb3d.py --suite gt --quick        # fast smoke
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rb3d_pcfm_common import RB3DData, load_flow_model
from rb3d_pcfm_sampler import PCFMProjector, PTCProjector, pcfm_sample
from verify_rb3d_multisolution import (_k, ddz, lap, lap_gen, _sine_basis,
                                       _rms_i, leray_residual, _proj_pack,
                                       ddx_fd, ddy_fd)
from rb3d_classify import classify_purity


# ---------------------------------------------------------------------------
#  RESUME SUPPORT: both suite loops do independent, expensive work per
#  (model, Ra, Pr) point (a full PCFM projection of k_samples fields). A long
#  suite (heldout can be 150+ points) getting Ctrl+C'd or hitting a Kaggle
#  session timeout previously lost ALL of that suite's progress. Here every
#  finished point is appended to a checkpoint file as one JSON line; on
#  restart, already-done (model, Ra, Pr) keys are skipped and their records
#  are loaded back in, so the run picks up exactly where it stopped.
# ---------------------------------------------------------------------------
def _ckpt_path(outdir, tag):
    return os.path.join(outdir, f'.{tag}_checkpoint.jsonl')


def _ckpt_load(outdir, tag):
    """Return (done_keys:set[(model,Ra,Pr)], records:{model:[rec,...]}) from
    any prior partial run. Coverage/gallery are DERIVED data (cheap to rebuild
    from records) and are not stored -- only the records themselves, which are
    the expensive-to-recompute artifact (each cost a full PCFM projection of
    k_samples fields). Silently starts fresh if the file is missing or its
    last line is truncated (a hard kill mid-write) -- resumability must never
    be able to crash a run, only save it time."""
    path = _ckpt_path(outdir, tag)
    done, records = set(), defaultdict(list)
    if not os.path.exists(path):
        return done, records
    n_read = 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break                       # truncated last line -- stop here
            m, Ra, Pr = row['model'], row['Ra'], row['Pr']
            done.add((m, Ra, Pr))
            for rec in row['recs']:
                for k in list(rec):
                    if k.endswith('_pf') and isinstance(rec[k], list):
                        rec[k] = tuple(rec[k])
                records[m].append(rec)
            n_read += 1
    if n_read:
        print(f'[resume] {tag}: loaded {n_read} finished point(s) from '
              f'{path} ({sum(len(v) for v in records.values())} samples) '
              f'-- skipping those, continuing the rest.')
    return done, records


def _ckpt_append(outdir, tag, model, Ra, Pr, recs):
    path = _ckpt_path(outdir, tag)
    row = dict(model=model, Ra=Ra, Pr=Pr, recs=recs)
    with open(path, 'a') as fh:
        fh.write(json.dumps(row) + '\n')
        fh.flush()
        os.fsync(fh.fileno())              # survive a hard kill, not just Ctrl+C


def _ckpt_clear(outdir, tag):
    path = _ckpt_path(outdir, tag)
    if os.path.exists(path):
        os.remove(path)


# ============================================================================
#  Physics residual (same operators as the verifier)
# ============================================================================
class Residual:
    def __init__(self, Nx, Ny, Nz, aspect, device='cpu'):
        self.device = device
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        Gx, Gy = aspect
        self.dx, self.dy = Gx / Nx, Gy / Ny
        self.dz = 1.0 / (Nz - 1)
        self.z = torch.linspace(0, 1, Nz, dtype=torch.float64)
        self.kx = _k(Nx, Gx, device, torch.float64)
        self.ky = _k(Ny, Gy, device, torch.float64)
        self.sine = _sine_basis(Nz, device, torch.float64)
        self.proj = _proj_pack(Nx, Ny, Nz, self.kx, self.ky, device,
                               torch.float64)

    def __call__(self, fields, Ra, Pr):
        """fields: (4,Nx,Ny,Nz) [u,v,w,T']. Returns (mom_bulk, temp, cont)."""
        u, v, w, Tp = (fields[i].double() for i in range(4))
        kx, ky, dz, dx, dy = self.kx, self.ky, self.dz, self.dx, self.dy
        lapT = lap_gen(Tp, kx, ky, self.sine)

        def advF(a):
            return u * ddx_fd(a, dx) + v * ddy_fd(a, dy) + w * ddz(a, dz)

        # continuity
        div = ddx_fd(u, dx) + ddy_fd(v, dy) + ddz(w, dz)
        s_div = max(_rms_i(ddx_fd(u, dx)), _rms_i(ddy_fd(v, dy)),
                    _rms_i(ddz(w, dz)), 1e-12)
        r_cont = _rms_i(div) / s_div
        # temperature
        r_temp = _rms_i(advF(Tp) - w - lapT) / max(_rms_i(lapT), 1e-12)
        # momentum (pressure-free), bulk (drop 3 wall planes)
        Ru = advF(u) - Pr * lap_gen(u, kx, ky, self.sine)
        Rv = advF(v) - Pr * lap_gen(v, kx, ky, self.sine)
        Rw = advF(w) - Pr * lap_gen(w, kx, ky, self.sine) - Pr * Ra * Tp
        Ru, Rv, Rw = leray_residual(Ru, Rv, Rw, kx, ky, dz, self.proj)
        c = 3
        rcb = lambda t: float(t[:, :, c:self.Nz - c].pow(2).mean().sqrt())
        s_Mb = max(rcb(Pr * Ra * Tp), 1e-12)
        r_mom = math.sqrt(rcb(Ru) ** 2 + rcb(Rv) ** 2 + rcb(Rw) ** 2) / s_Mb
        return r_mom, r_temp, r_cont


# ============================================================================
#  Symmetry alignment + data errors
# ============================================================================
def align_to(gen, ref):
    """Align gen (4,Nx,Ny,Nz) to ref over the RB symmetry group (x/y
    translations via FFT cross-correlation + x/y reflections). Returns the
    aligned gen and (best L2)."""
    Nx, Ny = gen.shape[1], gen.shape[2]
    best = None; best_l2 = math.inf
    refh = torch.fft.rfft2(ref, dim=(1, 2))
    for fx in (False, True):
        for fy in (False, True):
            g = gen.clone()
            if fx:
                g = torch.flip(g, dims=(1,)); g[0] = -g[0]
            if fy:
                g = torch.flip(g, dims=(2,)); g[1] = -g[1]
            # cross-correlation summed over channels & z -> best (sx,sy)
            gh = torch.fft.rfft2(g, dim=(1, 2))
            corr = torch.fft.irfft2((refh.conj() * gh).sum(dim=(0, 3)),
                                    s=(Nx, Ny), dim=(0, 1))
            idx = int(corr.reshape(-1).argmax())
            sx, sy = idx // Ny, idx % Ny
            ga = torch.roll(g, shifts=(-sx, -sy), dims=(1, 2))
            l2 = float((ga - ref).pow(2).sum())
            if l2 < best_l2:
                best_l2 = l2; best = ga
    return best, best_l2


def data_errors(a, ref):
    mse = float((a - ref).pow(2).mean())
    nrmse = float((a - ref).norm() / ref.norm().clamp_min(1e-12))
    return mse, nrmse


# ============================================================================
#  Calibrate per-planform validity thresholds on the training bank
# ============================================================================
def calibrate_thresholds(data, resid, mult=1.5, q=0.95):
    """Per-planform validity thresholds from the TRAINING data's own residual
    distribution. Takes the already-loaded RB3DData object -- an earlier
    version re-loaded the bank from disk with torch.load, which held a SECOND
    ~8.6 GB copy (plus deserialisation transient) in the same process as
    RB3DData's copy and OOM-killed the hi-res evaluation (exit -9)."""
    per = defaultdict(list)
    dev = getattr(resid, 'device', 'cpu')
    sc = data.scale[:, None, None, None].to(dev)
    for i, (Ra, Pr, mode) in enumerate(data.keys):
        # data.fields lives in CPU RAM (single-copy design keeps the 8.6 GB
        # dataset off the GPU); move each de-normalised entry to the verifier's
        # device so the FFT wavenumber tensors match. One entry at a time =>
        # negligible GPU memory.
        f = (data.fields[i].to(dev)) * sc
        rm, rt, rc = resid(f, Ra, Pr)
        per[tuple(mode)].append(max(rm, rt))          # gate on mom & temp
    thr = {}
    for mode, vals in per.items():
        t = torch.tensor(vals)
        thr[mode] = float(mult * torch.quantile(t, q))
    floor = max(thr.values()) if thr else 0.3
    return thr, floor


# ============================================================================
#  Draw K relaxed samples at one (Ra,Pr) from a model
# ============================================================================
def physics_L2(rm, rt, rc):
    """Single scalar 'physics residual / L2' of a field = RMS of the three
    relative PDE-residual components (bulk momentum, temperature, continuity).
    This is the PDE residual of the OUTPUT (no PDE is solved) -- low = the
    field satisfies the steady equations."""
    return math.sqrt(rm ** 2 + rt ** 2 + rc ** 2)


def draw_paired(model, ck, data, projector, resid, Ra, Pr, k, n_step,
                proj_start, batch, seed0, n_newton=1, floor=None):
    """Draw k samples and return, for each, BOTH the VANILLA (raw flow, no
    correction) field and the PCFM (projected) field. Same seed => same noise
    => a true paired ablation of the correction alone.

    ADAPTIVE RE-PROJECTION: if `floor` is given, samples whose residual is
    still above it after the standard pass get ONE extra, longer final
    projection. Rationale: a sample the flow emits as a BLEND of two branches
    relaxes through the neighbourhood of a saddle, where the dynamics slow
    down exponentially -- a fixed step budget strands exactly those samples at
    rm ~ 0.1-0.3 (counted invalid) while easy samples finish early. Retrying
    only the failures buys validity where it is cheap to buy and costs nothing
    elsewhere."""
    out = []
    done = 0
    device = next(model.parameters()).device
    while done < k:
        b = min(batch, k - done)
        Ra_l, Pr_l = [Ra] * b, [Pr] * b
        f_van, d_van = pcfm_sample(model, ck, data, Ra_l, Pr_l, projector,
                                   n_step=n_step, seed=seed0 + done,
                                   vanilla=True)
        f_pcf, d_pcf = pcfm_sample(model, ck, data, Ra_l, Pr_l, projector,
                                   n_step=n_step, proj_start=proj_start,
                                   n_newton=n_newton, seed=seed0 + done)
        if floor is not None:
            retry = [j for j in range(b)
                     if max(resid(f_pcf[j], Ra, Pr)[:2]) > floor]
            if retry:
                phys = f_pcf[retry].to(device)
                Ra_t = torch.full((len(retry),), Ra, device=device)
                Pr_t = torch.full((len(retry),), Pr, device=device)
                try:
                    phys, _, _ = projector.project(phys, Ra_t, Pr_t,
                                                   n_newton=n_newton,
                                                   final=True)
                except TypeError:
                    phys, _, _ = projector.project(phys, Ra_t, Pr_t,
                                                   n_newton=n_newton)
                f_pcf[retry] = phys.detach().cpu()
        for j in range(b):
            vf, pf_ = f_van[j], f_pcf[j]
            rm_v, rt_v, rc_v = resid(vf, Ra, Pr)
            rm_p, rt_p, rc_p = resid(pf_, Ra, Pr)
            # "relax" column now = how far the PROJECTION moved the sample
            disp = float((pf_ - vf).norm() / vf.norm().clamp_min(1e-9))
            out.append(dict(
                van_field=vf, pcfm_field=pf_,
                van_pf=tuple(_classify1(vf, data.aspect)),
                pcfm_pf=tuple(_classify1(pf_, data.aspect)),
                van_res=(rm_v, rt_v, rc_v), pcfm_res=(rm_p, rt_p, rc_p),
                relax=disp))
        done += b
    return out


# Purity gate: labels below PURITY_MIN become ('unknown',0,0) instead of a
# forced guess. Training branches sit at purity ~1.00, so 0.90 leaves margin
# for generator sharpness while still rejecting blends/noise/harmonic artifacts.
PURITY_MIN = 0.90


def _classify1(field, aspect):
    lab, _pur, _ = classify_purity(field[2], aspect[0], aspect[1],
                                   purity_min=PURITY_MIN)
    return lab


# ============================================================================
#  SUITE: Test-GT / Test-drift (reference-based)
# ============================================================================
def run_reference_suite(models, data, projector, resid, thresholds, floor,
                        bank, args, outdir, drift=False):
    os.makedirs(outdir, exist_ok=True)
    # group references by (Ra,Pr) -> {planform: field}
    refs_by_pt = defaultdict(dict)
    for (Ra, Pr, mode), e in bank['entries'].items():
        pf = tuple(e['planform'])
        z = data.z
        Tp = e['grid_T'].float() - (1.0 - z)[None, None, :]
        refs_by_pt[(Ra, Pr)][pf] = torch.stack(
            [e['grid_u'].float(), e['grid_v'].float(),
             e['grid_w'].float(), Tp])
    points = sorted(refs_by_pt)
    tag = 'drift' if drift else 'gt'
    print(f'[{tag}] {len(points)} point(s), '
          f'{sum(len(r) for r in refs_by_pt.values())} references')

    records = {m: [] for m in models}          # per (model): list of sample recs
    coverage = {m: defaultdict(lambda: defaultdict(int)) for m in models}
    gallery = {m: {} for m in models}          # NOT resumed (illustrative only;
                                                # rebuilding needs the fields,
                                                # which the checkpoint omits to
                                                # stay small -- see _ckpt_append)
    done_keys, r_records = _ckpt_load(outdir, tag)
    for m, recs in r_records.items():
        records.setdefault(m, []).extend(recs)
        for rec in recs:                       # replay the SAME coverage
            for variant in ('van', 'pcfm'):     # increment used in the loop
                pf = tuple(rec[variant + '_pf'])
                coverage[m][(rec['Ra'], rec['Pr'])][f'{variant}_{pf}'] += 1
    for mname, (model, ck) in models.items():
        for pi, (Ra, Pr) in enumerate(points):
            if (mname, Ra, Pr) in done_keys:
                continue
            refs = refs_by_pt[(Ra, Pr)]
            samples = draw_paired(model, ck, data, projector, resid, Ra, Pr,
                                  args.k_samples, args.n_step, args.proj_start,
                                  args.batch, 1000 * pi, args.n_newton,
                                  floor=floor)
            for s in samples:
                rec = dict(Ra=Ra, Pr=Pr, relax=s['relax'])
                for variant in ('van', 'pcfm'):
                    pf = s[variant + '_pf']
                    rm, rt, rc = s[variant + '_res']
                    thr = thresholds.get(pf, floor)
                    rec[variant + '_pf'] = pf
                    rec[variant + '_physL2'] = physics_L2(rm, rt, rc)
                    rec[variant + '_mom'] = rm
                    rec[variant + '_temp'] = rt
                    rec[variant + '_cont'] = rc
                    rec[variant + '_valid'] = bool(max(rm, rt) < thr)
                    rec[variant + '_nrmse'] = float('nan')
                    # NRMSE to SAME-planform ref (strict) AND to BEST-matching
                    # ref (min over branches). best<<same => branch/label
                    # mismatch; best~=same => right branch but imperfect.
                    if refs:
                        rec[variant + '_nrmse_best'] = min(
                            data_errors(align_to(s[variant + '_field'], rr)[0],
                                        rr)[1] for rr in refs.values())
                    else:
                        rec[variant + '_nrmse_best'] = float('nan')
                    coverage[mname][(Ra, Pr)][f'{variant}_{pf}'] += 1
                    if pf in refs:               # aligned NRMSE vs its branch
                        ga, _ = align_to(s[variant + '_field'], refs[pf])
                        _, nrmse = data_errors(ga, refs[pf])
                        rec[variant + '_nrmse'] = nrmse
                        if variant == 'pcfm' and rec['pcfm_valid'] and \
                           pf not in gallery[mname].get((Ra, Pr), {}):
                            gallery[mname].setdefault((Ra, Pr), {})[pf] = (
                                ga.cpu(), refs[pf].cpu())
                records[mname].append(rec)
            sub = [r for r in records[mname] if r['Ra'] == Ra and r['Pr'] == Pr]
            _ckpt_append(outdir, tag, mname, Ra, Pr, sub)
            vpL = np.mean([r['van_physL2'] for r in sub])
            ppL = np.mean([r['pcfm_physL2'] for r in sub])
            pv = sum(r['pcfm_valid'] for r in sub)
            cov = sorted({r['pcfm_pf'] for r in sub if r['pcfm_valid']})
            print(f'  [{tag}:{mname}] Ra={Ra:7.0f} Pr={Pr:4.2f}  '
                  f'physL2 vanilla={vpL:.3f} -> PCFM={ppL:.3f}  '
                  f'PCFMvalid={pv}/{args.k_samples}  covered={cov}', flush=True)

    _write_reference_metrics(records, data, outdir, tag)
    _fig_coverage(coverage, points, outdir, tag)
    _fig_gallery(gallery, data, outdir, tag)
    _fig_physL2(records, outdir, tag)
    if not drift:
        _fig_nrmse_by_branch(records, data, outdir)
    _ckpt_clear(outdir, tag)          # suite finished cleanly -- drop the log
    return records


def _write_reference_metrics(records, data, outdir, tag):
    hdr = ['model', 'planform', 'n',
           'van_physL2', 'pcfm_physL2',
           'van_valid%', 'pcfm_valid%',
           'van_NRMSE%', 'pcfm_NRMSE%', 'pcfm_NRMSEbest%',
           'pcfm_mom', 'pcfm_temp', 'pcfm_cont', 'proj_disp']
    csv = [','.join(hdr)]
    summary = []
    for m, recs in records.items():
        by_pf = defaultdict(list)
        for r in recs:
            by_pf[r['pcfm_pf']].append(r)     # group by the projected planform
        def line(label, sub):
            vn_v = [r['van_nrmse'] for r in sub if not math.isnan(r['van_nrmse'])]
            vn_p = [r['pcfm_nrmse'] for r in sub if not math.isnan(r['pcfm_nrmse'])]
            vn_b = [r['pcfm_nrmse_best'] for r in sub
                    if not math.isnan(r['pcfm_nrmse_best'])]
            return [m, label, len(sub),
                    round(np.mean([r['van_physL2'] for r in sub]), 4),
                    round(np.mean([r['pcfm_physL2'] for r in sub]), 4),
                    round(100 * np.mean([r['van_valid'] for r in sub]), 1),
                    round(100 * np.mean([r['pcfm_valid'] for r in sub]), 1),
                    round(100 * np.mean(vn_v), 2) if vn_v else float('nan'),
                    round(100 * np.mean(vn_p), 2) if vn_p else float('nan'),
                    round(100 * np.mean(vn_b), 2) if vn_b else float('nan'),
                    round(np.mean([r['pcfm_mom'] for r in sub]), 4),
                    round(np.mean([r['pcfm_temp'] for r in sub]), 4),
                    round(np.mean([r['pcfm_cont'] for r in sub]), 4),
                    round(np.nanmean([r['relax'] for r in sub]), 4)]
        for pf, sub in sorted(by_pf.items()):
            csv.append(','.join(str(x) for x in line(str(pf), sub)))
        row = line('AVG', recs)
        csv.append(','.join(str(x) for x in row))
        summary.append(dict(model=m, van_physL2=row[3], pcfm_physL2=row[4],
                            van_valid=row[5], pcfm_valid=row[6],
                            van_nrmse=row[7], pcfm_nrmse=row[8],
                            pcfm_nrmse_best=row[9]))
    with open(os.path.join(outdir, 'metrics.csv'), 'w') as fh:
        fh.write('\n'.join(csv))
    json.dump(summary, open(os.path.join(outdir, 'metrics.json'), 'w'),
              indent=2)
    print(f'  [{tag}] wrote metrics.csv / metrics.json '
          f'(vanilla vs PCFM columns)')


# ============================================================================
#  SUITE: Test-heldout (no reference -> physics + coverage only)
# ============================================================================
def run_heldout_suite(models, data, projector, resid, thresholds, floor,
                      bank, args, outdir):
    os.makedirs(outdir, exist_ok=True)
    points = sorted({(k[0], k[1]) for k in bank['entries']})
    if args.max_heldout_points:
        points = points[:args.max_heldout_points]
    print(f'[heldout] {len(points)} unconverged point(s) '
          f'(physics-residual scoring only)')
    records = {m: [] for m in models}
    coverage = {m: defaultdict(int) for m in models}
    done_keys, r_records = _ckpt_load(outdir, 'heldout')
    for m, recs in r_records.items():
        records.setdefault(m, []).extend(recs)
        for rec in recs:
            if rec.get('pcfm_valid'):
                coverage[m][tuple(rec['pcfm_pf'])] += 1
    for mname, (model, ck) in models.items():
        for pi, (Ra, Pr) in enumerate(points):
            if (mname, Ra, Pr) in done_keys:
                continue
            samples = draw_paired(model, ck, data, projector, resid, Ra, Pr,
                                  args.k_samples, args.n_step, args.proj_start,
                                  args.batch, 7000 * pi, args.n_newton,
                                  floor=floor)
            new_recs = []
            for s in samples:
                rec = dict(Ra=Ra, Pr=Pr, relax=s['relax'])
                for variant in ('van', 'pcfm'):
                    pf = s[variant + '_pf']
                    rm, rt, rc = s[variant + '_res']
                    thr = thresholds.get(pf, floor)
                    rec[variant + '_pf'] = pf
                    rec[variant + '_physL2'] = physics_L2(rm, rt, rc)
                    rec[variant + '_valid'] = bool(max(rm, rt) < thr)
                    if variant == 'pcfm' and rec['pcfm_valid']:
                        coverage[mname][pf] += 1
                records[mname].append(rec)
                new_recs.append(rec)
            _ckpt_append(outdir, 'heldout', mname, Ra, Pr, new_recs)
            v = sum(1 for r in new_recs if r['pcfm_valid'])
            print(f'  [heldout:{mname}] Ra={Ra:7.0f} Pr={Pr:4.2f}  '
                  f'PCFMvalid={v}/{args.k_samples}', flush=True)
    # metrics (vanilla vs PCFM)
    csv = ['model,van_physL2,pcfm_physL2,van_valid%,pcfm_valid%,'
           'distinct_valid_planforms']
    summary = []
    for m, recs in records.items():
        vpL = np.mean([r['van_physL2'] for r in recs])
        ppL = np.mean([r['pcfm_physL2'] for r in recs])
        vv = 100 * np.mean([r['van_valid'] for r in recs])
        pv = 100 * np.mean([r['pcfm_valid'] for r in recs])
        dp = len({r['pcfm_pf'] for r in recs if r['pcfm_valid']})
        csv.append(f'{m},{vpL:.4f},{ppL:.4f},{vv:.1f},{pv:.1f},{dp}')
        summary.append(dict(model=m, van_physL2=round(vpL, 4),
                            pcfm_physL2=round(ppL, 4), van_valid=round(vv, 1),
                            pcfm_valid=round(pv, 1), distinct_valid=dp))
        print(f'  [heldout:{m}] physL2 vanilla={vpL:.3f}->PCFM={ppL:.3f}  '
              f'valid {vv:.0f}%->{pv:.0f}%  distinct valid={dp}')
    with open(os.path.join(outdir, 'metrics.csv'), 'w') as fh:
        fh.write('\n'.join(csv))
    json.dump(summary, open(os.path.join(outdir, 'metrics.json'), 'w'),
              indent=2)
    _fig_heldout_coverage(coverage, data, outdir)
    _fig_residual_hist(records, outdir)
    _fig_physL2(records, outdir, 'heldout')
    _ckpt_clear(outdir, 'heldout')    # suite finished cleanly -- drop the log
    return records


# ============================================================================
#  SUITE: resolution independence
# ============================================================================
def run_resolution_suite(models, data, resid_cache, res_bank, args, outdir):
    os.makedirs(outdir, exist_ok=True)
    factors = res_bank['res_factors']
    base = res_bank['base_grid']
    print(f'[resolution] factors {factors} on {len(res_bank["entries"])} refs')
    # references grouped by (Ra,Pr): {factor: {planform: field}}
    refs = defaultdict(lambda: defaultdict(dict))
    for (Ra, Pr, mode), e in res_bank['entries'].items():
        for f, fld in e['resampled'].items():
            z = torch.linspace(0, 1, fld['Nz'])
            Tp = fld['grid_T'] - (1.0 - z)[None, None, :]
            refs[f][(Ra, Pr)][tuple(e['planform'])] = torch.stack(
                [fld['grid_u'], fld['grid_v'], fld['grid_w'], Tp])
    rows = ['model,factor,Nx,Ny,mean_NRMSE%,valid%,n']
    curve = defaultdict(list)
    for mname, (model, ck) in models.items():
        for f in factors:
            nx = list(refs[f].values())[0]
            any_field = list(list(refs[f].values())[0].values())[0]
            Nx, Ny, Nz = any_field.shape[1:]
            # a resolution-specific data view + relaxer
            dview = _ResView(data, Nx, Ny, Nz)
            dev_r = next(model.parameters()).device
            if args.projector == 'ptc':
                rproj = PTCProjector(Nx, Ny, Nz, data.aspect, dev_r,
                                     steps=args.ptc_steps,
                                     final_steps=args.ptc_final)
            else:
                rproj = PCFMProjector(Nx, Ny, Nz, data.aspect, dev_r,
                                      cg_iters=args.cg_iters)
            rr = resid_cache.get((Nx, Ny, Nz))
            if rr is None:
                rr = Residual(Nx, Ny, Nz, data.aspect)
                resid_cache[(Nx, Ny, Nz)] = rr
            nrmses, valids = [], []
            for pi, (Ra, Pr) in enumerate(sorted(refs[f])):
                samples = draw_paired(model, ck, dview, rproj, rr, Ra, Pr,
                                      args.k_res, args.n_step, args.proj_start,
                                      args.batch, 50 * pi, args.n_newton)
                rf = refs[f][(Ra, Pr)]
                for s in samples:
                    pf = s['pcfm_pf']
                    rm, rt, _ = s['pcfm_res']
                    valids.append(int(max(rm, rt) < args.res_valid_tol))
                    if pf in rf:
                        ga, _ = align_to(s['pcfm_field'], rf[pf])
                        _, nr = data_errors(ga, rf[pf])
                        nrmses.append(nr)
            mn = 100 * np.mean(nrmses) if nrmses else float('nan')
            vp = 100 * np.mean(valids) if valids else float('nan')
            rows.append(f'{mname},{f},{Nx},{Ny},{mn:.3f},{vp:.1f},{len(nrmses)}')
            curve[mname].append((f, mn, vp))
            print(f'  [{mname}] factor {f:4.2f} ({Nx}x{Ny}): '
                  f'NRMSE={mn:.2f}% valid={vp:.1f}%', flush=True)
    with open(os.path.join(outdir, 'metrics.csv'), 'w') as fh:
        fh.write('\n'.join(rows))
    _fig_resolution(curve, base, outdir)
    return curve


class _ResView:
    """A lightweight RB3DData-like view at a different (Nx,Ny) for sampling."""
    def __init__(self, data, Nx, Ny, Nz):
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.aspect = data.aspect
        self.z = torch.linspace(0, 1, Nz)
        self.scale = data.scale
        self.Ra_rng, self.Pr_rng = data.Ra_rng, data.Pr_rng
        self.norm_params = data.norm_params


# ============================================================================
#  Figures
# ============================================================================
def _fig_coverage(coverage, points, outdir, tag):
    models = list(coverage)
    fig, axes = plt.subplots(1, len(models), figsize=(6.5 * len(models), 4.2),
                             squeeze=False)
    for ax, m in zip(axes[0], models):
        pfs = sorted({k[len('pcfm_'):] for pt in coverage[m].values()
                      for k in pt if k.startswith('pcfm_')})
        cmap = plt.get_cmap('viridis', max(len(pfs), 1))
        xs = np.arange(len(points)); bottoms = np.zeros(len(points))
        for j, pf in enumerate(pfs):
            cnts = [coverage[m][pt].get('pcfm_' + pf, 0) for pt in points]
            ax.bar(xs, cnts, bottom=bottoms, color=cmap(j), label=pf)
            bottoms += np.array(cnts)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{r:.0f}\n{p:.2f}' for r, p in points], fontsize=7)
        ax.set_ylabel('samples')
        ax.set_title(f'{m}: PCFM planform produced / point')
        ax.legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, 'figA_coverage.png'),
                                    dpi=130); plt.close(fig)


def _fig_gallery(gallery, data, outdir, tag):
    x = (torch.arange(data.Nx) * (data.aspect[0] / data.Nx)).numpy()
    for m, gal in gallery.items():
        if not gal:
            continue
        key = max(gal, key=lambda k: len(gal[k]))
        got = gal[key]; pfs = sorted(got)
        fig, axes = plt.subplots(len(pfs), 2, figsize=(11, 1.9 * len(pfs)),
                                 squeeze=False)
        for row, pf in enumerate(pfs):
            gen, ref = got[pf]
            mid = gen.shape[2] // 2                       # y mid-plane, show u_z
            for col, (fld, ttl) in enumerate([(gen, 'PCFM generated'),
                                              (ref, 'ground truth')]):
                ax = axes[row, col]
                im = ax.imshow(fld[2, :, mid, :].numpy().T, origin='lower',
                               aspect='auto', cmap='RdBu_r',
                               extent=[x[0], x[-1], 0, 1])
                plt.colorbar(im, ax=ax, fraction=0.025)
                ax.set_title(f'{pf} {ttl} (u_z, y-mid)', fontsize=8)
                ax.set_ylabel('z', fontsize=8)
        fig.suptitle(f'{m}: PCFM vs GT @ Ra={key[0]:.0f} Pr={key[1]:.2f} '
                     f'(phase-aligned)', fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(outdir, f'figB_gen_vs_gt_{m}.png'), dpi=130)
        plt.close(fig)


def _fig_physL2(records, outdir, tag):
    """Headline ablation: vanilla vs PCFM physics residual, per model."""
    models = list(records)
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 2, 4.2))
    x = np.arange(len(models)); w = 0.38
    van = [np.mean([r['van_physL2'] for r in records[m]]) for m in models]
    pcf = [np.mean([r['pcfm_physL2'] for r in records[m]]) for m in models]
    ax.bar(x - w / 2, van, w, label='vanilla (raw flow)', color='#d1495b')
    ax.bar(x + w / 2, pcf, w, label='PCFM (projected)', color='#2e86ab')
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('physics L2 (PDE residual of output)')
    if min([v for v in van + pcf if v > 0], default=1) > 0:
        ax.set_yscale('log')
    ax.set_title(f'{tag}: vanilla vs PCFM physics residual (lower = better)')
    for i in range(len(models)):
        if van[i] > 0 and pcf[i] > 0:
            ax.text(x[i], max(van[i], pcf[i]) * 1.1,
                    f'{van[i]/pcf[i]:.0f}x', ha='center', fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figF_physL2_vanilla_vs_pcfm.png'), dpi=130)
    plt.close(fig)


def _fig_nrmse_by_branch(records, data, outdir):
    models = list(records)
    pfs = sorted({r['pcfm_pf'] for m in models for r in records[m]
                  if not math.isnan(r['pcfm_nrmse'])})
    if not pfs:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(pfs) + 2, 4))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        means = []
        for pf in pfs:
            vn = [r['pcfm_nrmse'] * 100 for r in records[m]
                  if r['pcfm_pf'] == pf and not math.isnan(r['pcfm_nrmse'])]
            means.append(np.mean(vn) if vn else 0)
        ax.bar(np.arange(len(pfs)) + i * w, means, w, label=m)
    ax.set_xticks(np.arange(len(pfs)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([str(p) for p in pfs], fontsize=7, rotation=20)
    ax.set_ylabel('PCFM Data NRMSE %'); ax.set_title('NRMSE per planform (PCFM)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figC_nrmse_by_branch.png'), dpi=130)
    plt.close(fig)


def _fig_heldout_coverage(coverage, data, outdir):
    models = list(coverage)
    fig, ax = plt.subplots(figsize=(8, 4))
    pfs = sorted({pf for m in models for pf in coverage[m]})
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        ax.bar(np.arange(len(pfs)) + i * w,
               [coverage[m].get(pf, 0) for pf in pfs], w, label=m)
    ax.set_xticks(np.arange(len(pfs)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([str(p) for p in pfs], fontsize=7, rotation=20)
    ax.set_ylabel('valid PCFM samples'); ax.legend()
    ax.set_title('held-out: valid steady branches produced (no GT available)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figA_coverage.png'), dpi=130)
    plt.close(fig)


def _fig_residual_hist(records, outdir):
    models = list(records)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in models:
        ax.hist([r['van_physL2'] for r in records[m]], bins=40, alpha=0.4,
                label=f'{m} vanilla', histtype='stepfilled')
        ax.hist([r['pcfm_physL2'] for r in records[m]], bins=40, alpha=0.6,
                label=f'{m} PCFM', histtype='step', lw=2)
    ax.set_xlabel('physics L2 (PDE residual of output)')
    ax.set_ylabel('samples'); ax.legend(fontsize=8)
    ax.set_title('held-out: vanilla vs PCFM physics residual')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figD_residual_hist.png'), dpi=130)
    plt.close(fig)


def _fig_resolution(curve, base, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, pts in curve.items():
        pts = sorted(pts)
        fs = [p[0] for p in pts]; nr = [p[1] for p in pts]
        ax.plot(fs, nr, 'o-', label=f'{m} NRMSE')
    ax.axvline(1.0, color='gray', ls='--', lw=1, label='trained resolution')
    ax.set_xlabel(f'resolution factor  (1.0 = {base["Nx"]}x{base["Ny"]})')
    ax.set_ylabel('Data NRMSE %')
    ax.set_title('FNO resolution independence: NRMSE vs input resolution')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figE_nrmse_vs_resolution.png'), dpi=130)
    plt.close(fig)


# ============================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--splits',
                   default='./datasets/rb3d_multisolution/splits')
    p.add_argument('--cond-ckpt', default='ckpt_rb3d_cond.pt')
    p.add_argument('--uncond-ckpt', default='ckpt_rb3d_uncond.pt')
    p.add_argument('--out', default='rb3d_eval')
    p.add_argument('--device', default=None,
                   help='cpu | cuda | cuda:0 ... (default: cuda if available)')
    p.add_argument('--dual-gpu', dest='dual_gpu', action='store_true',
                   help='with --suite all, shard the four suites across two '
                        'GPUs as independent processes (~2x, no contention)')
    p.add_argument('--suite', default='all',
                   choices=['all', 'gt', 'drift', 'heldout', 'resolution'])
    p.add_argument('--k-samples', type=int, default=24)
    p.add_argument('--k-res', type=int, default=8)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--projector', choices=['ptc', 'gn'], default='ptc',
                   help="h(u)=0 solver inside the PCFM loop. 'ptc' = pseudo-"
                        "transient continuation (IMEX+Leray steps; robust, "
                        "fixes momentum AND temperature; reaches stable "
                        "branches only). 'gn' = matrix-free Gauss-Newton "
                        "(stability-blind, can reach unstable branches, but "
                        "at 128x64x49 needs a far larger --cg-iters than is "
                        "practical and is NOT recommended for evaluation).")
    p.add_argument('--ptc-steps', type=int, default=75,
                   help='IMEX steps per INTERLEAVED projection (ptc only)')
    p.add_argument('--ptc-final', type=int, default=600,
                   help='IMEX steps for the FINAL projection (ptc only)')
    p.add_argument('--proj-start', type=float, default=None,
                   help='begin corrections once flow time tau >= this (0.0 = '
                        'fully interleaved, 1.0 = final projection only). '
                        'Default: 0.9 for ptc (interleaved PTC is expensive; '
                        'a few late nudges + a strong final projection is the '
                        'efficient regime), 0.6 for gn.')
    p.add_argument('--n-newton', type=int, default=1,
                   help='Gauss-Newton updates per correction (paper uses 1)')
    p.add_argument('--cg-iters', type=int, default=12,
                   help='matrix-free CG iterations inside each GN solve')
    p.add_argument('--cal-mult', type=float, default=1.5)
    p.add_argument('--cal-q', type=float, default=0.95)
    p.add_argument('--res-valid-tol', type=float, default=0.25)
    p.add_argument('--max-heldout-points', type=int, default=0,
                   help='cap heldout points for speed (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fast', action='store_true',
                   help='~10x faster full run (k=12, heldout capped 40, '
                        'relax 600 + early-stop); recommended default')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.k_samples = 6; args.k_res = 3; args.n_step = 20
        args.max_heldout_points = 3
    if args.fast:
        # ~10x faster full run: fewer samples, capped heldout, shorter relax
        # (early-stop usually ends relaxation sooner anyway), same suites.
        args.k_samples = 12; args.k_res = 6; args.cg_iters = 8
        if args.max_heldout_points == 0:
            args.max_heldout_points = 40
    return args


def _run_dual_gpu(args):
    if args.proj_start is None:
        args.proj_start = 0.9 if args.projector == 'ptc' else 0.6
    """Shard the four suites across two GPUs -- but with AT MOST ONE process
    per GPU at a time, and staggered starts. The earlier version launched all
    four suites at once (3 processes on cuda:1 + 1 on cuda:0); each process
    loads the full train bank (~8.6 GB steady + ~2x transient during
    torch.load), so 3-4 concurrent loads stacked to ~50-70 GB and the kernel
    OOM-killed the workers (exit -9). Grouping per GPU and chaining suites
    keeps peak RAM to ~2 processes (~1 loading + 1 steady), which fits."""
    import subprocess as _sp
    import threading, time as _time
    groups = [('cuda:0', ['heldout']),
              ('cuda:1', ['gt', 'drift', 'resolution'])]
    base = [sys.executable, os.path.abspath(__file__),
            '--splits', args.splits, '--out', args.out,
            '--cond-ckpt', args.cond_ckpt, '--uncond-ckpt', args.uncond_ckpt,
            '--k-samples', str(args.k_samples), '--k-res', str(args.k_res),
            '--batch', str(args.batch), '--n-step', str(args.n_step),
            '--projector', args.projector,
            '--ptc-steps', str(args.ptc_steps),
            '--ptc-final', str(args.ptc_final),
            '--proj-start', str(args.proj_start), '--n-newton',
            str(args.n_newton), '--cg-iters', str(args.cg_iters),
            '--seed', str(args.seed)]
    if args.fast:
        base.append('--fast')
    if args.max_heldout_points:
        base += ['--max-heldout-points', str(args.max_heldout_points)]
    fails = []

    def run_group(dev, suites, delay):
        _time.sleep(delay)                 # stagger the torch.load transients
        for suite in suites:
            print(f'[dual-gpu] {suite} on {dev}', flush=True)
            p = _sp.Popen(base + ['--suite', suite, '--device', dev])
            if p.wait() != 0:
                fails.append((suite, p.returncode))
                print(f'[dual-gpu] suite {suite} failed '
                      f'(code {p.returncode})', flush=True)

    threads = [threading.Thread(target=run_group, args=(dev, suites, i * 45))
               for i, (dev, suites) in enumerate(groups)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('[dual-gpu] all suites done.' if not fails
          else f'[dual-gpu] failed: {fails}')
    sys.exit(0 if not fails else 1)


def main():
    args = parse_args()
    if getattr(args, 'dual_gpu', False) and args.suite == 'all':
        try:
            import torch as _t
            if _t.cuda.device_count() >= 2:
                _run_dual_gpu(args)
                return
            print('[eval] --dual-gpu ignored: <2 GPUs visible')
        except Exception:
            pass
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    train_path = os.path.join(args.splits, 'train_bank.pt')
    data = RB3DData(train_path, device=device)
    if args.proj_start is None:
        args.proj_start = 0.9 if args.projector == 'ptc' else 0.6
    if args.projector == 'ptc':
        projector = PTCProjector(data.Nx, data.Ny, data.Nz, data.aspect,
                                 device, steps=args.ptc_steps,
                                 final_steps=args.ptc_final)
    else:
        projector = PCFMProjector(data.Nx, data.Ny, data.Nz, data.aspect,
                                  device, cg_iters=args.cg_iters)
    resid = Residual(data.Nx, data.Ny, data.Nz, data.aspect)
    resid_cache = {(data.Nx, data.Ny, data.Nz): resid}

    models = {}
    for tag, path in (('cond', args.cond_ckpt), ('uncond', args.uncond_ckpt)):
        if os.path.exists(path):
            models[tag] = load_flow_model(path, device)
        else:
            print(f'[warn] {tag} checkpoint {path} not found -- skipping')
    assert models, 'no checkpoints found'

    print('[eval] calibrating validity thresholds on the training bank ...')
    thresholds, floor = calibrate_thresholds(data, resid,
                                             mult=args.cal_mult, q=args.cal_q)
    print(f'[eval] thresholds (per planform): '
          f'{ {str(k): round(v,3) for k,v in thresholds.items()} } floor={floor:.3f}')

    summary = {}
    def load(name):
        return torch.load(os.path.join(args.splits, name),
                          map_location='cpu', weights_only=False)

    if args.suite in ('all', 'gt'):
        rec = run_reference_suite(models, data, projector, resid, thresholds,
                                  floor, load('test_gt_bank.pt'), args,
                                  os.path.join(args.out, 'gt'), drift=False)
        summary['gt'] = {m: dict(
            van_physL2=float(np.mean([r['van_physL2'] for r in rec[m]])),
            pcfm_physL2=float(np.mean([r['pcfm_physL2'] for r in rec[m]])),
            pcfm_valid_pct=100*np.mean([r['pcfm_valid'] for r in rec[m]]),
            pcfm_nrmse_pct=100*float(np.nanmean([r['pcfm_nrmse'] for r in rec[m]])))
            for m in rec}
    if args.suite in ('all', 'drift'):
        p = os.path.join(args.splits, 'test_drift_bank.pt')
        if os.path.exists(p):
            rec = run_reference_suite(models, data, projector, resid, thresholds,
                                      floor, load('test_drift_bank.pt'), args,
                                      os.path.join(args.out, 'drift'),
                                      drift=True)
            summary['drift'] = {m: dict(
                pcfm_nrmse_pct=100*float(np.nanmean([r['pcfm_nrmse'] for r in rec[m]])),
                van_physL2=float(np.mean([r['van_physL2'] for r in rec[m]])),
                pcfm_physL2=float(np.mean([r['pcfm_physL2'] for r in rec[m]])))
                for m in rec}
    if args.suite in ('all', 'heldout'):
        rec = run_heldout_suite(models, data, projector, resid, thresholds,
                                floor, load('test_heldout_bank.pt'), args,
                                os.path.join(args.out, 'heldout'))
        summary['heldout'] = {m: dict(
            van_physL2=float(np.mean([r['van_physL2'] for r in rec[m]])),
            pcfm_physL2=float(np.mean([r['pcfm_physL2'] for r in rec[m]])),
            pcfm_valid_pct=100*np.mean([r['pcfm_valid'] for r in rec[m]]),
            distinct_valid=len({r['pcfm_pf'] for r in rec[m] if r['pcfm_valid']}))
            for m in rec}
    if args.suite in ('all', 'resolution'):
        p = os.path.join(args.splits, 'res_probe_bank.pt')
        if os.path.exists(p):
            curve = run_resolution_suite(models, data, resid_cache, load(
                'res_probe_bank.pt'), args, os.path.join(args.out, 'resolution'))
            summary['resolution'] = {m: [[f, round(nr, 3)]
                                     for f, nr, _ in sorted(v)]
                                     for m, v in curve.items()}

    json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'),
              indent=2)
    print(f'\n[eval] DONE. organized outputs under {args.out}/')
    print(f'[eval] summary: {json.dumps(summary, indent=2)}')


if __name__ == '__main__':
    main()