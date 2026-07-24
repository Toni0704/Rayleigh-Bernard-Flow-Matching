#!/usr/bin/env python3
"""
novelty_audit.py
================================================================================
Decide -- defensibly -- whether a PCFM-generated sample is a GENUINE steady
solution of 3D Rayleigh-Benard that lies OUTSIDE the training modes.

THE CLAIM AND WHY IT IS EASY TO GET WRONG
--------------------------------------------------------------------------------
"The sample passes a residual tolerance, so it is a solution; its planform is
not in the training set, so it is a new solution."  Both arrows are unsafe:

  * Passing a LOOSE residual on a COARSE grid is weak evidence.  The discrete
    operator admits fixed points the continuum PDE does not.
  * A planform LABEL can be an artifact.  The old classifier reported a roll
    with a strong 2nd harmonic as rect(6,0), and a blend of two rolls as a
    single roll -- both look like novel modes and are not.
  * A "new" field may be a known branch under a SYMMETRY (translation /
    reflection), which is not a new solution at all.
  * The PROJECTOR may have done the discovering, not the generative model.

ON "INTERPOLATION"
--------------------------------------------------------------------------------
A convex mix a*u1 + (1-a)*u2 of two solutions of a NONLINEAR PDE is generically
NOT a solution: the advective term contributes cross terms a(1-a)(u1.grad u2 +
u2.grad u1) that do not cancel.  Consistently, the raw flow output has a large
residual (van_physL2 ~ 0.7).  So the honest mechanism is NOT "interpolation
produces valid solutions".  It is:

    the flow model produces an off-manifold interpolant  ->  Gauss-Newton
    root-finds the NEAREST zero of h(u), which may sit in a basin that was
    never seeded during data generation.

Because Newton does not care about dynamical stability, that zero can be an
UNSTABLE steady branch -- one a time-marching solver could never reach.  That
is a real and publishable capability, but the credit is shared: the model
supplies the initial guess, the projection supplies the solution.  This script
measures that split explicitly (see GATE 6).

THE EVIDENCE CHAIN (a candidate must pass ALL gates)
--------------------------------------------------------------------------------
  GATE 1  purity        harmonic-aware classifier, purity >= threshold
                        (calibrated on the training bank).  Rejects blends,
                        noise, and harmonic mislabels.  'mixed'/'unknown' fail.
  GATE 2  off-manifold  the label is not a training mode.
  GATE 3  residual      strict fp64 residual, discrete AND continuum, at or
                        below the level the REFERENCE data itself achieves.
  GATE 4  distinctness  after symmetry alignment (x/y translations and
                        reflections), rel-L2 to EVERY training branch at that
                        (Ra,Pr) exceeds --dist-tol.  Kills symmetry copies.
  GATE 5  refinement    Fourier-resample x2 and re-project; the residual must
                        NOT blow up.  A discrete-only artifact degrades here;
                        a continuum solution persists.
  GATE 6  isolation     perturb the candidate and re-run Gauss-Newton: it must
                        return to itself (an isolated root), not wander off.
                        Also reports how far the projector moved the raw
                        sample (credit attribution).

Only a candidate passing 1-6 warrants the claim, and even then it should be
confirmed by seeding an INDEPENDENT solver with the candidate's planform.

USAGE
    python novelty_audit.py --splits ./datasets/rb3d_hires/splits \\
        --ckpt ckpt_rb3d_cond.pt --k-samples 32 --purity-min 0.9
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import torch

from rb3d_pcfm_common import RB3DData, load_flow_model
from rb3d_pcfm_sampler import PCFMProjector, pcfm_sample
from rb3d_classify import classify_purity, calibrate_purity
from evaluate_rb3d import Residual, physics_L2, align_to, data_errors


# ---------------------------------------------------------------------------
def fourier_upsample(field, factor=2):
    """Exact Fourier interpolation in x,y (periodic). field: (4,Nx,Ny,Nz)."""
    C, Nx, Ny, Nz = field.shape
    nx, ny = int(Nx * factor), int(Ny * factor)
    out = torch.zeros(C, nx, ny, Nz, dtype=field.dtype)
    for c in range(C):
        ah = torch.fft.rfft2(field[c].double(), dim=(0, 1))
        kyc = ny // 2 + 1
        b = torch.zeros(nx, kyc, Nz, dtype=ah.dtype)
        keep = min(Nx, nx)
        pos = keep // 2 + 1
        neg = keep - pos
        kk = min(ah.shape[1], kyc)
        b[:pos, :kk] = ah[:pos, :kk]
        if neg > 0:
            b[nx - neg:, :kk] = ah[Nx - neg:, :kk]
        b = b * (nx * ny) / (Nx * Ny)
        out[c] = torch.fft.irfft2(b, s=(nx, ny), dim=(0, 1)).to(field.dtype)
    return out


def rel_residual(verifier, field, Ra, Pr):
    """Physics residual (physical L2) of a field, using the VERIFIER's operators
    -- the same residual the evaluation reports as physL2. NOTE: this must NOT
    use PCFMProjector.res, whose output is ROW-SCALED (divided by the per-block
    buoyancy/temperature norms) so that dividing it AGAIN by the buoyancy scale
    collapses it to ~0 -- that bug made the GATE-3 threshold 0.0000 and caused
    every candidate to fail automatically."""
    with torch.no_grad():
        rm, rt, rc = verifier(field, Ra, Pr)
        return physics_L2(rm, rt, rc)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--splits', default='./datasets/rb3d_multisolution/splits')
    p.add_argument('--ckpt', default='ckpt_rb3d_cond.pt')
    p.add_argument('--out', default='rb3d_novelty')
    p.add_argument('--k-samples', type=int, default=32)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--n-step', type=int, default=100)
    p.add_argument('--proj-start', type=float, default=0.6)
    p.add_argument('--cg-iters', type=int, default=15)
    p.add_argument('--n-newton', type=int, default=1)
    p.add_argument('--purity-min', type=float, default=0.90,
                   help='minimum spectral purity to trust a label. Training '
                        'branches sit at ~1.00; 0.90 leaves margin for the '
                        'generator\'s finite sharpness.')
    p.add_argument('--dist-tol', type=float, default=0.30,
                   help='min symmetry-aligned rel-L2 to every training branch')
    p.add_argument('--res-slack', type=float, default=1.5,
                   help='candidate residual may exceed the reference data\'s '
                        'own residual by at most this factor (GATE 3)')
    p.add_argument('--points', type=int, default=3,
                   help='how many (Ra,Pr) points to probe')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.splits, 'train_bank.pt')
    data = RB3DData(train_path, device=device)
    bank = torch.load(train_path, map_location='cpu', weights_only=False)
    train_modes = {tuple(m) for m in bank['mode_list']}
    model, ck = load_flow_model(args.ckpt, device)
    projector = PCFMProjector(data.Nx, data.Ny, data.Nz, data.aspect, device,
                              cg_iters=args.cg_iters)
    resid = Residual(data.Nx, data.Ny, data.Nz, data.aspect, device=device)

    thr_cal, per_mode = calibrate_purity(bank, quantile=0.05)
    purity_min = args.purity_min
    print(f'[calib] training-branch purity: 5th pct = {thr_cal:.4f}')
    for m, (mu, n) in sorted(per_mode.items()):
        print(f'          {str(m):>15}: mean purity {mu:.4f}  (n={n})')
    print(f'[calib] using purity_min = {purity_min:.2f}')
    print(f'[calib] training modes = {sorted(train_modes)}')

    # GATE-3 baseline: reuse the EVALUATION's calibrated per-planform validity
    # thresholds (1.5x the 95th-percentile of each training branch's own bulk
    # residual) rather than a from-scratch median -- keeps the audit's "real
    # solution" bar identical to the evaluation's "valid" bar and avoids
    # tiny-sample noise. `ref_level` (floor) = the loosest per-planform
    # threshold, used for off-target labels that have no training branch.
    from evaluate_rb3d import calibrate_thresholds
    thresholds, ref_level = calibrate_thresholds(data, resid)
    refs_by_pt = defaultdict(dict)
    z = data.z
    for (Ra, Pr, mode), e in bank['entries'].items():
        Tp = e['grid_T'].float() - (1.0 - z)[None, None, :]
        f = torch.stack([e['grid_u'].float(), e['grid_v'].float(),
                         e['grid_w'].float(), Tp])
        refs_by_pt[(Ra, Pr)][tuple(mode)] = f
    pts = sorted(refs_by_pt)[:args.points]
    print(f'[calib] GATE-3 residual floor = {ref_level:.4f}  (per-planform: '
          f'{ {str(k): round(v,3) for k,v in thresholds.items()} })')

    records, candidates = [], []
    for pi, (Ra, Pr) in enumerate(pts):
        refs = refs_by_pt[(Ra, Pr)]
        done = 0
        while done < args.k_samples:
            b = min(args.batch, args.k_samples - done)
            f_van, _ = pcfm_sample(model, ck, data, [Ra] * b, [Pr] * b,
                                   projector, n_step=args.n_step,
                                   vanilla=True, seed=args.seed + 1000 * pi + done)
            f_pcf, _ = pcfm_sample(model, ck, data, [Ra] * b, [Pr] * b,
                                   projector, n_step=args.n_step,
                                   proj_start=args.proj_start,
                                   n_newton=args.n_newton,
                                   seed=args.seed + 1000 * pi + done)
            for j in range(b):
                van, pcf = f_van[j], f_pcf[j]
                lab, pur, _ = classify_purity(pcf[2], *data.aspect,
                                              purity_min=purity_min)
                rr = rel_residual(resid, pcf.to(device), Ra, Pr)
                rec = dict(Ra=Ra, Pr=Pr, label=lab, purity=pur, res=rr)
                # ---- gates
                g1 = lab[0] not in ('unknown', 'mixed') and pur >= purity_min
                g2 = g1 and (tuple(lab) not in train_modes)
                g3 = rr <= thresholds.get(tuple(lab), ref_level)
                # distinctness to every training branch at this point
                dmin = math.inf
                for ref in refs.values():
                    _, l2 = align_to(pcf, ref)
                    a, _ = align_to(pcf, ref)
                    dmin = min(dmin, data_errors(a, ref)[1])
                g4 = dmin > args.dist_tol
                rec.update(g1_purity=g1, g2_offmanifold=g2, g3_residual=g3,
                           g4_distinct=g4, dist_min=dmin,
                           proj_disp=float((pcf - van).norm()
                                           / van.norm().clamp_min(1e-9)))
                records.append(rec)
                if g1 and g2 and g3 and g4:
                    candidates.append((rec, pcf.clone(), Ra, Pr))
            done += b
        sub = [r for r in records if r['Ra'] == Ra]
        print(f'[probe] Ra={Ra:7.0f} Pr={Pr:4.2f}  '
              f'labels={dict(Counter(str(r["label"]) for r in sub))}')

    print(f'\n[audit] {len(candidates)}/{len(records)} samples passed GATES 1-4')

    # ---- GATES 5 & 6 on survivors -----------------------------------------
    confirmed = []
    for (rec, field, Ra, Pr) in candidates:
        # GATE 5: refinement persistence
        up = fourier_upsample(field.cpu(), 2).to(device)
        pj2 = PCFMProjector(up.shape[1], up.shape[2], up.shape[3], data.aspect,
                            device, cg_iters=args.cg_iters)
        resid2 = Residual(up.shape[1], up.shape[2], up.shape[3], data.aspect,
                          device=device)
        r_before = rel_residual(resid2, up, Ra, Pr)
        Rat = torch.tensor([Ra], device=device)
        Prt = torch.tensor([Pr], device=device)
        with torch.enable_grad():
            up_p, _, _ = pj2.project(up[None], Rat, Prt, n_newton=2)
        r_after = rel_residual(resid2, up_p[0], Ra, Pr)
        g5 = r_after <= max(2.0 * rec['res'], args.res_slack * ref_level)

        # GATE 6: isolation -- perturb, re-project, must return
        eps = 0.05 * field.std()
        pert = (field + eps * torch.randn_like(field)).to(device)
        with torch.enable_grad():
            back, _, _ = projector.project(pert[None], Rat, Prt, n_newton=3)
        back = back[0]
        ret = float((back - field.to(device)).norm()
                    / field.norm().clamp_min(1e-9))
        g6 = ret < 0.10

        rec.update(g5_refinement=g5, r_refined=r_after,
                   g6_isolated=g6, return_dist=ret)
        status = 'CONFIRMED' if (g5 and g6) else 'rejected'
        print(f'  {str(rec["label"]):>16} Ra={Ra:.0f} Pr={Pr:.2f} purity={rec["purity"]:.3f} '
              f'res={rec["res"]:.4f} dist={rec["dist_min"]:.3f} '
              f'refined={r_after:.4f} return={ret:.3f} -> {status}')
        if g5 and g6:
            confirmed.append(rec)

    # ---- report ------------------------------------------------------------
    summary = dict(
        n_samples=len(records), purity_min=purity_min,
        ref_residual_level=ref_level, train_modes=[list(m) for m in train_modes],
        n_pass_gates_1_4=len(candidates), n_confirmed=len(confirmed),
        label_histogram={k: v for k, v in
                         Counter(str(r['label']) for r in records).items()},
        mean_proj_displacement=float(
            torch.tensor([r['proj_disp'] for r in records]).mean()),
        confirmed=[{k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in r.items()} for r in confirmed],
    )
    json.dump(summary, open(os.path.join(args.out, 'novelty_summary.json'), 'w'),
              indent=2, default=str)

    print('\n' + '=' * 74)
    print(f'label histogram      : {summary["label_histogram"]}')
    print(f'mean projection move : {summary["mean_proj_displacement"]:.3f}  '
          f'(large => the PROJECTOR, not the model, found the solution)')
    print(f'passed gates 1-4     : {len(candidates)}')
    print(f'CONFIRMED novel      : {len(confirmed)}')
    if confirmed:
        print('\nA confirmed candidate is a steady state of the DISCRETE system '
              'that is\noff-manifold, distinct under symmetry, survives grid '
              'refinement, and is an\nisolated root. To claim it as a NEW SOLUTION '
              'OF THE PDE, now seed an\nINDEPENDENT solver with its planform and '
              'show it converges there.')
    else:
        print('\nNo confirmed novel solution. The off-manifold labels seen in '
              'evaluation are\nexplained by low purity (blends), harmonic '
              'mislabelling, symmetry copies, or\nresiduals above the reference '
              'level -- i.e. artifacts, not discoveries.')
    print('=' * 74)


if __name__ == '__main__':
    main()