#!/usr/bin/env python3
"""
diagnose_training.py
================================================================================
Answer "did my model actually train?" by inspecting the RAW (vanilla) flow
output, before any PCFM correction. This separates "the model learned the
branches" from "the projection did the work".

Checks (all fast -- pure network sampling, no relaxation):
  1. checkpoint header      : best val loss, iters trained, early-stopped?
  2. field realism          : peak/mean of PRE-relaxation samples vs the true
                              training-data scale (a collapsed-to-mean model
                              produces near-zero-amplitude blur)
  3. vanilla planform      : what planforms the RAW (uncorrected) samples
                              A trained multimodal model should produce SEVERAL
                              distinct roll planforms across seeds BEFORE any
                              relaxation. If everything is noise/one blob here,
                              the model didn't learn -- the eval numbers are all
                              the relaxer.
  4. conditioning response  : does changing (Ra,Pr) change the sample amplitude
                              the way physics demands (peak ~ sqrt(Ra))? Only
                              meaningful for the conditioned model.
  5. projection displacement: how FAR the Gauss-Newton correction moves a sample. Small
                              => the model already sits near the constraint
                              manifold (good). Large => PCFM is doing the work.

USAGE
    python diagnose_training.py --splits ./datasets/rb3d_multisolution/splits \
        --cond-ckpt ckpt_rb3d_cond.pt --uncond-ckpt ckpt_rb3d_uncond.pt
"""

import argparse
import math
import os

import numpy as np
import torch
from collections import Counter

from rb3d_pcfm_common import RB3DData, load_flow_model
from rb3d_pcfm_sampler import PCFMProjector, pcfm_sample
from verify_rb3d_multisolution import classify


def hdr(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck['cfg']
    print(f'  best val loss : {ck["best"]:.4f}')
    print(f'  iters trained : {ck["iter"]}  (of {cfg["iters"]} planned) '
          f'{"[EARLY-STOPPED]" if ck["iter"] < cfg["iters"] else "[full]"}')
    print(f'  model         : hidden={cfg["hidden"]} modes='
          f'({cfg["modes_x"]},{cfg["modes_y"]},{cfg["modes_z"]}) '
          f'layers={cfg["n_layers"]}')
    return ck


def assess(name, model, ck, data, device, args):
    print(f'\n{"="*66}\n{name.upper()} MODEL\n{"="*66}')
    print(f'[1] checkpoint header:')
    hdr({'cond': args.cond_ckpt, 'uncond': args.uncond_ckpt}[name], device)

    # true data scale for reference
    tr_peak = float(data.fields_raw[:, 2].abs().flatten(1).amax(1).mean())
    print(f'\n[2] field realism (raw flow output, {args.n} samples @ mid-box):')
    Ra0 = float(np.exp(0.5 * (np.log(data.Ra_rng[0]) + np.log(data.Ra_rng[1]))))
    Pr0 = float(np.exp(0.5 * (np.log(data.Pr_rng[0]) + np.log(data.Pr_rng[1]))))
    proj = PCFMProjector(data.Nx, data.Ny, data.Nz, data.aspect, device,
                         cg_iters=args.cg_iters)
    fpre, dpre = pcfm_sample(model, ck, data, [Ra0] * args.n, [Pr0] * args.n,
                             proj, n_step=args.n_step, vanilla=True, seed=0)
    dpre = dict(planform=[classify(fpre[b, 2].double(), *data.aspect)
                          for b in range(fpre.shape[0])])
    gen_peak = fpre[:, 2].abs().flatten(1).amax(1)
    print(f'  generated peak|w| : mean={float(gen_peak.mean()):.2f} '
          f'(range {float(gen_peak.min()):.1f}-{float(gen_peak.max()):.1f})')
    print(f'  training peak|w|  : {tr_peak:.2f}')
    ratio = float(gen_peak.mean()) / max(tr_peak, 1e-9)
    verdict = ('GOOD (realistic amplitude)' if 0.4 < ratio < 2.5
               else 'SUSPECT: amplitude far from data '
                    '(collapsed to mean? under-trained?)')
    print(f'  amplitude ratio   : {ratio:.2f}  -> {verdict}')

    print(f'\n[3] VANILLA (raw flow) planform diversity (the key test):')
    pfs = dpre['planform']
    from collections import Counter
    cnt = Counter(str(p) for p in pfs)
    print(f'  raw-sample planforms: {dict(cnt)}')
    n_distinct = len(cnt)
    # how "roll-like" -- a real branch has ONE dominant wavevector on an axis
    roll_like = sum(1 for p in pfs if p[0] == 'roll')
    print(f'  distinct planforms  : {n_distinct}   roll-like: {roll_like}/{len(pfs)}')
    if n_distinct >= 3 and roll_like >= len(pfs) // 2:
        print('  -> GOOD: raw generator already produces several roll branches '
              '(multimodality learned, not just relaxer).')
    elif roll_like >= len(pfs) // 2:
        print('  -> PARTIAL: roll-like but low diversity (mode collapse? '
              'try more iters / stronger augment).')
    else:
        print('  -> POOR: raw samples are not roll-like. The model is '
              'under-trained; downstream valid% would be the RELAXER, not the '
              'model. Train longer / check loss curve.')

    if name == 'cond':
        print(f'\n[4] conditioning response (peak|w| should grow ~sqrt(Ra)):')
        for Ra in (data.Ra_rng[0], Ra0, data.Ra_rng[1]):
            f, _ = pcfm_sample(model, ck, data, [Ra] * args.n, [Pr0] * args.n,
                               proj, n_step=args.n_step, vanilla=True, seed=1)
            pk = float(f[:, 2].abs().flatten(1).amax(1).mean())
            print(f'  Ra={Ra:7.0f}: mean peak|w|={pk:6.2f}  '
                  f'(sqrt(Ra)={math.sqrt(Ra):.1f})')
        print('  -> peak should INCREASE with Ra if conditioning works.')

    print(f'\n[5] PCFM projection displacement (how far the correction moves a '
          f'sample -- small = the raw flow already sits near the manifold):')
    nb = min(args.n, 6)
    f_post, d_post = pcfm_sample(model, ck, data, [Ra0] * nb, [Pr0] * nb, proj,
                                 n_step=args.n_step, proj_start=args.proj_start,
                                 n_newton=1, seed=0)
    disp = (f_post - fpre[:nb]).flatten(1).norm(dim=1) / \
        fpre[:nb].flatten(1).norm(dim=1).clamp_min(1e-9)
    print(f'  |PCFM - vanilla| / |vanilla| : mean={float(disp.mean()):.3f} '
          f'max={float(disp.max()):.3f}')
    print(f'  residual  before -> after    : '
          f'{float(d_post["res_before"].mean()):.3e} -> '
          f'{float(d_post["res_after"].mean()):.3e}')
    pf_post = [classify(f_post[b, 2].double(), *data.aspect)
               for b in range(nb)]
    print(f'  post-projection planforms    : '
          f'{dict(Counter(str(p) for p in pf_post))}')
    print('  (PCFM moves the sample MINIMALLY -- unlike a solver, which marches '
          'to an attractor. Large displacement => the flow model is far off.)')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--splits', default='./datasets/rb3d_multisolution/splits')
    p.add_argument('--cond-ckpt', default='ckpt_rb3d_cond.pt')
    p.add_argument('--uncond-ckpt', default='ckpt_rb3d_uncond.pt')
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--n-step', type=int, default=50)
    p.add_argument('--proj-start', type=float, default=0.6)
    p.add_argument('--cg-iters', type=int, default=8)
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = RB3DData(os.path.join(args.splits, 'train_bank.pt'), device=device)
    for name, path in (('cond', args.cond_ckpt), ('uncond', args.uncond_ckpt)):
        if os.path.exists(path):
            model, ck = load_flow_model(path, device)
            assess(name, model, ck, data, device, args)
        else:
            print(f'[skip] {name}: {path} not found')
    print(f'\n{"="*66}\nBOTTOM LINE: if [3] says GOOD for at least the conditioned '
          f'model,\nthe generator learned the branches and the pipeline is '
          f'sound.\nIf [3] says POOR, train longer before trusting eval '
          f'numbers.\n{"="*66}')


if __name__ == '__main__':
    main()
