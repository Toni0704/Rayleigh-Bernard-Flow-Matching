#!/usr/bin/env python3
"""
train_rb3d_pcfm_conditioned.py
================================================================================
Train the (Ra,Pr)-CONDITIONED flow-matching model on the RB3D multi-solution
training split -- the recommended variant (3D analog of the 2D conditioned
model). Learns p(u,v,w,T' | Ra,Pr): a 5-planform conditional distribution over
the coexisting branches {x-rolls 2,3,4; y-rolls 2,3}. The branch label is NEVER
shown -- mode diversity comes purely from the noise seed.

Run (Kaggle T4):
    python train_rb3d_pcfm_conditioned.py \
        --bank ./datasets/rb3d_multisolution/splits/train_bank.pt
    python train_rb3d_pcfm_conditioned.py --quick     # smoke test

After training a short physics-relaxed sanity sample is drawn and its
planforms / relaxation rel-change printed, so a broken checkpoint is caught
immediately (healthy: planforms within the trained set, rel-change < ~1e-3).
"""

import argparse
import os

import numpy as np
import torch

from rb3d_pcfm_common import (RB3DData, RB3DRelaxer, default_cfg,
                              sample_fields, train_flow_model)


def parse_args():
    cfg = default_cfg()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank',
                   default='./datasets/rb3d_multisolution/splits/train_bank.pt')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default=None,
                   help='cuda:0 | cuda:1 | cpu (default: auto). Put the two '
                        'models on separate GPUs to train both at full speed.')
    p.add_argument('--ckpt-name', default='ckpt_rb3d_cond.pt')
    p.add_argument('--iters', type=int, default=cfg['iters'])
    p.add_argument('--batch', type=int, default=cfg['batch'])
    p.add_argument('--lr', type=float, default=cfg['lr'])
    p.add_argument('--hidden', type=int, default=cfg['hidden'])
    p.add_argument('--modes-x', type=int, default=cfg['modes_x'])
    p.add_argument('--modes-y', type=int, default=cfg['modes_y'])
    p.add_argument('--modes-z', type=int, default=cfg['modes_z'])
    p.add_argument('--n-layers', type=int, default=cfg['n_layers'])
    p.add_argument('--time-emb', type=int, default=cfg['time_emb'])
    p.add_argument('--seed', type=int, default=cfg['seed'])
    p.add_argument('--no-augment', dest='augment', action='store_false')
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--amp', action='store_true',
                   help='mixed precision: ~1.6-2x larger batch fits, and '
                        'faster steps on T4 tensor cores. Use '
                        'find_max_batch.py --amp first to size --batch.')
    p.add_argument('--no-sanity', dest='sanity', action='store_false')
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()
    if args.quick:
        args.iters = 400
        args.eval_every = 100
        args.hidden, args.modes_x, args.modes_y, args.modes_z = 16, 8, 4, 4
        args.batch = 4
    return args


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out = args.out or ('/kaggle/working' if os.path.isdir('/kaggle/working')
                       else '.')
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if str(device).startswith('cuda'):
        torch.backends.cudnn.benchmark = True
        di = torch.device(device).index or 0
        torch.cuda.set_device(di)
        torch.cuda.reset_peak_memory_stats(di)
        print(f'[env] {torch.cuda.get_device_name(di)} (cuda:{di})')

    cfg = default_cfg()
    cfg.update(iters=args.iters, batch=args.batch, lr=args.lr,
               hidden=args.hidden, modes_x=args.modes_x, modes_y=args.modes_y,
               modes_z=args.modes_z, n_layers=args.n_layers,
               time_emb=args.time_emb, seed=args.seed, augment=args.augment)
    print(f'[cfg] {cfg}')

    data = RB3DData(args.bank, device=device, val_frac=cfg['val_frac'],
                    seed=args.seed)
    ckpt_path = os.path.join(out, args.ckpt_name)
    model, info = train_flow_model(data, cfg, cond=True, device=device,
                                   ckpt_path=ckpt_path,
                                   eval_every=args.eval_every,
                                   patience=args.patience, amp=args.amp)
    if str(device).startswith('cuda'):
        di = torch.device(device).index or 0
        print(f'[env] peak GPU memory: '
              f'{torch.cuda.max_memory_allocated(di)/1e9:.2f} GB')

    if args.sanity:
        print('\n[sanity] 5 physics-relaxed samples at a mid-box point ...')
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        from rb3d_pcfm_common import load_flow_model
        model, ck = load_flow_model(ckpt_path, device)
        relaxer = RB3DRelaxer(data, device)
        Ra0 = float(np.exp(0.5 * (np.log(data.Ra_rng[0])
                                  + np.log(data.Ra_rng[1]))))
        Pr0 = float(np.exp(0.5 * (np.log(data.Pr_rng[0])
                                  + np.log(data.Pr_rng[1]))))
        rs = 300 if args.quick else cfg['relax_steps']
        fields, diag = sample_fields(model, ck, data, [Ra0] * 5, [Pr0] * 5,
                                     relaxer=relaxer, n_step=cfg['n_step'],
                                     relax_steps=rs, seed=0)
        print(f'[sanity] Ra={Ra0:.0f} Pr={Pr0:.2f} | '
              f'planforms={diag["planform"]} | '
              f'relax rel-change='
              f'{[f"{r:.1e}" for r in diag["relax_rel_change"].tolist()]} | '
              f'{diag["seconds"]:.1f}s for 5 samples')
        print('[sanity] healthy: planforms within the trained roll set and '
              'rel-change well below 1e-2 (sample sits on a steady branch).')


if __name__ == '__main__':
    main()