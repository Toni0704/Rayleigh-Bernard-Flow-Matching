#!/usr/bin/env python3
"""
find_max_batch.py
================================================================================
Empirically determine the largest --batch that fits on THIS GPU, for the
current grid/hidden/modes config, WITHOUT waiting for a real training run to
OOM hours in. Doubles the batch size until it fails, then binary-searches the
gap. Reports peak memory at the chosen batch so you can judge headroom.

Runs a handful of real forward+backward steps (the actual cfm_loss / FNO3d /
autograd graph, not a rough proxy), so the number is trustworthy -- guessing
from a parameter count is not a substitute for asking the GPU directly.

USAGE
    python find_max_batch.py --bank ./datasets/rb3d_hires/splits/train_bank.pt
    python find_max_batch.py --bank train_bank.pt --hidden 32 --cond
    python find_max_batch.py --bank train_bank.pt --amp        # test with AMP on
"""

import argparse
import gc

import torch

from rb3d_pcfm_common import RB3DData, FNO3d, cfm_loss, default_cfg


def try_batch(data, cfg, cond, device, batch, amp, n_steps=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model = FNO3d(cfg, cond=cond, Nz=data.Nz).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    try:
        for _ in range(n_steps):
            x1, c = data.batch(batch, augment=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=amp):
                loss = cfm_loss(model, x1, c if cond else None)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        ok = True
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        ok = False
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return ok, peak


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--hidden', type=int, default=default_cfg()['hidden'])
    p.add_argument('--modes-x', type=int, default=default_cfg()['modes_x'])
    p.add_argument('--modes-y', type=int, default=default_cfg()['modes_y'])
    p.add_argument('--modes-z', type=int, default=default_cfg()['modes_z'])
    p.add_argument('--n-layers', type=int, default=default_cfg()['n_layers'])
    p.add_argument('--cond', action='store_true', default=True)
    p.add_argument('--uncond', dest='cond', action='store_false')
    p.add_argument('--amp', action='store_true',
                   help='probe with mixed precision (roughly halves memory, '
                        'usually the batch you actually want to train with)')
    p.add_argument('--start', type=int, default=4)
    p.add_argument('--headroom', type=float, default=0.85,
                   help='recommend a batch using this fraction of the GPU\'s '
                        'total memory, leaving margin for eval-time spikes')
    args = p.parse_args()

    assert torch.cuda.is_available(), 'no CUDA device visible'
    total = torch.cuda.get_device_properties(args.device).total_memory / 1e9
    print(f'[probe] {torch.cuda.get_device_name(args.device)}: {total:.1f} GB total, '
          f'AMP={"on" if args.amp else "off"}')

    # data stays in CPU RAM; only per-iteration batches go to the GPU. (The
    # single-copy RB3DData holds ~4*Nx*Ny*Nz*N*4 bytes once -- ~8.6 GB for the
    # hi-res train bank -- so keep it off the GPU and out of a second copy.)
    data = RB3DData(args.bank, device=args.device)
    # cuFFT half-precision FFTs require power-of-two dims; the FNO's rfftn under
    # autocast will crash otherwise. Match the trainer: silently drop AMP so the
    # probe reports the batch you can ACTUALLY train at.
    def _p2(n):
        return (n & (n - 1)) == 0
    if args.amp and not (_p2(data.Nx) and _p2(data.Ny) and _p2(data.Nz)):
        print(f'[probe] AMP disabled: grid {data.Nx}x{data.Ny}x{data.Nz} has a '
              f'non-power-of-two dim (cuFFT cannot do half-precision FFTs on '
              f'it). Probing in fp32 -- the number below is your real ceiling.')
        args.amp = False
    cfg = default_cfg()
    cfg.update(hidden=args.hidden, modes_x=args.modes_x, modes_y=args.modes_y,
              modes_z=args.modes_z, n_layers=args.n_layers)

    # phase 1: double until it breaks (or memory exceeds the card)
    b = args.start
    last_ok, last_peak = None, None
    while True:
        ok, peak = try_batch(data, cfg, args.cond, args.device, b, args.amp)
        print(f'  batch={b:4d}: {"OK " if ok else "OOM"}  peak={peak:.2f} GB')
        if not ok:
            break
        last_ok, last_peak = b, peak
        b *= 2
        if b > 4096:
            break
    if last_ok is None:
        print(f'[probe] even batch={args.start} does not fit. Try --hidden '
              f'smaller or --amp.')
        return
    lo, hi = last_ok, b

    # phase 2: binary search the gap
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, peak = try_batch(data, cfg, args.cond, args.device, mid, args.amp)
        print(f'  batch={mid:4d}: {"OK " if ok else "OOM"}  peak={peak:.2f} GB')
        if ok:
            lo, last_peak = mid, peak
        else:
            hi = mid

    print(f'\n[probe] max batch that fits (3-step test): {lo}  '
          f'(peak {last_peak:.2f} GB / {total:.1f} GB = '
          f'{100*last_peak/total:.0f}%)')
    # recommend a safety-margined batch: real training holds a few more
    # buffers (EMA snapshot each eval, cudnn workspace) than this short probe
    rec = max(args.start, int(lo * args.headroom))
    print(f'[probe] recommended --batch for real training: {rec}  '
          f'(applies a {100*(1-args.headroom):.0f}% safety margin for '
          f'eval-time memory spikes not exercised by this probe)')
    if not args.amp:
        print('[probe] re-run with --amp to see the batch you could reach '
              'with mixed precision (usually ~1.6-2x larger, and faster '
              'per-step on T4 tensor cores).')


if __name__ == '__main__':
    main()