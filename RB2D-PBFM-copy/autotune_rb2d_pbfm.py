#!/usr/bin/env python3
"""
autotune_rb2d_pbfm.py
================================================================================
Measure, rather than guess, the largest PBFM training configuration your GPUs
can actually run -- and how fast each configuration is.

WHY THIS EXISTS
---------------
"Is global batch 64 the most we can do on 2xT4?" cannot be answered from theory:
peak memory depends on the unrolling depth, whether backprop goes through the
last step or all of them, the ConFIG double backward, and cuFFT workspace. So
this script runs the REAL training step (residual + unrolling + ConFIG, the same
code path as train_rb2d_pbfm_conditioned.py) at a grid of settings, records peak
CUDA memory and throughput, and reports what fits with margin.

IMPORTANT -- BIGGER BATCH IS NOT BETTER
---------------------------------------
The PBFM paper (Appendix G) deliberately keeps the GLOBAL batch size CONSTANT
regardless of GPU count, with a fixed learning rate, so that results do not
depend on the hardware. Raising the global batch changes the optimisation
problem and would need the LR retuned. Multi-GPU is therefore used to cut
WALL-CLOCK at fixed global batch -- not to inflate the batch.

So the right question is not "how large can the batch be" but "given the paper's
global batch, what is the best use of the spare memory". For RB the answer is
usually MORE UNROLLING: the paper shows unrolling is what reduces the physics
residual (it mitigates Jensen's gap), and residual is exactly what is too high
in the RB runs. This script therefore sweeps unroll as well as batch.

USAGE
-----
    python autotune_rb2d_pbfm.py --bank ./rb2d/split/train_bank.pt --gpus 2
    python autotune_rb2d_pbfm.py --batches 32,64,128 --unrolls 1,2,4,6

Each measurement runs on ONE GPU with that GPU's share of the global batch
(batch//gpus), which is exactly what each rank does in a real run.
"""
import argparse
import time

import torch

import rb2d_pbfm_common as C


def try_config(data, per_gpu_batch, unroll, backprop, device, cond=True,
               hidden=64, modes_x=20, modes_z=12, n_layers=4, time_emb=32,
               steps=6, warm=2):
    """Run a few REAL PBFM steps. Returns (ok, peak_GB, sec_per_iter, err)."""
    try:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        sigma = data.sigma.to(device).view(1, 3, 1, 1)
        fields = data.fields.to(device)
        Ra = torch.tensor([p[0] for p in data.params], device=device)
        Pr = torch.tensor([p[1] for p in data.params], device=device)
        cond_all = torch.tensor(
            [[data.lognorm_Ra(p[0]), data.lognorm_Pr(p[1])] for p in data.params],
            device=device) if cond else None
        zc = (1.0 - data.z.to(device))[None, None, :]

        model = C.FlowFNO2d(hidden, modes_x, modes_z, n_layers, time_emb,
                            in_ch=3, cond=cond).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0,
                                betas=(0.5, 0.999))
        params = list(model.parameters())

        t0 = None
        for s in range(steps):
            idx = torch.randint(0, data.N, (per_gpu_batch,), device=device)
            x1 = fields[idx] / sigma
            cb = cond_all[idx] if cond else None
            eps = torch.randn_like(x1)
            t = C.sample_logit_normal_t(per_gpu_batch, device)
            xt = (1 - t)[:, None, None, None] * eps + t[:, None, None, None] * x1
            tgt = x1 - eps

            opt.zero_grad(set_to_none=True)
            v = model(xt, t, cb)
            loss_fm = ((v - tgt) ** 2).mean()

            # unrolled x~1 (Algorithm 1)
            dt = (1.0 - t).clamp_min(1e-4) / unroll
            et = t.clone()
            xp = xt + dt[:, None, None, None] * v
            for i in range(1, unroll):
                et = et + dt
                if backprop == 'last' and i < unroll - 1:
                    with torch.no_grad():
                        vv = model(xp, et, cb)
                else:
                    vv = model(xp, et, cb)
                xp = xp + dt[:, None, None, None] * vv
            f = xp * sigma
            r = C.steady_residual_rel(f[:, 0], f[:, 1], f[:, 2] + zc,
                                      Ra[idx], Pr[idx], data.dx, data.dz)
            loss_r = ((t ** 1.0) ** 2 * r['sq']).mean()

            # ConFIG: two backward passes (this is the memory/time peak)
            loss_fm.backward(retain_graph=True)
            g_fm = C._flat_grads(params)
            opt.zero_grad(set_to_none=True)
            loss_r.backward()
            g_r = C._flat_grads(params)
            C._unflat_to_grads(C.config_direction(g_fm, g_r), params)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if s == warm - 1:
                if device.startswith('cuda'):
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        spi = (time.perf_counter() - t0) / max(1, steps - warm)
        peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 3
                if device.startswith('cuda') else float('nan'))
        del model, opt, fields
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        return True, peak, spi, ''
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return False, float('nan'), float('nan'), 'OOM'
    except RuntimeError as e:
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        msg = str(e)
        return False, float('nan'), float('nan'), \
            ('OOM' if 'out of memory' in msg.lower() else msg[:60])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', default='./rb2d/split/train_bank.pt')
    p.add_argument('--gpus', type=int, default=None)
    p.add_argument('--batches', default='32,64,128,256',
                   help='GLOBAL batch sizes to try')
    p.add_argument('--unrolls', default='1,2,4,6')
    p.add_argument('--backprop', default='last', choices=['last', 'all'])
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--modes-x', dest='modes_x', type=int, default=20)
    p.add_argument('--modes-z', dest='modes_z', type=int, default=12)
    p.add_argument('--n-layers', dest='n_layers', type=int, default=4)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--margin', type=float, default=0.85,
                   help='fraction of GPU memory considered safe (default 0.85)')
    args = p.parse_args()

    gpus = args.gpus or (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    gpus = max(1, gpus)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print('[autotune] no CUDA visible -- memory numbers will be NaN and '
              'timings are not representative of a T4.')
    else:
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024 ** 3
        print(f'[autotune] {torch.cuda.device_count()} GPU(s) visible; '
              f'using {gpus}. GPU0 = {props.name}, {total:.1f} GB')
    total = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
             if device != 'cpu' else float('nan'))

    data = C.RB2DData(C.find_file(args.bank) or args.bank, device='cpu')
    print(f'[autotune] grid {data.Nx}x{data.Nz}, {data.N} fields, '
          f'backprop={args.backprop}')
    print(f'[autotune] each row runs ONE rank\'s real work = global_batch/{gpus}\n')

    batches = [int(x) for x in args.batches.split(',')]
    unrolls = [int(x) for x in args.unrolls.split(',')]

    print(f'{"global":>7} {"per-GPU":>8} {"unroll":>7} {"peak GB":>9} '
          f'{"s/iter":>8} {"it/s":>7} {"samples/s":>10}  status')
    results = []
    for gb in batches:
        pg = max(1, gb // gpus)
        for un in unrolls:
            ok, peak, spi, err = try_config(
                data, pg, un, args.backprop, device,
                hidden=args.hidden, modes_x=args.modes_x, modes_z=args.modes_z,
                n_layers=args.n_layers, steps=args.steps)
            if ok:
                its = 1.0 / spi
                print(f'{gb:>7} {pg:>8} {un:>7} {peak:>9.2f} {spi:>8.3f} '
                      f'{its:>7.2f} {gb*its:>10.1f}  ok')
                results.append((gb, pg, un, peak, spi, its))
            else:
                print(f'{gb:>7} {pg:>8} {un:>7} {"--":>9} {"--":>8} '
                      f'{"--":>7} {"--":>10}  {err}')

    if results and total == total:
        safe = [r for r in results if r[3] <= args.margin * total]
        print(f'\n[autotune] memory budget: {args.margin:.0%} of {total:.1f} GB '
              f'= {args.margin*total:.1f} GB')
        if safe:
            deep = max(safe, key=lambda r: (r[2], -r[3]))
            print(f'[autotune] deepest unrolling that fits with margin: '
                  f'global batch {deep[0]}, unroll {deep[2]}  '
                  f'({deep[3]:.2f} GB, {deep[5]:.2f} it/s)')
            at64 = [r for r in safe if r[0] == 64]
            if at64:
                b = max(at64, key=lambda r: r[2])
                print(f'[autotune] at the PAPER global batch of 64, the largest '
                      f'unrolling that fits is {b[2]} '
                      f'({b[3]:.2f} GB, {b[5]:.2f} it/s)')
                print(f'[autotune] RECOMMENDED (paper-faithful, spends spare '
                      f'memory on physics rather than batch):\n'
                      f'    python train_rb2d_pbfm_conditioned.py --gpus {gpus} '
                      f'--batch 64 --unroll {b[2]} --unroll-start 1 '
                      f'--backprop {args.backprop}')
        else:
            print('[autotune] nothing fit inside the margin; reduce --hidden or '
                  'use --backprop last.')
    print('\n[autotune] NOTE: raising the GLOBAL batch above 64 departs from the '
          'paper setup and would require re-tuning the learning rate. Prefer '
          'spending spare memory on --unroll, which is what lowers the residual.')


if __name__ == '__main__':
    main()