#!/usr/bin/env python3
"""
rb3d_pcfm_common.py
================================================================================
Shared library for the RB3D multimodal flow-matching surrogate: dataset,
exact-symmetry augmentation, FNO3d velocity model, OT conditional flow-matching
training (EMA + early stopping + resume), and physics-projected sampling.

Mirrors rb2d_pcfm_common.py, upgraded one dimension:
  * fields are (4, Nx, Ny, Nz): (u, v, w, T') with T' = T - (1-z)
  * FNO2d -> FNO3d (modes_x, modes_y, modes_z)
  * augmentation: exact x/y-translations and x/y-reflections (the symmetry
    group of RB in a periodic rectangular box; NO x<->y swap -- the box is
    rectangular so that is not a symmetry)
  * the sampling-time physics projection is PSEUDO-TIME RELAXATION with the
    verified generator solver (imported from generate_rb3d_multisolution):
    steady branches are attractors of the IMEX+Leray flow, so a short
    relaxation at the target (Ra,Pr) snaps a generated sample onto the
    nearest steady branch and enforces incompressibility + walls exactly --
    the role PCFM's Gauss-Newton projection played in 1D/2D.

The (Ra,Pr)-CONDITIONED model receives log-normalised (Ra,Pr) as constant
input channels; the UNCONDITIONED model learns the marginal over the whole
bank and meets (Ra,Pr) only through the relaxation, exactly as in Bratu.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from generate_rb3d_multisolution import (build_vertical, _kxy, leray_project,
                                         imex_step, classify_planform)


# ============================================================================
#  Config
# ============================================================================
def default_cfg():
    return dict(
        iters=30000, batch=12, lr=1e-3, wd=1e-5,
        hidden=32, modes_x=16, modes_y=8, modes_z=8, n_layers=4, time_emb=8,
        ema=0.999, val_frac=0.05, seed=0, augment=True,
        # sampling
        n_step=50,
        relax_steps=1500, relax_cfl=0.30, relax_check=250,
    )


# ============================================================================
#  Data
# ============================================================================
class RB3DData:
    """Loads a split bank (train_bank.pt). Exposes:
       fields  (N,4,Nx,Ny,Nz) float32, T' channel (base profile removed)
       params  (N,2) float32  log-normalised (Ra,Pr) in [-1,1]
       scale   (4,) per-channel std used for normalisation
    """

    def __init__(self, path, device='cpu', val_frac=0.05, seed=0):
        bank = torch.load(path, map_location='cpu', weights_only=False)
        ent = bank['entries']
        keys = [k for k, e in ent.items() if e.get('converged', True)]
        keys.sort()
        g = bank['grid']
        self.Nx, self.Ny, self.Nz = g['Nx'], g['Ny'], g['Nz']
        self.aspect = tuple(bank['aspect'])
        self.mode_list = [tuple(m) for m in bank['mode_list']]
        z = torch.linspace(0.0, 1.0, self.Nz)
        self.z = z
        # Store the data EXACTLY ONCE. The old code kept fields_raw AND a
        # normalised copy (2x), while torch.load's bank dict was still alive
        # during construction (3x transient) -- ~26 GB for the hi-res train
        # bank, which OOM-kills the process on a ~13 GB Kaggle box. Here we
        # move each entry into one preallocated array, release the source as we
        # go, then normalise IN PLACE.
        N = len(keys)
        self.fields = torch.empty(N, 4, self.Nx, self.Ny, self.Nz,
                                  dtype=torch.float32)
        P = []
        base = (1.0 - z)[None, None, :]
        for i, k in enumerate(keys):
            e = ent[k]
            self.fields[i, 0] = e['grid_u'].float()
            self.fields[i, 1] = e['grid_v'].float()
            self.fields[i, 2] = e['grid_w'].float()
            self.fields[i, 3] = e['grid_T'].float() - base
            P.append([k[0], k[1]])
            ent[k] = None                    # release source tensors early
        self.keys = keys
        del bank, ent
        import gc as _gc; _gc.collect()
        pr = torch.tensor(P, dtype=torch.float64)
        self.Ra_rng = (float(pr[:, 0].min()), float(pr[:, 0].max()))
        self.Pr_rng = (float(pr[:, 1].min()), float(pr[:, 1].max()))
        self.params = self.norm_params(pr[:, 0], pr[:, 1]).float()
        self.Ra = pr[:, 0].float()
        self.Pr = pr[:, 1].float()
        # scale-only normalisation (keeps translation/reflection augments
        # EXACT); done IN PLACE, channel by channel, so no second copy exists.
        self.scale = self.fields.std(dim=(0, 2, 3, 4)).clamp_min(1e-8)
        for c in range(4):
            self.fields[:, c] /= self.scale[c]
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(N, generator=gen)
        n_val = max(8, int(val_frac * N))
        self.val_idx = perm[:n_val]
        self.train_idx = perm[n_val:]
        self.device = device
        print(f'[data] {path}: {N} entries '
              f'({len(self.train_idx)} train / {len(self.val_idx)} val), '
              f'grid {self.Nx}x{self.Ny}x{self.Nz}, '
              f'Ra {self.Ra_rng[0]:.0f}-{self.Ra_rng[1]:.0f}, '
              f'Pr {self.Pr_rng[0]:.2f}-{self.Pr_rng[1]:.2f}, '
              f'~{self.fields.numel()*4/1e9:.1f} GB in RAM')
        print(f"[data] channel scales (u,v,w,T'): "
              f'{[round(float(s),3) for s in self.scale]}')

    @property
    def fields_raw(self):
        """Back-compat: de-normalised fields, reconstructed on demand (used by
        diagnostics only). Materialising this doubles memory; on large banks
        prefer `self.fields[idx] * self.scale[...]` on just the slice needed."""
        return self.fields * self.scale[None, :, None, None, None]

    def norm_params(self, Ra, Pr):
        lr0, lr1 = math.log(5000.0), math.log(30000.0)
        lp0, lp1 = math.log(0.5), math.log(7.0)
        a = 2 * (torch.log(torch.as_tensor(Ra, dtype=torch.float64)) - lr0) \
            / (lr1 - lr0) - 1
        b = 2 * (torch.log(torch.as_tensor(Pr, dtype=torch.float64)) - lp0) \
            / (lp1 - lp0) - 1
        return torch.stack([a, b], dim=-1)

    def batch(self, bs, augment=True, val=False, generator=None):
        idx_pool = self.val_idx if val else self.train_idx
        sel = idx_pool[torch.randint(len(idx_pool), (bs,), generator=generator)]
        x = self.fields[sel].clone()
        c = self.params[sel].clone()
        if augment:
            # exact symmetries of RB in the periodic rectangular box
            for b in range(bs):
                sx = int(torch.randint(self.Nx, (1,), generator=generator))
                sy = int(torch.randint(self.Ny, (1,), generator=generator))
                x[b] = torch.roll(x[b], shifts=(sx, sy), dims=(1, 2))
                if torch.rand((), generator=generator) < 0.5:   # x-reflection
                    x[b] = torch.flip(x[b], dims=(1,))
                    x[b, 0] = -x[b, 0]                          # u -> -u
                if torch.rand((), generator=generator) < 0.5:   # y-reflection
                    x[b] = torch.flip(x[b], dims=(2,))
                    x[b, 1] = -x[b, 1]                          # v -> -v
        return x.to(self.device), c.to(self.device)


# ============================================================================
#  FNO3d velocity model
# ============================================================================
class SpectralConv3d(nn.Module):
    def __init__(self, ch, mx, my, mz):
        super().__init__()
        self.ch, self.mx, self.my, self.mz = ch, mx, my, mz
        sc = 1.0 / (ch * ch)
        # 4 corner blocks: (+-kx, +-ky, kz>=0) -- z axis is rfft-halved
        self.w = nn.ParameterList([
            nn.Parameter(sc * torch.randn(ch, ch, mx, my, mz, 2))
            for _ in range(4)])

    def forward(self, x):
        B, C, Nx, Ny, Nz = x.shape
        xh = torch.fft.rfftn(x, dim=(2, 3, 4))
        out = torch.zeros(B, C, Nx, Ny, Nz // 2 + 1,
                          dtype=xh.dtype, device=x.device)
        mx, my, mz = self.mx, self.my, self.mz

        def mul(block, w):
            wc = torch.view_as_complex(w)
            return torch.einsum('bixyz,ioxyz->boxyz', block, wc)

        out[:, :, :mx, :my, :mz] = mul(xh[:, :, :mx, :my, :mz], self.w[0])
        out[:, :, -mx:, :my, :mz] = mul(xh[:, :, -mx:, :my, :mz], self.w[1])
        out[:, :, :mx, -my:, :mz] = mul(xh[:, :, :mx, -my:, :mz], self.w[2])
        out[:, :, -mx:, -my:, :mz] = mul(xh[:, :, -mx:, -my:, :mz], self.w[3])
        return torch.fft.irfftn(out, s=(Nx, Ny, Nz), dim=(2, 3, 4))


class FNO3d(nn.Module):
    """Velocity field v_theta(x_t, t [, cond]) for flow matching."""

    def __init__(self, cfg, cond=True, Nz=25):
        super().__init__()
        self.cond = cond
        F = cfg['time_emb']
        self.freqs = nn.Parameter(2.0 ** torch.arange(F).float() * math.pi,
                                  requires_grad=False)
        in_ch = 4 + 1 + 2 * F + (2 if cond else 0)   # state + zcoord + temb + cond
        h = cfg['hidden']
        self.lift = nn.Conv3d(in_ch, h, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv3d(h, cfg['modes_x'], cfg['modes_y'], cfg['modes_z'])
             for _ in range(cfg['n_layers'])])
        self.point = nn.ModuleList(
            [nn.Conv3d(h, h, 1) for _ in range(cfg['n_layers'])])
        self.proj = nn.Sequential(nn.Conv3d(h, h, 1), nn.GELU(),
                                  nn.Conv3d(h, 4, 1))
        zc = torch.linspace(0, 1, Nz)
        self.register_buffer('zcoord', zc.view(1, 1, 1, 1, Nz))

    def forward(self, x, t, cond=None):
        B, C, Nx, Ny, Nz = x.shape
        ang = t.view(B, 1) * self.freqs[None, :]
        temb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)   # (B,2F)
        temb = temb.view(B, -1, 1, 1, 1).expand(-1, -1, Nx, Ny, Nz)
        zc = self.zcoord.expand(B, 1, Nx, Ny, Nz)
        feats = [x, zc, temb]
        if self.cond:
            cc = cond.view(B, 2, 1, 1, 1).expand(-1, -1, Nx, Ny, Nz)
            feats.append(cc)
        hdn = self.lift(torch.cat(feats, dim=1))
        for sp, pw in zip(self.spectral, self.point):
            hdn = torch.nn.functional.gelu(sp(hdn) + pw(hdn))
        return self.proj(hdn)


# ============================================================================
#  Training (OT conditional flow matching)
# ============================================================================
def cfm_loss(model, x1, cond, generator=None):
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device, generator=generator)
    x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
    xt = (1 - t.view(-1, 1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1, 1) * x1
    v = model(xt, t, cond)
    return (v - (x1 - x0)).pow(2).mean()


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


def train_flow_model(data, cfg, cond, device, ckpt_path,
                     eval_every=500, patience=8, amp=False):
    """amp=True enables mixed precision (autocast + GradScaler): roughly halves
    activation memory and speeds up matmul/FFT-heavy ops on T4 tensor cores,
    letting a larger --batch fit in the same GPU memory. Safe to toggle
    independently of --batch; use find_max_batch.py --amp to see how much
    headroom it buys on your actual GPU before committing to a batch size."""
    torch.manual_seed(cfg['seed'])
    model = FNO3d(cfg, cond=cond, Nz=data.Nz).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f'[train] FNO3d {"COND" if cond else "UNCOND"} '
          f'{n_par/1e6:.2f}M params, {cfg["iters"]} iters, batch {cfg["batch"]}, '
          f'amp={"on" if amp else "off"}')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['wd'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['iters'])
    ema = EMA(model, cfg['ema'])
    amp_device = 'cuda' if str(device).startswith('cuda') else 'cpu'
    # cuFFT can only do HALF-PRECISION FFTs when every transformed dimension is
    # a power of two. The FNO's rfftn runs under autocast, so on a grid like
    # 128x64x49 (49 is not a power of two) AMP crashes with "cuFFT only supports
    # dimensions whose sizes are powers of two when computing in half
    # precision". Detect this and fall back to full precision rather than die.
    def _p2(n):
        return (n & (n - 1)) == 0
    if amp and not (_p2(data.Nx) and _p2(data.Ny) and _p2(data.Nz)):
        print(f'[train] AMP requested but grid {data.Nx}x{data.Ny}x{data.Nz} '
              f'has a non-power-of-two dimension; cuFFT cannot do half-'
              f'precision FFTs on it. Disabling AMP (training in fp32). '
              f'This is a hard cuFFT limitation, not a config error.',
              flush=True)
        amp = False
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and amp_device == 'cuda'))
    is_cuda = str(device).startswith('cuda')

    start_it, best, bad = 0, float('inf'), 0
    last_path = ckpt_path.replace('.pt', '_last.pt')
    resume_from = last_path if os.path.exists(last_path) else ckpt_path
    if os.path.exists(resume_from):
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        if ck.get('cfg', {}).get('hidden') == cfg['hidden'] and \
           ck.get('cond') == cond and 'opt' in ck:
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            ema.shadow = ck['ema']
            start_it, best = ck['iter'], ck['best']
            bad = ck.get('bad', 0)
            if 'scaler' in ck and amp:
                scaler.load_state_dict(ck['scaler'])
            print(f'[train] resumed from {os.path.basename(resume_from)} '
                  f'@ iter {start_it} (best {best:.4f}, bad {bad}/{patience})')

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    it = start_it
    for it in range(start_it, cfg['iters']):
        x1, c = data.batch(cfg['batch'], augment=cfg['augment'])
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(amp_device, enabled=(amp and is_cuda)):
            loss = cfm_loss(model, x1, c if cond else None)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                      # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        sched.step(); ema.update(model)

        if (it + 1) % 100 == 0:
            dt = (time.perf_counter() - t0) / (it + 1 - start_it)
            eta = dt * (cfg['iters'] - it - 1) / 60
            mem = (f' peak={torch.cuda.max_memory_allocated(device)/1e9:.2f}GB'
                  if is_cuda else '')
            print(f'  it {it+1:6d}/{cfg["iters"]} loss={float(loss.detach()):.4f} '
                  f'{1e3*dt:.0f} ms/it ({cfg["batch"]/dt:.1f} fields/s) '
                  f'ETA {eta:.0f}m{mem}', flush=True)

        if (it + 1) % eval_every == 0:
            model_ema = FNO3d(cfg, cond=cond, Nz=data.Nz).to(device)
            model_ema.load_state_dict(ema.shadow)
            model_ema.eval()
            with torch.no_grad():
                g = torch.Generator(device='cpu').manual_seed(1234)
                vs = []
                for _ in range(4):
                    xv, cv = data.batch(min(16, cfg['batch'] * 2),
                                        augment=False, val=True)
                    gv = torch.Generator(device=xv.device).manual_seed(
                        int(torch.randint(2**31, (1,), generator=g)))
                    vs.append(float(cfm_loss(model_ema, xv,
                                             cv if cond else None,
                                             generator=gv)))
                vloss = float(np.mean(vs))
            improved = vloss < best - 1e-4
            if improved:
                best, bad = vloss, 0
            else:
                bad += 1
            payload = dict(model=model.state_dict(), ema=ema.shadow,
                           opt=opt.state_dict(), sched=sched.state_dict(),
                           iter=it + 1, best=best, bad=bad, cfg=cfg, cond=cond,
                           scale=data.scale, Nz=data.Nz,
                           Ra_rng=data.Ra_rng, Pr_rng=data.Pr_rng,
                           scaler=scaler.state_dict() if amp else None)
            # always refresh the "last" checkpoint (crash/timeout safety);
            # overwrite the "best" checkpoint only when val improves.
            torch.save(payload, last_path)
            if improved:
                torch.save(payload, ckpt_path)
            mem = (f' mem={torch.cuda.memory_allocated(device)/1e9:.2f}/'
                  f'{torch.cuda.max_memory_allocated(device)/1e9:.2f}GB'
                  if is_cuda else '')
            print(f'  [eval] it {it+1} val={vloss:.4f} best={best:.4f} '
                  f'{"saved-best" if improved else f"(bad {bad}/{patience})"} '
                  f'[last saved]{mem}', flush=True)
            if bad >= patience:
                print(f'[train] early stop at iter {it+1}')
                break
    print(f'[train] done ({(time.perf_counter()-t0)/60:.1f} min), '
          f'best val {best:.4f} -> {ckpt_path}')
    return model, dict(best=best, iters=it + 1)


def load_flow_model(path, device):
    assert os.path.exists(path), f'checkpoint not found: {path}'
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FNO3d(ck['cfg'], cond=ck['cond'], Nz=ck['Nz']).to(device)
    model.load_state_dict(ck['ema'])           # sample from the EMA weights
    model.eval()
    return model, ck


# ============================================================================
#  Sampling: flow integration + pseudo-time physics relaxation
# ============================================================================
class RB3DRelaxer:
    """Physics projection for generated samples: a short IMEX+Leray pseudo-
    time relaxation at the target (Ra,Pr). Enforces incompressibility and
    wall BCs exactly (every step projects), and pulls the sample onto the
    nearest steady branch (steady states are the attractors)."""

    def __init__(self, data_or_ck, device):
        # accept RB3DData or any lightweight view exposing Nx,Ny,Nz,aspect
        if hasattr(data_or_ck, 'Nx') and hasattr(data_or_ck, 'aspect'):
            self.Nx, self.Ny, self.Nz = (data_or_ck.Nx, data_or_ck.Ny,
                                         data_or_ck.Nz)
            self.aspect = data_or_ck.aspect
        else:
            raise ValueError('pass an RB3DData or a grid view '
                             '(needs Nx,Ny,Nz,aspect)')
        Gx, Gy = self.aspect
        self.dx, self.dy = Gx / self.Nx, Gy / self.Ny
        self.dz = 1.0 / (self.Nz - 1)
        self.device = device
        self.VS = build_vertical(self.Nz, device, torch.float32)
        kx, ky = _kxy(self.Nx, self.Ny, self.dx, self.dy, device, torch.float32)
        self.kx, self.ky = kx, ky
        self.k2 = kx[:, None] ** 2 + ky[None, :] ** 2

    def relax(self, u, v, w, Tp, Ra_l, Pr_l, steps, cfl=0.30, check=250,
              early_tol=2e-4):
        """u..Tp: (B,Nx,Ny,Nz) float32 on device. Returns relaxed fields and
        per-sample rel-change diagnostics. Early-stops the WHOLE batch once
        every sample's relative change per check-interval is < early_tol (most
        samples reach steady state well before the step budget -- big eval
        speedup with no accuracy loss)."""
        B = u.shape[0]
        Ra = torch.as_tensor(Ra_l, device=self.device, dtype=torch.float32)
        Pr = torch.as_tensor(Pr_l, device=self.device, dtype=torch.float32)
        u, v, w = leray_project(u, v, w, self.VS, self.k2, self.kx, self.ky)
        min_d = min(self.dx, self.dy, self.dz)
        rel = torch.full((B,), float('inf'), device=self.device)
        prev = torch.cat([f.flatten(1) for f in (u, v, w, Tp)], 1)
        s = 0
        while s < steps:
            umax = torch.stack([u, v, w]).abs().flatten(2).amax(2).amax(0)
            dt = cfl * min_d / torch.clamp(1.5 * umax, min=10.0)
            n = min(check, steps - s)
            for _ in range(n):
                u, v, w, Tp = imex_step(u, v, w, Tp, self.VS, self.kx, self.ky,
                                        self.k2, self.dx, self.dy, self.dz,
                                        dt, Pr, Ra)
                u, v, w = leray_project(u, v, w, self.VS, self.k2,
                                        self.kx, self.ky)
            s += n
            cur = torch.cat([f.flatten(1) for f in (u, v, w, Tp)], 1)
            if not torch.isfinite(cur).all():          # a diverging sample:
                bad = ~torch.isfinite(cur).all(dim=1)  # freeze it at prev
                cur[bad] = prev[bad]
                sz = (self.Nx, self.Ny, self.Nz)
                P = self.Nx * self.Ny * self.Nz
                u[bad] = prev[bad, :P].view(-1, *sz)
                v[bad] = prev[bad, P:2 * P].view(-1, *sz)
                w[bad] = prev[bad, 2 * P:3 * P].view(-1, *sz)
                Tp[bad] = prev[bad, 3 * P:].view(-1, *sz)
            rel = (cur - prev).abs().amax(1) / cur.abs().amax(1).clamp_min(1e-6)
            prev = cur
            if float(rel.max()) < early_tol:           # whole batch steady
                break
        return u, v, w, Tp, rel


@torch.no_grad()
def sample_fields(model, ck, data, Ra_l, Pr_l, relaxer=None, n_step=50,
                  relax_steps=1500, relax_cfl=0.30, seed=0):
    """Full pipeline: flow-match sample -> denormalise -> physics relaxation.
    Returns fields (B,4,Nx,Ny,Nz) [u,v,w,T'] and diagnostics dict."""
    device = next(model.parameters()).device
    B = len(Ra_l)
    t0 = time.perf_counter()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(B, 4, data.Nx, data.Ny, data.Nz, generator=gen).to(device)
    cond = data.norm_params(torch.tensor(Ra_l, dtype=torch.float64),
                            torch.tensor(Pr_l, dtype=torch.float64)) \
        .float().to(device)
    cond = cond if ck['cond'] else None
    ts = torch.linspace(0, 1, n_step + 1, device=device)
    for i in range(n_step):                                  # Heun
        t_a = ts[i].expand(B)
        dt = float(ts[i + 1] - ts[i])
        v1 = model(x, t_a, cond)
        x_e = x + dt * v1
        v2 = model(x_e, ts[i + 1].expand(B), cond)
        x = x + 0.5 * dt * (v1 + v2)
    scale = ck['scale'].to(device)
    f = x * scale[None, :, None, None, None]                 # denormalise
    u, v, w, Tp = f[:, 0].contiguous(), f[:, 1].contiguous(), \
        f[:, 2].contiguous(), f[:, 3].contiguous()
    diag = {}
    if relaxer is not None and relax_steps > 0:
        u, v, w, Tp = (t.float() for t in (u, v, w, Tp))
        u, v, w, Tp, rel = relaxer.relax(u, v, w, Tp, Ra_l, Pr_l,
                                         relax_steps, cfl=relax_cfl)
        diag['relax_rel_change'] = rel.cpu()
    fields = torch.stack([u, v, w, Tp], dim=1)
    pf = classify_planform(w, data.aspect[0], data.aspect[1])
    diag['planform'] = pf
    diag['seconds'] = time.perf_counter() - t0
    return fields.cpu(), diag