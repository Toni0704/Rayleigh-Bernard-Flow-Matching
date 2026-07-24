#!/usr/bin/env python3
"""
rb3d_pbfm_common.py
================================================================================
Shared library for PHYSICS-BASED FLOW MATCHING (PBFM) on 3-D steady
Rayleigh-Benard convection. The 3-D analog of RB2D-PBFM-copy/rb2d_pbfm_common.py,
built to REUSE the existing 3-D PCFM code rather than duplicate it, so PCFM and
PBFM are architecturally identical and directly comparable:

  * `FNO3d`, `RB3DData`, `EMA`, `load_flow_model`      <- imported from
    rb3d_pcfm_common.py (no re-implementation: one architecture definition).
  * the differentiable physics residual `h(u)` (Leray-projected bulk momentum +
    temperature, correctly row-scaled, wall-masked)    <- imported from
    `rb3d_pcfm_sampler.SteadyResidual`. It was built for PCFM's Gauss-Newton
    projector, but it is already exactly what PBFM's training-time residual
    loss needs (batched, differentiable, matches the generator's operators).
    SteadyResidual deliberately leaves out continuity (PCFM enforces div(u)=0
    by exactly Leray-projecting the *state*, outside Newton); PBFM's headline
    sampler has no such projection step, so a continuity term is added here via
    `divergence_rms` (generate_rb3d_multisolution.py, also batched/differentiable).
  * `RB3DRelaxer`                                       <- imported for the
    optional sampling-time ablations only (PBFM's headline path is
    projection-free).

The crucial difference from PCFM is *when* physics is imposed:
    PCFM : plain flow-matching training; physics enforced at SAMPLING time by
           an iterative Gauss-Newton / IMEX projection (expensive inference).
    PBFM : physics enforced at TRAINING time by an unrolled residual loss
           combined conflict-free (ConFIG) with the flow-matching loss;
           sampling is a plain ODE integration with NO projection.

Checkpoint format matches rb3d_pcfm_common.train_flow_model's payload exactly
(`model, ema, opt, sched, iter, best, bad, cfg, cond, scale, Nz, Ra_rng, Pr_rng`)
so `rb3d_pcfm_common.load_flow_model` loads PBFM checkpoints unmodified --
vanilla, PBFM and PCFM are all one architecture, one loader.
"""

import os
import time

import torch

# RB3DData/load_flow_model are re-exported (unused directly in this file) so
# the training script can do `import rb3d_pbfm_common as C; C.RB3DData(...)`,
# one import surface, matching the rb2d_pbfm_common.py convention.
from rb3d_pcfm_common import FNO3d, RB3DData, EMA, RB3DRelaxer, load_flow_model  # noqa: F401
from rb3d_pcfm_sampler import SteadyResidual
from generate_rb3d_multisolution import divergence_rms, leray_project

ARCH_VERSION = 'rb3d-pbfm-v1'


# ============================================================================
#  0.  Multi-GPU (dual T4) helpers -- ported verbatim from rb2d_pbfm_common.py.
#      Generic (no RB-dimension-specific logic); kept here rather than a
#      shared import so this file has no dependency on the 2-D folder.
#
#      Design note -- why NO DistributedDataParallel wrapper: ConFIG needs the
#      FM gradient and the residual gradient SEPARATELY (two backward passes
#      per iteration). DDP's autograd hooks all-reduce on the first backward
#      and raise "reducer has already been marked ready" on the second unless
#      wrapped in no_sync() -- at which point DDP is doing nothing for us
#      anyway. Instead: every rank holds an identical replica, each loss's
#      gradients are flattened into ONE contiguous real vector and reduced
#      with exactly one all_reduce per loss.
# ============================================================================
def ddp_is_torchrun():
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def ddp_setup(rank, world_size, backend=None, port='29500'):
    if world_size <= 1:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    import torch.distributed as dist
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', port)
    if backend is None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    if not dist.is_initialized():
        kw = {}
        if backend == 'nccl' and torch.cuda.is_available():
            try:
                kw['device_id'] = torch.device(f'cuda:{rank}')
            except Exception:
                pass
        try:
            dist.init_process_group(backend=backend, rank=rank,
                                    world_size=world_size, **kw)
        except TypeError:
            dist.init_process_group(backend=backend, rank=rank,
                                    world_size=world_size)
    if torch.cuda.is_available():
        return f'cuda:{rank}'
    return 'cpu'


def ddp_cleanup():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main(rank=0):
    return rank == 0


def ddp_allreduce_mean_(vec):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        vec.div_(dist.get_world_size())
    return vec


def ddp_broadcast_flag(flag, src=0, device='cpu'):
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return flag
    t = torch.tensor([1 if flag else 0], device=device)
    dist.broadcast(t, src=src)
    return bool(t.item())


def launch(fn, gpus, *args):
    """Run fn(rank, world_size, *args) on `gpus` processes. Works under
    torchrun (uses its env) and standalone (spawns)."""
    if ddp_is_torchrun():
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        return fn(rank, world, *args)
    if gpus is None:
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpus = max(1, int(gpus))
    if gpus == 1:
        return fn(0, 1, *args)
    import torch.multiprocessing as mp
    mp.spawn(fn, args=(gpus, *args), nprocs=gpus, join=True)


# ============================================================================
#  1.  ConFIG conflict-free gradient combination (Liu et al. 2025a) -- ported
#      verbatim from rb2d_pbfm_common.py (generic, dtype-agnostic).
# ============================================================================
def _flat_grads(params):
    gs = []
    for p in params:
        g = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        if torch.is_complex(g):
            g = torch.view_as_real(g)
        gs.append(g.reshape(-1))
    return torch.cat(gs)


def _unflat_to_grads(vec, params):
    i = 0
    for p in params:
        if torch.is_complex(p):
            n = p.numel() * 2
            chunk = vec[i:i + n].reshape(*p.shape, 2).contiguous()
            p.grad = torch.view_as_complex(chunk).clone()
        else:
            n = p.numel()
            p.grad = vec[i:i + n].view_as(p).clone()
        i += n


def config_direction(g_fm, g_r, eps=1e-12):
    """gv = U[U(O(g_fm,g_r)) + U(O(g_r,g_fm))]; g = (g_fm.gv + g_r.gv) gv."""
    def U(g):
        n = g.norm()
        return g / n if n > eps else g

    def O(a, b):
        aa = (a * a).sum().clamp_min(eps)
        return b - ((a * b).sum() / aa) * a

    n_fm, n_r = g_fm.norm(), g_r.norm()
    if n_fm < eps:
        return g_r.clone()
    if n_r < eps:
        return g_fm.clone()
    gv = U(U(O(g_fm, g_r)) + U(O(g_r, g_fm)))
    scale = (g_fm * gv).sum() + (g_r * gv).sum()
    return scale * gv


# ============================================================================
#  2.  Config
# ============================================================================
def default_cfg_pbfm():
    """PCFM's default_cfg() layered with PBFM-specific keys. lr/optimizer
    settings follow the PBFM paper's Appendix G (fixed LR, beta1=0.5, wd=0),
    same as RB2D-PBFM-copy's defaults."""
    from rb3d_pcfm_common import default_cfg
    cfg = default_cfg()
    cfg.update(
        lr=3e-4, wd=0.0, adam_b1=0.5, adam_b2=0.999, lr_schedule='constant',
        unroll=2, unroll_start=None, curriculum_end=None, backprop='last',
        resid_p=1.0, use_config=True, resid_weight=1.0, warmup_iters=1000,
        grad_clip=1.0, physics=True,
        early_stop_metric='gen_res',        # 'fm' or 'gen_res'
        val_gen_samples=8, val_gen_steps=25, val_resid_samples=16,
        val_every=500, patience=8,
    )
    return cfg


# ============================================================================
#  3.  Batch sampling -- mirrors RB3DData.batch's augmentation exactly, but
#      also returns the RAW (Ra,Pr) needed by the physics residual (the
#      governing equations use physical Ra,Pr, not the log-normalised cond
#      vector RB3DData.batch returns).
# ============================================================================
def sample_batch_3d(data, bs, augment=True, val=False, generator=None):
    idx_pool = data.val_idx if val else data.train_idx
    sel = idx_pool[torch.randint(len(idx_pool), (bs,), generator=generator)]
    x = data.fields[sel].clone()
    c = data.params[sel].clone()
    ra = data.Ra[sel].clone()
    pr = data.Pr[sel].clone()
    if augment:
        for b in range(bs):
            sx = int(torch.randint(data.Nx, (1,), generator=generator))
            sy = int(torch.randint(data.Ny, (1,), generator=generator))
            x[b] = torch.roll(x[b], shifts=(sx, sy), dims=(1, 2))
            if torch.rand((), generator=generator) < 0.5:      # x-reflection
                x[b] = torch.flip(x[b], dims=(1,))
                x[b, 0] = -x[b, 0]
            if torch.rand((), generator=generator) < 0.5:      # y-reflection
                x[b] = torch.flip(x[b], dims=(2,))
                x[b, 1] = -x[b, 1]
    dev = data.device
    return x.to(dev), c.to(dev), ra.to(dev), pr.to(dev)


def sample_logit_normal_t(B, device, generator=None):
    """t = sigmoid(N(0,1)) -> logit-normal on (0,1), concentrated near 0.5
    (PBFM Algorithm 1 default; matches rb2d_pbfm_common.py)."""
    return torch.sigmoid(torch.randn(B, device=device, generator=generator))


# ============================================================================
#  4.  Differentiable 3-D physics residual: SteadyResidual (momentum, temp)
#      + divergence_rms (continuity), combined the way RB2D's steady_residual_rel
#      combines cont/temp/vort: per-sample relative residuals, `combined`=max,
#      `sq`=sum-of-squares (the training-loss aggregate).
# ============================================================================
def residual_rel_3d(state, Ra, Pr, res):
    """state:(B,4,Nx,Ny,Nz) physical (u,v,w,T'). Ra,Pr:(B,) tensors.
    Returns dict of per-sample (B,) relative residuals: mom, temp, cont,
    combined(=max), sq(=mom^2+temp^2+cont^2). Differentiable w.r.t. state."""
    h = res(state, Ra, Pr)                              # (B,4,Nx,Ny,Nz)
    I = slice(1, -1)                                    # interior z, drop walls
    # momentum rows are already wall_cut-masked inside SteadyResidual (matches
    # evaluate_rb3d.py::Residual's c=3 bulk trim); temperature isn't, so slice
    # it here to match evaluate_rb3d.py::Residual's _rms_i (interior) convention.
    mom = h[:, :3].pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    temp = h[:, 3, :, :, I].pow(2).mean(dim=(1, 2, 3)).sqrt()
    u, v, w = state[:, 0], state[:, 1], state[:, 2]
    cont = divergence_rms(u, v, w, res.VS, res.gkx, res.gky)
    combined = torch.stack([mom, temp, cont], dim=0).max(dim=0).values
    sq = mom ** 2 + temp ** 2 + cont ** 2
    return dict(mom=mom, temp=temp, cont=cont, combined=combined, sq=sq)


# ============================================================================
#  5.  PBFM training (Algorithm 1: unroll + residual loss + ConFIG)
# ============================================================================
def unroll_predict(model, x_t, t, cond_b, n_steps, backprop='last'):
    """Evolve x_t at time t to t=1 in n_steps Euler substeps."""
    dt = (1.0 - t).clamp_min(1e-4) / n_steps
    et = t.clone()
    v = model(x_t, et, cond_b)
    x1 = x_t + dt.view(-1, 1, 1, 1, 1) * v
    for i in range(1, n_steps):
        et = et + dt
        if backprop == 'last' and i < n_steps - 1:
            with torch.no_grad():
                vv = model(x1, et, cond_b)
        else:
            vv = model(x1, et, cond_b)
        x1 = x1 + dt.view(-1, 1, 1, 1, 1) * vv
    return v, x1


def train_pbfm_3d(data, cfg, cond, device, ckpt_path, eval_every=500,
                  patience=8, rank=0, world_size=1, resume=True):
    """PBFM training loop (multi-GPU aware, dual-T4 ready).

      L_FM = || v(x_t,t) - (x1-eps) ||^2
      L_R  = mean( (t_start^resid_p)^2 * combined_residual_sq )
             on the UNROLLED prediction x~1, physics deferred until
             `cfg['warmup_iters']`.
      combined conflict-free via ConFIG (cfg['use_config']=True) or a fixed
      weight cfg['resid_weight'].

    Checkpointing: every `eval_every` iters, `<ckpt_path minus .pt>_last.pt`
    (full resume state: model/opt/sched/ema/iter/best/bad/history/arch) is
    OVERWRITTEN UNCONDITIONALLY -- a timeout loses at most eval_every iters --
    and `ckpt_path` (EMA weights, PCFM-payload-compatible) is overwritten only
    on validation improvement. Auto-resumes from `_last.pt` unless resume=False.
    A resume file whose `arch`/hidden/cond doesn't match the current network is
    archived (`.stale`) and training restarts clean instead of crashing.
    """
    torch.manual_seed(cfg['seed'])
    dev = device
    main = is_main(rank)

    global_batch = cfg['batch']
    if global_batch % world_size != 0 and main:
        print(f'[train] NOTE: global batch {global_batch} not divisible by '
              f'{world_size} GPUs; rounding down.')
    batch = max(1, global_batch // world_size)

    model = FNO3d(cfg, cond=cond, Nz=data.Nz).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    if main:
        print(f'[train] PBFM FNO3d {"COND" if cond else "UNCOND"} '
              f'{n_par/1e6:.2f}M params | world_size={world_size} device={dev} '
              f'| global batch={global_batch} (per-GPU {batch})')
    ema = EMA(model, cfg['ema'])
    # PBFM paper App. G: AdamW, wd=0, beta1=0.5, beta2=0.999, fixed LR.
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                            weight_decay=cfg['wd'],
                            betas=(cfg['adam_b1'], cfg['adam_b2']))
    if cfg['lr_schedule'] == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['iters'])
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    params = list(model.parameters())

    res = SteadyResidual(data.Nx, data.Ny, data.Nz, data.aspect, dev,
                         dtype=torch.float32)
    scale = data.scale.to(dev).view(1, 4, 1, 1, 1)

    start_it, best, bad = 0, float('inf'), 0
    history = {'it': [], 'fm': [], 'res': [], 'val': []}
    last_path = ckpt_path.replace('.pt', '_last.pt')
    if resume and os.path.exists(last_path):
        try:
            ck = torch.load(last_path, map_location=dev, weights_only=False)
            ok = (ck.get('arch') == ARCH_VERSION
                 and ck.get('cfg', {}).get('hidden') == cfg['hidden']
                 and ck.get('cond') == cond and 'opt' in ck)
            if not ok:
                raise RuntimeError(
                    f"arch/config mismatch (got arch={ck.get('arch')!r}, "
                    f"want {ARCH_VERSION!r})")
            model.load_state_dict(ck['model'])
            opt.load_state_dict(ck['opt'])
            sched.load_state_dict(ck['sched'])
            ema.shadow = ck['ema']
            start_it, best = ck['iter'], ck['best']
            bad = ck.get('bad', 0)
            history = ck.get('history', history)
            if main:
                print(f'[train] resumed from {os.path.basename(last_path)} '
                      f'@ iter {start_it} (best {best:.4e}, bad {bad}/{patience})')
        except Exception as e:
            if main:
                stale = last_path + '.stale'
                try:
                    os.replace(last_path, stale)
                except Exception:
                    stale = '(could not rename)'
                print(f'[train] *** cannot resume from {last_path}: {e}')
                print(f'[train] *** archived to {stale}; STARTING FROM SCRATCH.')
            start_it, best, bad = 0, float('inf'), 0
            history = {'it': [], 'fm': [], 'res': [], 'val': []}

    torch.manual_seed(cfg['seed'] + 1000 * rank + 7 * start_it)

    # ---- FIXED validation problem (deterministic eps/t, frozen once) --------
    v_sel = data.val_idx
    gcpu = torch.Generator().manual_seed(cfg['seed'] + 12345)
    v_eps = torch.randn(v_sel.numel(), 4, data.Nx, data.Ny, data.Nz,
                        generator=gcpu).to(dev)
    v_t = torch.sigmoid(torch.randn(v_sel.numel(), generator=gcpu)).to(dev)
    v_res_cap = min(cfg['val_resid_samples'], v_sel.numel())
    v_gen_cap = min(cfg['val_gen_samples'], v_sel.numel())

    def resid_loss_fn(x1_norm, ra, pr, t_start):
        f = x1_norm * scale
        r = residual_rel_3d(f, ra, pr, res)
        wt = t_start.clamp_min(1e-3) ** cfg['resid_p']
        return (wt ** 2 * r['sq']).mean(), float(r['combined'].mean().detach())

    model.train()
    n_bad_res = 0
    t0 = time.perf_counter()
    for it in range(start_it, cfg['iters']):
        x1, cond_b, ra, pr = sample_batch_3d(data, batch, augment=cfg['augment'])
        cond_arg = cond_b if cond else None
        eps = torch.randn_like(x1)
        t = sample_logit_normal_t(batch, dev)
        x_t = (1 - t).view(-1, 1, 1, 1, 1) * eps + t.view(-1, 1, 1, 1, 1) * x1
        target = x1 - eps

        if cfg['unroll_start'] is None or cfg['unroll_start'] >= cfg['unroll']:
            n_unroll = cfg['unroll']
        else:
            c_end = cfg['curriculum_end'] or int(0.5 * cfg['iters'])
            frac = min(1.0, max(0.0, (it - cfg['warmup_iters'])
                                / max(1, c_end - cfg['warmup_iters'])))
            n_unroll = int(round(cfg['unroll_start']
                                 + frac * (cfg['unroll'] - cfg['unroll_start'])))
        do_res = cfg['physics'] and (it >= cfg['warmup_iters'])
        if main and cfg['physics'] and do_res and it == cfg['warmup_iters']:
            print(f'[train] --- residual loss ON at it={it} (unroll->{cfg["unroll"]}, '
                  f'p={cfg["resid_p"]}, '
                  f'{"ConFIG" if cfg["use_config"] else f"fixed w={cfg["resid_weight"]}"}) ---',
                  flush=True)

        opt.zero_grad(set_to_none=True)
        v = model(x_t, t, cond_arg)
        loss_fm = ((v - target) ** 2).mean()

        if do_res:
            _, x1p = unroll_predict(model, x_t, t, cond_arg, n_unroll,
                                    backprop=cfg['backprop'])
            loss_res, res_val = resid_loss_fn(x1p, ra, pr, t)
            if not torch.isfinite(loss_res):
                n_bad_res += 1
                if main and n_bad_res in (1, 10, 100, 1000):
                    print(f'[train] non-finite residual at it={it} '
                          f'(count={n_bad_res}); skipping physics term', flush=True)
                loss_fm.backward()
                if world_size > 1:
                    gg = _flat_grads(params)
                    ddp_allreduce_mean_(gg)
                    _unflat_to_grads(gg, params)
                res_val = float('nan')
            elif cfg['use_config']:
                loss_fm.backward(retain_graph=True)
                g_fm = _flat_grads(params)
                opt.zero_grad(set_to_none=True)
                loss_res.backward()
                g_r = _flat_grads(params)
                if world_size > 1:
                    ddp_allreduce_mean_(g_fm)
                    ddp_allreduce_mean_(g_r)
                g = config_direction(g_fm, g_r)
                _unflat_to_grads(g, params)
            else:
                (loss_fm + cfg['resid_weight'] * loss_res).backward()
                if world_size > 1:
                    gg = _flat_grads(params)
                    ddp_allreduce_mean_(gg)
                    _unflat_to_grads(gg, params)
        else:
            loss_fm.backward()
            if world_size > 1:
                gg = _flat_grads(params)
                ddp_allreduce_mean_(gg)
                _unflat_to_grads(gg, params)
            res_val = float('nan')

        torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip'])
        opt.step()
        sched.step()
        ema.update(model)

        if (it % cfg.get('log_every', 100) == 0 or it == cfg['iters'] - 1) and main:
            history['it'].append(it)
            history['fm'].append(float(loss_fm.detach()))
            history['res'].append(res_val)
            spd = (it - start_it + 1) / max(1e-9, time.perf_counter() - t0)
            res_s = (f'{res_val:.4e}' if res_val == res_val
                     else ('off(FM baseline)' if not cfg['physics']
                           else f'warmup({max(0, cfg["warmup_iters"] - it)} left)'))
            print(f'[train] it={it:6d} fm={float(loss_fm.detach()):.4e} '
                  f'res={res_s} lr={sched.get_last_lr()[0]:.2e} '
                  f'{spd:.2f} it/s (global batch {global_batch})', flush=True)

        if (it + 1) % eval_every == 0 or it == cfg['iters'] - 1:
            stop = False
            if main:
                model_ema = FNO3d(cfg, cond=cond, Nz=data.Nz).to(dev)
                model_ema.load_state_dict(ema.shadow)
                model_ema.eval()
                with torch.no_grad():
                    tot, cnt = 0.0, 0
                    for s in range(0, v_sel.numel(), batch):
                        sl = slice(s, min(s + batch, v_sel.numel()))
                        vb = v_sel[sl]
                        xv = data.fields[vb].to(dev)
                        cv = data.params[vb].to(dev) if cond else None
                        ev, tv = v_eps[sl], v_t[sl]
                        xtv = ((1 - tv).view(-1, 1, 1, 1, 1) * ev
                               + tv.view(-1, 1, 1, 1, 1) * xv)
                        pv = model_ema(xtv, tv, cv)
                        tot += float(((pv - (xv - ev)) ** 2).mean()) * xv.shape[0]
                        cnt += xv.shape[0]
                    val = tot / max(cnt, 1)

                    val_res = float('nan')
                    if do_res and v_res_cap > 0:
                        vb = v_sel[:v_res_cap]
                        xv = data.fields[vb].to(dev)
                        cv = data.params[vb].to(dev) if cond else None
                        ev, tv = v_eps[:v_res_cap], v_t[:v_res_cap]
                        xtv = ((1 - tv).view(-1, 1, 1, 1, 1) * ev
                               + tv.view(-1, 1, 1, 1, 1) * xv)
                        _, x1v = unroll_predict(model_ema, xtv, tv, cv, cfg['unroll'])
                        fv = x1v * scale
                        rv = residual_rel_3d(fv, data.Ra[vb].to(dev),
                                             data.Pr[vb].to(dev), res)
                        val_res = float(rv['combined'].mean())

                    # generation residual: integrate from PURE noise (what the
                    # sampler will actually produce) -- the deployment metric.
                    val_gen = float('nan')
                    if v_gen_cap > 0:
                        vb = v_sel[:v_gen_cap]
                        cv = data.params[vb].to(dev) if cond else None
                        ggen = torch.Generator().manual_seed(cfg['seed'] + 999)
                        xg = torch.randn(v_gen_cap, 4, data.Nx, data.Ny, data.Nz,
                                         generator=ggen).to(dev)
                        dtg = 1.0 / cfg['val_gen_steps']
                        for si in range(cfg['val_gen_steps']):
                            tg = torch.full((v_gen_cap,), si / cfg['val_gen_steps'],
                                            device=dev)
                            xg = xg + model_ema(xg, tg, cv) * dtg
                        fg = xg * scale
                        rg = residual_rel_3d(fg, data.Ra[vb].to(dev),
                                             data.Pr[vb].to(dev), res)
                        val_gen = float(rg['combined'].mean())

                history['val'].append((it, val, val_res, val_gen))
                vr = f' val_res={val_res:.4e}' if val_res == val_res else ''
                vg = f' val_gen_res={val_gen:.4e}' if val_gen == val_gen else ''
                print(f'[train]   val(ema fm)={val:.4e} (best fm {best:.4e})'
                      f'{vr}{vg}  [{cnt} fixed fields]', flush=True)

                metric = val_gen if (cfg['early_stop_metric'] == 'gen_res'
                                     and val_gen == val_gen) else val
                improved = metric < best - 1e-6
                if improved:
                    best, bad = metric, 0
                    torch.save(dict(model=model.state_dict(), ema=ema.shadow,
                                    opt=opt.state_dict(), sched=sched.state_dict(),
                                    iter=it + 1, best=best, bad=bad, cfg=cfg,
                                    cond=cond, scale=data.scale, Nz=data.Nz,
                                    Ra_rng=data.Ra_rng, Pr_rng=data.Pr_rng,
                                    arch=ARCH_VERSION, physics=cfg['physics']),
                              ckpt_path)
                else:
                    bad += 1
                # "last": full resume state, overwritten every eval regardless
                torch.save(dict(model=model.state_dict(), ema=ema.shadow,
                                opt=opt.state_dict(), sched=sched.state_dict(),
                                iter=it + 1, best=best, bad=bad, cfg=cfg,
                                cond=cond, scale=data.scale, Nz=data.Nz,
                                Ra_rng=data.Ra_rng, Pr_rng=data.Pr_rng,
                                arch=ARCH_VERSION, history=history,
                                physics=cfg['physics']),
                          last_path)
                print(f'  [ckpt] it {it+1} metric({cfg["early_stop_metric"]})='
                      f'{metric:.4e} {"saved-best" if improved else f"(bad {bad}/{patience})"} '
                      f'[last saved]', flush=True)
                stop = patience > 0 and bad >= patience
            stop = ddp_broadcast_flag(stop, src=0, device=dev)
            if stop:
                if main:
                    print(f'[train] early stop at it={it} (best {best:.4e})')
                break

    if main:
        print(f'[train] done ({(time.perf_counter()-t0)/60:.1f} min), '
              f'best {best:.4e} -> {ckpt_path}')
    return ckpt_path


# ============================================================================
#  6.  PBFM sampling (plain ODE integration; physics already in the weights).
#      Wraps rb3d_pcfm_common.sample_fields (relaxer=None => no projection,
#      the headline PBFM path) and adds the physics-residual diagnostic that
#      sample_fields doesn't itself compute. Ablation paths keep the exact
#      same PCFM-style relaxer / a single non-iterative Leray cleanup, off by
#      default -- not the headline number.
# ============================================================================
@torch.no_grad()
def pbfm_sample_3d(model, ck, data, Ra_l, Pr_l, n_step=50, seed=0,
                   project=False, hard_cleanup=False, relaxer=None,
                   relax_steps=1500, relax_cfl=0.30, res=None):
    """Draw len(Ra_l) samples.
    DEFAULT (project=False, hard_cleanup=False): pure ODE integration, no
    projection/clean-up -- the true PBFM path, physics lives in the weights.
    Ablations (not the headline):
      hard_cleanup=True : one non-iterative Leray projection (incompressibility
                          + walls only, no physics-residual minimisation).
      project=True      : PCFM-style IMEX+Leray pseudo-time relaxation via
                          `RB3DRelaxer` (needs `relaxer=` or one is built here).
    """
    from rb3d_pcfm_common import sample_fields
    device = next(model.parameters()).device
    do_relax = project and relax_steps > 0
    if do_relax and relaxer is None:
        relaxer = RB3DRelaxer(data, device)
    fields, diag = sample_fields(model, ck, data, Ra_l, Pr_l,
                                 relaxer=relaxer if do_relax else None,
                                 n_step=n_step,
                                 relax_steps=relax_steps if do_relax else 0,
                                 relax_cfl=relax_cfl, seed=seed)
    fields = fields.to(device)
    u, v, w, Tp = fields[:, 0], fields[:, 1], fields[:, 2], fields[:, 3]
    if hard_cleanup and not do_relax:
        if res is None:
            res = SteadyResidual(data.Nx, data.Ny, data.Nz, data.aspect, device)
        u, v, w = leray_project(u.contiguous(), v.contiguous(), w.contiguous(),
                                res.VS, res.k2, res.gkx, res.gky)
        fields = torch.stack([u, v, w, Tp], dim=1)

    if res is None:
        res = SteadyResidual(data.Nx, data.Ny, data.Nz, data.aspect, device)
    Ra_t = torch.tensor(Ra_l, device=device, dtype=torch.float32)
    Pr_t = torch.tensor(Pr_l, device=device, dtype=torch.float32)
    r = residual_rel_3d(fields.to(device), Ra_t, Pr_t, res)
    diag['residual'] = r['combined'].cpu()
    diag['residual_components'] = dict(mom=r['mom'].cpu(), temp=r['temp'].cpu(),
                                       cont=r['cont'].cpu())
    return fields.cpu(), diag
