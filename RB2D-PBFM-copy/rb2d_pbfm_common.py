#!/usr/bin/env python3
"""
rb2d_pbfm_common.py
================================================================================
Shared library for PHYSICS-BASED FLOW MATCHING (PBFM) on 2-D steady
Rayleigh-Benard convection with five coexisting roll branches.

This is the PBFM counterpart of the RB2D PCFM pipeline. It is a *fusion* of

  (a) the RB2D PCFM infrastructure  -- data bank, FNO2d velocity field,
      spectral-x / FD-z differential operators, the IMEX eigen-basis, the
      relative combined steady residual used to score samples, symmetry-aware
      alignment, and per-branch threshold calibration; and

  (b) the Bratu-2D PBFM training recipe -- a PDE-residual loss evaluated on the
      *unrolled* clean-sample prediction x~1 (Algorithm 1 of Baldan et al.,
      "Physics vs Distributions: Pareto Optimal Flow Matching with Physics
      Constraints"), combined with the flow-matching loss in a CONFLICT-FREE
      manner via ConFIG, with t^p residual weighting and logit-normal t.

The crucial difference from PCFM is *when* physics is imposed:

    PCFM  : plain flow-matching training; physics enforced at SAMPLING time by
            an iterative IMEX projection (expensive inference).
    PBFM  : physics enforced at TRAINING time by the residual loss; sampling is
            a plain ODE integration with NO projection (inference speed == FM).

So the headline PBFM sampler here is projection-free. An optional light
hard-constraint clean-up (single differentiation-free Poisson solve to restore
incompressibility + walls) and an optional PCFM-style IMEX projection are also
provided purely for ablation / comparison; both are OFF by default.

------------------------------------------------------------------------------
Governing equations (steady Boussinesq, vorticity-streamfunction form)
------------------------------------------------------------------------------
    div u = 0                                     (continuity)
    u . grad T          =        lap T            (temperature transport)
    u . grad w          = Pr lap w + Pr Ra dx T   (vorticity transport)
with  w = dx u_z - dz u_x,  lap psi = -w,  u_x = dz psi,  u_z = -dx psi.
No-slip walls (u_x=u_z=0 at z=0,1), T(0)=1, T(1)=0, x periodic (period L=aspect).

Training channels: (u_x, u_z, T'), T' = T - (1-z). Scale-only normalisation
(divide each channel by its dataset std; no mean subtraction -> the x-translation
and x-reflection symmetries stay exact and are used as lossless augmentation).

All differential operators used in the residual match the SCORER
(heldout_branch_test.py): exact spectral d/dx (FFT) + 2nd-order FD d/dz. This
keeps the residual minimised in training identical (up to the intrinsic
discretisation floor) to the one used to accept samples.
================================================================================
"""

import math
import os
import glob
import time
import copy

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
#  0.  File finder (Kaggle / local aware)
# ============================================================================
def find_file(name, extra_dirs=None):
    """Locate a checkpoint/bank file by name across common locations."""
    if os.path.isabs(name) and os.path.exists(name):
        return name
    cands = []
    search = ['.', './rb2d', './rb2d/split', './split', './heldout',
              '/kaggle/working', '/kaggle/input']
    if extra_dirs:
        search = list(extra_dirs) + search
    for d in search:
        cands.append(os.path.join(d, name))
        cands += glob.glob(os.path.join(d, '**', name), recursive=True)
    for c in cands:
        if os.path.exists(c):
            return c
    if os.path.exists(name):
        return name
    return None


# ============================================================================
#  0b.  Multi-GPU (Kaggle 2 x T4) helpers
# ============================================================================
#  Design note -- why NO DistributedDataParallel wrapper:
#
#  ConFIG needs the FM gradient and the residual gradient SEPARATELY, i.e. two
#  backward passes per iteration. DDP's autograd hooks fire an all-reduce during
#  each backward and raise "reducer has already been marked ready" on the second
#  one unless it is wrapped in no_sync() -- at which point DDP is doing nothing
#  for us anyway. On top of that, the FNO spectral weights are torch.cfloat, and
#  complex tensors are awkward for NCCL collectives.
#
#  So instead: every rank holds an identical replica (same init seed), we flatten
#  each loss's gradients into ONE contiguous REAL vector (complex params are
#  viewed as real), and issue exactly ONE all_reduce per loss. That is 1-2 NCCL
#  calls per iteration on a single contiguous buffer -- strictly less traffic
#  than DDP's bucketed reduction, no complex-dtype issues, and replicas stay
#  bit-identical because the averaged gradients are identical everywhere.
#
#  The whole data bank lives on each GPU as one tensor (it is small: a few
#  hundred MB at most), so there is NO DataLoader, no workers, no host-to-device
#  copies in the training loop -- which is exactly the "allocation delay" failure
#  mode to avoid on Kaggle's 2 x T4.
# ============================================================================
def ddp_is_torchrun():
    return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ


def ddp_setup(rank, world_size, backend=None, port='29500'):
    """Initialise the process group and bind this process to its GPU."""
    if world_size <= 1:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    import torch.distributed as dist
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', port)
    if backend is None:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)          # bind BEFORE init_process_group
    if not dist.is_initialized():
        kw = {}
        if backend == 'nccl' and torch.cuda.is_available():
            # without device_id NCCL "guesses" the mapping and can hang
            try:
                kw['device_id'] = torch.device(f'cuda:{rank}')
            except Exception:
                pass
        try:
            dist.init_process_group(backend=backend, rank=rank,
                                    world_size=world_size, **kw)
        except TypeError:                    # older torch: no device_id kwarg
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
    """In-place average of a flat tensor across ranks."""
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        vec.div_(dist.get_world_size())
    return vec


def ddp_broadcast_flag(flag, src=0, device='cpu'):
    """Broadcast a python bool from src to all ranks (for early stopping)."""
    import torch.distributed as dist
    if not (dist.is_available() and dist.is_initialized()
            and dist.get_world_size() > 1):
        return flag
    t = torch.tensor([1 if flag else 0], device=device)
    dist.broadcast(t, src=src)
    return bool(t.item())


def shard(items, rank, world_size):
    """Round-robin split of a list across ranks (used by the evaluators)."""
    if world_size <= 1:
        return list(items)
    return [x for i, x in enumerate(items) if i % world_size == rank]


def launch(fn, gpus, *args):
    """Run fn(rank, world_size, *args) on `gpus` processes.
    Works under torchrun (uses its env) and standalone (spawns)."""
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
#  1.  Batched differential operators   (fields shape (B, Nx, Nz))
#      spectral d/dx (periodic), 2nd-order FD d/dz (walls one-sided)
# ============================================================================
def kx_vec(Nx, dx, device, dtype):
    return 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype)


def dX(f, kx):
    """d/dx via FFT along axis 1.  f:(B,Nx,Nz)."""
    F = torch.fft.fft(f, dim=1)
    return torch.fft.ifft(1j * kx[None, :, None] * F, dim=1).real


def d2X(f, kx):
    F = torch.fft.fft(f, dim=1)
    return torch.fft.ifft(-(kx[None, :, None] ** 2) * F, dim=1).real


def ddz(f, dz):
    """2nd-order central d/dz along axis 2, one-sided at walls."""
    o = torch.empty_like(f)
    o[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
    o[:, :, 0] = (-3 * f[:, :, 0] + 4 * f[:, :, 1] - f[:, :, 2]) / (2 * dz)
    o[:, :, -1] = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * dz)
    return o


def d2z(f, dz):
    """2nd-order d2/dz2, interior only (walls left 0 -- matches scorer)."""
    o = torch.zeros_like(f)
    o[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / dz ** 2
    return o


def rms(t):
    """RMS over the last two spatial dims -> per-sample scalar (B,)."""
    return t.pow(2).mean(dim=(1, 2)).clamp_min(1e-30).sqrt()


# ============================================================================
#  2.  Batched relative combined steady residual (differentiable)
#      Identical physics to heldout_branch_test.steady_residual, batched.
# ============================================================================
def steady_residual_rel(u, w, Tfull, Ra, Pr, dx, dz, return_fields=False):
    """
    u,w,Tfull : (B,Nx,Nz).  Ra,Pr : (B,) or scalar tensors.
    Returns dict of per-sample relative residuals (each (B,)):
        'cont','temp','vort','combined'(=max), 'sq'(=cont^2+temp^2+vort^2).
    All differentiable w.r.t. u,w,Tfull.
    """
    B, Nx, Nz = u.shape
    kx = kx_vec(Nx, dx, u.device, u.dtype)
    if not torch.is_tensor(Ra):
        Ra = torch.as_tensor(Ra, device=u.device, dtype=u.dtype)
    if not torch.is_tensor(Pr):
        Pr = torch.as_tensor(Pr, device=u.device, dtype=u.dtype)
    Ra = Ra.reshape(-1, 1, 1).to(u.dtype)
    Pr = Pr.reshape(-1, 1, 1).to(u.dtype)

    lap = lambda f: d2X(f, kx) + d2z(f, dz)
    W = dX(w, kx) - ddz(u, dz)                       # vorticity from velocity
    I = slice(1, -1)                                 # interior rows in z

    # continuity
    R_c = dX(u, kx) + ddz(w, dz)
    # temperature transport
    adv_T = u * dX(Tfull, kx) + w * ddz(Tfull, dz)
    R_T = adv_T - lap(Tfull)
    # vorticity transport
    adv_W = u * dX(W, kx) + w * ddz(W, dz)
    buoy = Pr * Ra * dX(Tfull, kx)
    R_W = adv_W - Pr * lap(W) - buoy

    def rel(num, den):
        return rms(num[:, :, I]) / rms(den[:, :, I]).clamp_min(1e-12)

    c = rel(R_c, dX(u, kx))
    t = rel(R_T, lap(Tfull))
    v = rel(R_W, buoy)
    combined = torch.stack([c, t, v], dim=0).max(dim=0).values
    out = dict(cont=c, temp=t, vort=v, combined=combined,
               sq=c ** 2 + t ** 2 + v ** 2)
    if return_fields:
        out['R_c'], out['R_T'], out['R_W'] = R_c, R_T, R_W
    return out


def wavenumber(w):
    """Dominant horizontal wavenumber (roll number) of u_z at mid-height.
    w:(Nx,Nz) single field -> int; or (B,Nx,Nz) -> (B,) long."""
    if w.dim() == 2:
        mid = w.shape[1] // 2
        spec = torch.fft.rfft(w[:, mid]).abs()
        spec[0] = 0.0
        return int(spec.argmax())
    mid = w.shape[2] // 2
    spec = torch.fft.rfft(w[:, :, mid], dim=1).abs()
    spec[:, 0] = 0.0
    return spec.argmax(dim=1)


# ============================================================================
#  3.  IMEX eigen-basis  (for hard-constraint Poisson solve + optional relax)
#      Identical construction to generate_rb2d_multisolution.build_eig.
# ============================================================================
def build_eig(Nx, Nz, dx, dz, device, dtype):
    M = Nz - 2
    idz2 = 1.0 / dz ** 2
    main = torch.full((M,), -2.0 * idz2, device=device, dtype=dtype)
    off = torch.full((M - 1,), idz2, device=device, dtype=dtype)
    Lz = torch.diag(main) + torch.diag(off, 1) + torch.diag(off, -1)
    lam, V = torch.linalg.eigh(Lz)
    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype)
    Lk = lam[None, :] - (kx ** 2)[:, None]          # (Nx,M)
    return dict(V=V, Lk=Lk, idz2=idz2, Nx=Nx, M=M, dx=dx, dz=dz,
                dpois=(1.0 / Lk)[None])


def _solve_diag(rhs_hat, V, diag):
    Vc = V.to(rhs_hat.dtype)
    t = torch.einsum('nm,bkn->bkm', Vc, rhs_hat)
    t = t * diag
    return torch.einsum('im,bkm->bki', Vc, t)


def poisson_psi(omega, E):
    """Solve lap(psi) = -omega, psi=0 at walls, via FFT-x + z-eigenbasis.
    omega:(B,Nx,Nz) -> psi:(B,Nx,Nz).  Differentiable."""
    V, dpois = E['V'], E['dpois']
    oh = torch.fft.fft(omega, dim=1)
    ph = torch.zeros_like(oh)
    ph[:, :, 1:-1] = _solve_diag(-oh[:, :, 1:-1], V, dpois)
    psi = torch.fft.ifft(ph, dim=1).real
    psi = psi.clone()
    psi[:, :, 0] = 0.0
    psi[:, :, -1] = 0.0
    return psi


def enforce_hard(u, w, Tp, E):
    """Hard-constraint enforcement (PCFM 5.2), differentiable:
      omega = dx w - dz u ; solve psi ; u=dz psi, w=-dx psi (walls 0);
      T = T' + (1-z), pinned at walls.  Returns (u,w,Tp) cleaned."""
    dx, dz = E['dx'], E['dz']
    Nx = u.shape[1]
    kx = kx_vec(Nx, dx, u.device, u.dtype)
    omega = dX(w, kx) - ddz(u, dz)
    psi = poisson_psi(omega, E)
    u2 = ddz(psi, dz)
    w2 = -dX(psi, kx)
    u2 = u2.clone(); w2 = w2.clone()
    u2[:, :, 0] = 0.0; u2[:, :, -1] = 0.0
    w2[:, :, 0] = 0.0; w2[:, :, -1] = 0.0
    # temperature perturbation pinned to 0 at both walls
    Tp2 = Tp.clone()
    Tp2[:, :, 0] = 0.0
    Tp2[:, :, -1] = 0.0
    return u2, w2, Tp2


# ---- optional PCFM-style IMEX relaxation projector (ablation only) ----------
def _eigen_step(omega, T, E, dt, Pr, Ra, dBw, dBT):
    dx, dz = E['dx'], E['dz']
    Nx, idz2, V = E['Nx'], E['idz2'], E['V']
    B = omega.shape[0]
    kx = kx_vec(Nx, dx, omega.device, omega.dtype)
    oh = torch.fft.fft(omega, dim=1)
    ph = torch.zeros_like(oh)
    ph[:, :, 1:-1] = _solve_diag(-oh[:, :, 1:-1], V, E['dpois'])
    psi = torch.fft.ifft(ph, dim=1).real
    psi[:, :, 0] = 0.0; psi[:, :, -1] = 0.0
    wlo = -2.0 * psi[:, :, 1] / dz ** 2
    whi = -2.0 * psi[:, :, -2] / dz ** 2
    omega = omega.clone(); omega[:, :, 0] = wlo; omega[:, :, -1] = whi

    def jac(a, b):
        return ddz(a, dz) * dX(b, kx) - dX(a, kx) * ddz(b, dz)

    rhs_o = (omega + dt * (Pr * Ra * dX(T, kx) - jac(psi, omega)))[:, :, 1:-1]
    rhs_T = (T + dt * (-jac(psi, T)))[:, :, 1:-1]
    roh = torch.fft.fft(rhs_o, dim=1)
    cw = (dt * Pr * idz2).view(B, 1)
    roh[:, :, 0] = roh[:, :, 0] + cw * torch.fft.fft(wlo, dim=1)
    roh[:, :, -1] = roh[:, :, -1] + cw * torch.fft.fft(whi, dim=1)
    omega_new = torch.fft.ifft(_solve_diag(roh, V, dBw), dim=1).real
    rth = torch.fft.fft(rhs_T, dim=1)
    rth[:, 0, 0] = rth[:, 0, 0] + (dt.flatten() * idz2 * Nx)
    T_new = torch.fft.ifft(_solve_diag(rth, V, dBT), dim=1).real
    oo = omega.clone(); oo[:, :, 1:-1] = omega_new
    To = T.clone(); To[:, :, 1:-1] = T_new
    To[:, :, 0] = 1.0; To[:, :, -1] = 0.0
    return oo, To


@torch.no_grad()
def imex_project(u, w, Tp, Ra, Pr, E, cfl=0.30, umax_c=0.5, umax_floor=30.0,
                 max_steps=2500, check_every=100, rel_tol=0.02, stall_patience=4):
    """PCFM-style best-residual IMEX relaxation. ABLATION ONLY (not used by the
    default PBFM sampler). Returns (u,w,Tp) best-residual snapshot."""
    dx, dz = E['dx'], E['dz']
    B, Nx, Nz = u.shape
    kx = kx_vec(Nx, dx, u.device, u.dtype)
    Ra_t = torch.as_tensor(Ra, device=u.device, dtype=u.dtype).reshape(B, 1, 1)
    Pr_t = torch.as_tensor(Pr, device=u.device, dtype=u.dtype).reshape(B, 1, 1)
    u, w, Tp = enforce_hard(u, w, Tp, E)
    omega = dX(w, kx) - ddz(u, dz)
    T = Tp + (1.0 - torch.linspace(0, 1, Nz, device=u.device, dtype=u.dtype))[None, None, :]
    Umax = torch.clamp(umax_c * Ra_t.flatten().sqrt() * Pr_t.flatten().clamp(min=1).pow(0.25),
                       min=umax_floor)
    dt = (cfl * min(dx, dz) / Umax).view(B, 1, 1)
    Lk = E['Lk'][None]
    dBw = 1.0 / (1.0 - (dt * Pr_t) * Lk)
    dBT = 1.0 / (1.0 - dt * Lk)
    best = torch.full((B,), float('inf'), device=u.device)
    best_u = u.clone(); best_w = w.clone(); best_Tp = Tp.clone()
    stall = 0
    zc = torch.linspace(0, 1, Nz, device=u.device, dtype=u.dtype)[None, None, :]
    for s in range(max_steps):
        omega, T = _eigen_step(omega, T, E, dt, Pr_t, Ra_t, dBw, dBT)
        if (s + 1) % check_every == 0 or s == max_steps - 1:
            psi = poisson_psi(omega, E)
            uu = ddz(psi, dz); ww = -dX(psi, kx)
            uu[:, :, 0] = 0; uu[:, :, -1] = 0; ww[:, :, 0] = 0; ww[:, :, -1] = 0
            r = steady_residual_rel(uu, ww, T, Ra_t.flatten(), Pr_t.flatten(),
                                    dx, dz)['combined']
            improved = r < best
            best = torch.where(improved, r, best)
            for b in range(B):
                if improved[b]:
                    best_u[b] = uu[b]; best_w[b] = ww[b]; best_Tp[b] = (T[b] - zc[0])
            if float(best.max()) < rel_tol:
                break
            stall = stall + 1 if not improved.any() else 0
            if stall >= stall_patience:
                break
    return best_u, best_w, best_Tp


# ============================================================================
#  4.  Data bank
# ============================================================================
class RB2DData:
    """Loads a bank .pt, keeps converged+locked entries, converts stored
    (grid_psi, grid_T) -> training channels (u_x, u_z, T') on the native grid,
    computes scale-only normalisation, exposes unique (Ra,Pr) points."""

    def __init__(self, bank_path, device='cpu', dtype=torch.float32,
                 require_locked=True):
        bank = torch.load(find_file(bank_path) or bank_path,
                          map_location='cpu', weights_only=False)
        self.bank = bank
        g = bank['grid']
        self.Nx = int(g['Nx']); self.Nz = int(g['Nz'])
        self.x = g['x'].to(dtype); self.z = g['z'].to(dtype)
        self.dx = float(self.x[1] - self.x[0])
        self.dz = float(self.z[1] - self.z[0])
        self.aspect = float(bank.get('aspect', 8.0))
        self.n_list = sorted(bank.get('n_list', [2, 3, 4, 5, 6]))
        self.Ra_range = tuple(bank.get('Ra_range', (5000.0, 30000.0)))
        self.Pr_range = tuple(bank.get('Pr_range', (0.5, 7.0)))
        self.device = device
        self.dtype = dtype

        fields, params, rolls = [], [], []
        kx = kx_vec(self.Nx, self.dx, 'cpu', torch.float64)
        for (Ra, Pr, n), e in bank['entries'].items():
            if 'grid_psi' not in e or 'grid_T' not in e:
                continue
            if require_locked and (e.get('roll_mode') != n or not e.get('converged')):
                continue
            psi = e['grid_psi'].to(torch.float64)
            T = e['grid_T'].to(torch.float64)
            u = _ddz1(psi, self.dz); u[:, 0] = 0; u[:, -1] = 0
            w = -_dX1(psi, kx); w[:, 0] = 0; w[:, -1] = 0
            Tp = T - (1.0 - self.z.to(torch.float64))[None, :]
            f = torch.stack([u, w, Tp], dim=0).to(dtype)     # (3,Nx,Nz)
            fields.append(f)
            params.append((float(Ra), float(Pr)))
            rolls.append(int(n))
        if not fields:
            raise RuntimeError(f'no usable entries in {bank_path}')
        self.fields = torch.stack(fields, dim=0)             # (N,3,Nx,Nz)
        self.params = params
        self.rolls = rolls
        self.N = self.fields.shape[0]

        # scale-only normalisation (per channel std, no mean subtraction)
        std = self.fields.reshape(self.N, 3, -1).std(dim=(0, 2)).clamp_min(1e-8)
        self.sigma = std                                     # (3,)
        self.peak = self.fields.abs().reshape(self.N, 3, -1).amax(dim=(0, 2))

    # -- normalisation helpers --
    def norm(self, f):
        return f / self.sigma.to(f.device).view(1, 3, 1, 1)

    def denorm(self, f):
        return f * self.sigma.to(f.device).view(1, 3, 1, 1)

    # -- (Ra,Pr) log-normalisation for conditioning --
    def lognorm_Ra(self, Ra):
        a, b = math.log(self.Ra_range[0]), math.log(self.Ra_range[1])
        return (math.log(Ra) - a) / (b - a)

    def lognorm_Pr(self, Pr):
        a, b = math.log(self.Pr_range[0]), math.log(self.Pr_range[1])
        return (math.log(Pr) - a) / (b - a)

    def unique_points(self):
        seen = {}
        for i, (Ra, Pr) in enumerate(self.params):
            seen.setdefault((Ra, Pr), []).append(i)
        return seen

    def refs_at(self, Ra, Pr, tol=1e-3):
        """Return {n: normalized (3,Nx,Nz)} references at nearest (Ra,Pr)."""
        best = None; bd = 1e18
        for (RRa, RPr) in {(a, b) for (a, b) in self.params}:
            d = abs(RRa - Ra) + abs(RPr - Pr)
            if d < bd:
                bd = d; best = (RRa, RPr)
        out = {}
        for i, (RRa, RPr) in enumerate(self.params):
            if best is not None and abs(RRa - best[0]) < tol and abs(RPr - best[1]) < tol:
                out[self.rolls[i]] = self.fields[i]
        return best, out


def _ddz1(f, dz):
    o = torch.empty_like(f)
    o[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dz)
    o[:, 0] = (-3 * f[:, 0] + 4 * f[:, 1] - f[:, 2]) / (2 * dz)
    o[:, -1] = (3 * f[:, -1] - 4 * f[:, -2] + f[:, -3]) / (2 * dz)
    return o


def _dX1(f, kx):
    F = torch.fft.fft(f, dim=0)
    return torch.fft.ifft(1j * kx[:, None] * F, dim=0).real


# ============================================================================
#  5.  FNO2d velocity field  v_theta(x_t, t [, Ra,Pr])
# ============================================================================
# Bump this whenever the network definition changes. Checkpoints carry it so a
# stale file is detected by NAME instead of by a confusing key-mismatch crash.
ARCH_VERSION = 'pcfm-parity-v1'


class SpectralConv2d(nn.Module):
    """VERBATIM from rb2d_pcfm_common.py -- architectural parity with PCFM is
    mandatory: the paper compares vanilla / PBFM / PCFM, and per the eval spec
    vanilla and PCFM share one checkpoint. Any change here (parameter layout,
    FFT normalisation, mode truncation) would silently make the rows
    incomparable and would stop PCFM checkpoints from loading."""

    def __init__(self, ci, co, m1, m2):
        super().__init__()
        self.m1, self.m2, self.co = m1, m2, co
        sc = 1.0 / (ci * co)
        self.w1 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, 2))
        self.w2 = nn.Parameter(sc * torch.randn(ci, co, m1, m2, 2))

    @staticmethod
    def _cmul(a, b):
        return torch.einsum('bixy,ioxy->boxy', a, b)

    def forward(self, x):
        B, C, H, W = x.shape
        xf = torch.fft.rfft2(x, norm='ortho')
        m1 = min(self.m1, xf.shape[-2] // 2)
        m2 = min(self.m2, xf.shape[-1])
        w1 = torch.view_as_complex(self.w1)
        w2 = torch.view_as_complex(self.w2)
        out = torch.zeros(B, self.co, xf.shape[-2], xf.shape[-1],
                          dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = self._cmul(xf[:, :, :m1, :m2], w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = self._cmul(xf[:, :, -m1:, :m2], w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FlowFNO2d(nn.Module):
    """VERBATIM from rb2d_pcfm_common.py. Flow-matching velocity field
    v(x_t, t [, logRa, logPr]).

    NOTE ON THE TIME EMBEDDING. The frequencies DECAY (1 -> 1/2000) and are then
    scaled by 2000, so the maximum phase is 2000 rad. An earlier PBFM-only
    version of this file used ASCENDING frequencies (1 -> 1000) with the same
    2000x scale, giving a maximum phase of 2e6 rad; that aliased every channel
    and decorrelated adjacent sampler steps. Do not "improve" this function --
    it must stay bit-identical to the PCFM definition.
    """

    def __init__(self, hidden=64, modes_x=20, modes_z=12, n_layers=4,
                 time_emb=32, in_ch=3, cond=False):
        super().__init__()
        self.te = time_emb
        self.cond = cond
        lift_in = in_ch + 2 + time_emb + (2 if cond else 0)
        self.lift = nn.Conv2d(lift_in, hidden, 1)
        self.sp = nn.ModuleList([SpectralConv2d(hidden, hidden, modes_x, modes_z)
                                 for _ in range(n_layers)])
        self.w = nn.ModuleList([nn.Conv2d(hidden, hidden, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv2d(hidden, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, in_ch, 1))

    def _temb(self, t, H, W):
        half = self.te // 2
        fr = torch.exp(-math.log(2000) * torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None] * fr[None, :] * 2000
        e = torch.cat([a.sin(), a.cos()], -1)
        return e[:, :, None, None].expand(-1, -1, H, W)

    def forward(self, xt, t, cond_vec=None):
        B, C, H, W = xt.shape
        if t.dim() == 0 or t.numel() == 1:
            t = t.expand(B) if t.dim() else t.repeat(B)
        gx = torch.linspace(0, 1, H, device=xt.device).view(1, 1, H, 1).expand(B, 1, H, W)
        gz = torch.linspace(0, 1, W, device=xt.device).view(1, 1, 1, W).expand(B, 1, H, W)
        feats = [xt, gx, gz, self._temb(t.to(xt.dtype), H, W)]
        if self.cond:
            assert cond_vec is not None, 'conditioned model requires cond_vec (B,2)'
            feats.append(cond_vec.view(B, 2, 1, 1).expand(-1, -1, H, W).to(xt.dtype))
        h = self.lift(torch.cat(feats, dim=1))
        for sp_, w_ in zip(self.sp, self.w):
            h = torch.nn.functional.gelu(sp_(h) + w_(h))
        return self.proj(h)


# ============================================================================
#  6.  ConFIG conflict-free gradient combination (Liu et al. 2025a)
# ============================================================================
def _flat_grads(params):
    """Flatten all parameter grads into one REAL vector. Complex params (the
    FNO spectral weights) are split into (real, imag) so ConFIG runs in a real
    inner-product space."""
    gs = []
    for p in params:
        g = torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        if torch.is_complex(g):
            g = torch.view_as_real(g)          # (..., 2)
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
    """gv = U[U(O(g_fm,g_r)) + U(O(g_r,g_fm))]; g = (g_fm.gv + g_r.gv) gv.
       O(a,b) = b - (a.b/|a|^2) a  (component of b orthogonal to a)."""
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
#  7.  Augmentation (exact symmetries in normalized coords)
# ============================================================================
def augment(batch):
    """batch:(B,3,Nx,Nz) normalized. Random circular x-shift + p=0.5 reflection
    (flip x, negate u_x). Both exact -> still a valid solution in norm coords."""
    B, C, Nx, Nz = batch.shape
    out = batch
    shifts = torch.randint(0, Nx, (B,), device=batch.device)
    idx = (torch.arange(Nx, device=batch.device)[None, :] - shifts[:, None]) % Nx
    out = torch.gather(out, 2, idx[:, None, :, None].expand(B, C, Nx, Nz))
    flip = torch.rand(B, device=batch.device) < 0.5
    if flip.any():
        f = out[flip].flip(2)                       # flip x
        f[:, 0] = -f[:, 0]                           # negate u_x
        out = out.clone()
        out[flip] = f
    return out


# ============================================================================
#  8.  PBFM training  (Algorithm 1: unroll + residual loss + ConFIG)
# ============================================================================
def sample_logit_normal_t(B, device):
    """t = sigmoid(N(0,1)) -> logit-normal on (0,1), concentrated near 0.5."""
    return torch.sigmoid(torch.randn(B, device=device))


def train_pbfm(data, cond=True, iters=40000, batch=32, lr=3e-4,
               hidden=64, modes_x=20, modes_z=12, n_layers=4, time_emb=32,
               unroll=2, resid_p=1.0, backprop='last', use_config=True,
               resid_weight=1.0, ema_decay=0.999, grad_clip=1.0,
               val_frac=0.10, val_every=500, patience=8, warmup_iters=1000,
               device='cpu', ckpt_out='ckpt_rb2d_pbfm.pt', resume=True,
               log_every=200, seed=0, dtype=torch.float32,
               rank=0, world_size=1, return_model=False,
               val_resid_samples=32, physics=True, val_gen_samples=8,
               val_gen_steps=25, adam_b1=0.5, adam_b2=0.999,
               lr_schedule='constant', unroll_start=None, curriculum_end=None):
    """
    PBFM training loop (multi-GPU aware).

      * flow-matching loss  L_FM = || v(x_t,t) - (x1 - eps) ||^2
      * residual loss       L_R  = mean( (t_start^p)^2 * combined_residual^2 )
        where combined_residual is evaluated on the UNROLLED prediction x~1
        (denormalized, at the true (Ra,Pr) of each field).
      * the two are combined CONFLICT-FREE via ConFIG (use_config=True) or by a
        fixed weight resid_weight (use_config=False).
      * physics is deferred: no residual loss during the first `warmup_iters`
        (the FM prior needs to form before the residual gradient is meaningful).

    backprop: 'last' backprops the residual only through the final unroll step
              (memory-light, PBFM default); 'all' through every step.

    Multi-GPU: `batch` is the GLOBAL batch (paper App. G keeps it constant
    regardless of GPU count). Each rank takes batch//world_size samples, draws
    DIFFERENT data (per-rank RNG offset), and the flattened per-loss gradients
    are averaged with one all_reduce each -- so the update is exactly the
    single-GPU global-batch update and the LR needs no rescaling.
    """
    # identical init on every rank
    torch.manual_seed(seed)
    dev = device
    main = is_main(rank)

    # PBFM paper, Appendix G: "To avoid learning rate adjustments across
    # different hardware setups, we maintain a constant GLOBAL batch size,
    # independent of the number of GPUs." `batch` is therefore the GLOBAL batch;
    # each rank processes batch//world_size samples, so 1-GPU and 2-GPU runs are
    # the same optimisation problem and need no LR retuning.
    global_batch = batch
    if batch % world_size != 0 and main:
        print(f'[train] NOTE: global batch {batch} not divisible by '
              f'{world_size} GPUs; rounding down.')
    batch = max(1, batch // world_size)
    sigma = data.sigma.to(dev).view(1, 3, 1, 1)
    fields = data.fields.to(dev)
    N = data.N
    Ra_all = torch.tensor([p[0] for p in data.params], device=dev, dtype=dtype)
    Pr_all = torch.tensor([p[1] for p in data.params], device=dev, dtype=dtype)
    cond_all = None
    if cond:
        cond_all = torch.tensor(
            [[data.lognorm_Ra(p[0]), data.lognorm_Pr(p[1])] for p in data.params],
            device=dev, dtype=dtype)

    # train / val split
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_val = max(1, int(val_frac * N))
    val_idx = perm[:n_val].to(dev)
    tr_idx = perm[n_val:].to(dev)

    # ---- FIXED validation problem -------------------------------------------
    # The validation number drives best-checkpoint selection and early stopping,
    # so it must be a deterministic function of the weights. Drawing fresh noise
    # / fresh t / a fresh subset at every evaluation (as a naive implementation
    # does) makes it a high-variance random estimate: successive values are not
    # comparable, "improvements" are partly luck, and the saved best checkpoint
    # can be a noise fluke. So we freeze the val fields, the noise eps and the
    # times t ONCE, and reuse them for every evaluation.
    v_sel = val_idx                                     # all held-out fields
    gcpu = torch.Generator().manual_seed(seed + 12345)
    v_eps = torch.randn(v_sel.numel(), 3, data.Nx, data.Nz,
                        generator=gcpu).to(dev, dtype)
    v_t = torch.sigmoid(torch.randn(v_sel.numel(), generator=gcpu)).to(dev, dtype)
    v_res_cap = min(val_resid_samples, v_sel.numel())

    model = FlowFNO2d(hidden, modes_x, modes_z, n_layers, time_emb,
                      in_ch=3, cond=cond).to(dev)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    # PBFM paper, Appendix G: "AdamW optimizer with weight decay set to 0,
    # beta1 = 0.5, beta2 = 0.999, and a FIXED learning rate."
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0,
                            betas=(adam_b1, adam_b2))
    if lr_schedule == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    else:                                        # 'constant' = paper default
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    params = list(model.parameters())

    start_it = 0
    best_val = float('inf')
    bad = 0
    history = {'it': [], 'fm': [], 'res': [], 'val': []}
    resume_path = os.path.splitext(ckpt_out)[0] + '_resume.pt'
    if resume and os.path.exists(resume_path):
        # A resume file written by a DIFFERENT network definition cannot be
        # loaded. Rather than crashing mid-launch (and taking the other rank
        # down with it), archive it and start fresh.
        try:
            st = torch.load(resume_path, map_location=dev, weights_only=False)
            if st.get('arch') != ARCH_VERSION:
                raise RuntimeError(
                    f"architecture {st.get('arch', 'pre-parity')} != {ARCH_VERSION}")
            model.load_state_dict(st['model']); ema.load_state_dict(st['ema'])
            opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
            start_it = st['it']; best_val = st['best_val']; history = st['history']
            if main:
                print(f'[train] resumed from {resume_path} at iter {start_it}')
        except Exception as e:
            if main:
                stale = resume_path + '.stale'
                try:
                    os.replace(resume_path, stale)
                except Exception:
                    stale = '(could not rename)'
                print(f'[train] *** cannot resume from {resume_path}: {e}')
                print(f'[train] *** that checkpoint predates the PCFM-parity '
                      f'network (sp/w/proj). Archived to {stale}; '
                      f'STARTING FROM SCRATCH.')
            # NO barrier here: the ranks race on os.path.exists above, so one may
            # skip this block entirely and never reach a barrier placed inside it,
            # which would deadlock the other rank. Both independently fall back to
            # start_it=0, which needs no synchronisation.
            start_it = 0
            best_val = float('inf')
            history = {'it': [], 'fm': [], 'res': [], 'val': []}

    # per-rank RNG offset: replicas are identical, but each rank must draw a
    # DIFFERENT slice of the data, otherwise the second GPU just duplicates work.
    torch.manual_seed(seed + 1000 * rank + 7 * start_it)

    def norm_field(idx):
        return (fields[idx] / sigma)

    def unroll_predict(model, x_t, t, cond_b, n_steps):
        """Algorithm-1 unroll: evolve x_t at time t to t=1 in n_steps."""
        dt = (1.0 - t).clamp_min(1e-4) / n_steps
        et = t.clone()
        v = model(x_t, et, cond_b)                       # first velocity (FM target uses this t)
        x1 = x_t + dt[:, None, None, None] * v
        for i in range(1, n_steps):
            et = et + dt
            if backprop == 'last' and i < n_steps - 1:
                with torch.no_grad():
                    vv = model(x1, et, cond_b)
            else:
                vv = model(x1, et, cond_b)
            x1 = x1 + dt[:, None, None, None] * vv
        return v, x1

    zc = (1.0 - data.z.to(dev))[None, None, :]            # (1,1,Nz) base T profile

    def residual_loss(x1_norm, ra, pr, t_start):
        # denormalize -> physical channels
        f = x1_norm * sigma[0].view(1, 3, 1, 1) if x1_norm.dim() == 4 else x1_norm
        u = f[:, 0]; w = f[:, 1]; Tp = f[:, 2]
        Tfull = Tp + zc
        r = steady_residual_rel(u, w, Tfull, ra, pr, data.dx, data.dz)
        wt = (t_start.clamp_min(1e-3) ** resid_p)
        return (wt ** 2 * r['sq']).mean(), float(r['combined'].mean().detach())

    model.train()
    n_bad_res = 0
    t0 = time.perf_counter()
    for it in range(start_it, iters):
        bidx = tr_idx[torch.randint(0, tr_idx.numel(), (batch,), device=dev)]
        x1 = augment(norm_field(bidx))                   # (B,3,Nx,Nz)
        cond_b = cond_all[bidx] if cond else None
        ra = Ra_all[bidx]; pr = Pr_all[bidx]
        eps = torch.randn_like(x1)
        t = sample_logit_normal_t(batch, dev)
        x_t = (1.0 - t)[:, None, None, None] * eps + t[:, None, None, None] * x1
        target = x1 - eps

        # unrolling curriculum: ramp n from unroll_start to unroll between the
        # end of warmup and curriculum_end (paper S3.3).
        if unroll_start is None or unroll_start >= unroll:
            n_unroll = unroll
        else:
            c_end = curriculum_end if curriculum_end else int(0.5 * iters)
            frac = (it - warmup_iters) / max(1, c_end - warmup_iters)
            frac = min(1.0, max(0.0, frac))
            n_unroll = int(round(unroll_start + frac * (unroll - unroll_start)))
        do_res = physics and (it >= warmup_iters)
        if main and physics and do_res and it == warmup_iters:
            print(f'[train] --- residual loss ON at it={it} '
                  f'(unroll {unroll_start if unroll_start else unroll}->{unroll}, p={resid_p}, '
                  f'{"ConFIG" if use_config else f"fixed w={resid_weight}"}) ---',
                  flush=True)

        opt.zero_grad(set_to_none=True)
        v = model(x_t, t, cond_b)
        loss_fm = ((v - target) ** 2).mean()

        if do_res:
            # unroll for residual (recompute v inside if backprop=='all')
            _, x1p = unroll_predict(model, x_t, t, cond_b, n_unroll)
            loss_res, res_val = residual_loss(x1p, ra, pr, t)
            if not torch.isfinite(loss_res):
                # a diverged unroll would push inf/NaN through ConFIG and destroy
                # the weights; skip the physics term for this step instead
                n_bad_res += 1
                if main and n_bad_res in (1, 10, 100, 1000):
                    print(f'[train] non-finite residual at it={it} '
                          f'(count={n_bad_res}); skipping the physics term for '
                          f'this step', flush=True)
                loss_fm.backward()
                if world_size > 1:
                    gg = _flat_grads(params)
                    ddp_allreduce_mean_(gg)
                    _unflat_to_grads(gg, params)
                res_val = float('nan')
            elif use_config:
                loss_fm.backward(retain_graph=True)
                g_fm = _flat_grads(params)
                opt.zero_grad(set_to_none=True)
                loss_res.backward()
                g_r = _flat_grads(params)
                # ---- ONE all_reduce per loss: ConFIG then acts on the exact
                #      global-batch gradients, not on per-GPU ones ----
                if world_size > 1:
                    ddp_allreduce_mean_(g_fm)
                    ddp_allreduce_mean_(g_r)
                g = config_direction(g_fm, g_r)
                _unflat_to_grads(g, params)
            else:
                (loss_fm + resid_weight * loss_res).backward()
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

        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        sched.step()

        # EMA with warmup (bias correction). A fixed decay of 0.999 keeps
        # 0.999^500 ~= 61% of the RANDOM INIT after 500 steps, which makes the
        # EMA -- and therefore the reported validation loss -- look far worse
        # than the live model early in training. Ramping the decay in makes the
        # EMA track the model from the start and converge to `ema_decay` later.
        d = min(ema_decay, (1.0 + it) / (10.0 + it))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm, alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        if (it % log_every == 0 or it == iters - 1) and main:
            history['it'].append(it)
            history['fm'].append(float(loss_fm.detach()))
            history['res'].append(res_val)
            spd = (it - start_it + 1) / max(1e-9, time.perf_counter() - t0)
            gb = global_batch
            res_s = (f'{res_val:.4e}' if res_val == res_val
                     else ('off(FM baseline)' if not physics
                           else f'warmup({max(0, warmup_iters - it)} left)'))
            print(f'[train] it={it:6d} fm={float(loss_fm.detach()):.4e} '
                  f'res={res_s} lr={sched.get_last_lr()[0]:.2e} '
                  f'{spd:.1f} it/s (global batch {gb})', flush=True)

        # validation (EMA FM loss on held-out fields) -- rank 0 only, then the
        # stop decision is broadcast so every rank leaves the loop together.
        if (it + 1) % val_every == 0 or it == iters - 1:
            stop = False
            if main:
                ema.eval()
                with torch.no_grad():
                    # deterministic FM loss over ALL held-out fields, using the
                    # frozen (eps, t) -- comparable across evaluations
                    tot, cnt = 0.0, 0
                    for s in range(0, v_sel.numel(), batch):
                        sl = slice(s, min(s + batch, v_sel.numel()))
                        vb = v_sel[sl]
                        xv = norm_field(vb)
                        cv = cond_all[vb] if cond else None
                        ev = v_eps[sl]; tv = v_t[sl]
                        xtv = ((1 - tv)[:, None, None, None] * ev
                               + tv[:, None, None, None] * xv)
                        pv = ema(xtv, tv, cv)
                        tot += float(((pv - (xv - ev)) ** 2).mean()) * xv.shape[0]
                        cnt += xv.shape[0]
                    val = tot / max(cnt, 1)

                    # deterministic PHYSICS residual of the EMA's unrolled x~1
                    val_res = float('nan')
                    if do_res and v_res_cap > 0:
                        vb = v_sel[:v_res_cap]
                        xv = norm_field(vb)
                        cv = cond_all[vb] if cond else None
                        ev = v_eps[:v_res_cap]; tv = v_t[:v_res_cap]
                        xtv = ((1 - tv)[:, None, None, None] * ev
                               + tv[:, None, None, None] * xv)
                        _, x1v = unroll_predict(ema, xtv, tv, cv, unroll)
                        fv = x1v * sigma
                        rv = steady_residual_rel(fv[:, 0], fv[:, 1],
                                                 fv[:, 2] + zc,
                                                 Ra_all[vb], Pr_all[vb],
                                                 data.dx, data.dz)
                        val_res = float(rv['combined'].mean())

                    # GENERATION residual: integrate the ODE from PURE NOISE and
                    # measure the residual of the resulting samples. val_res
                    # above is optimistic because x_t is built from the true x1,
                    # so it carries ground-truth information; this one measures
                    # what the sampler will actually produce at evaluation time.
                    val_gen = float('nan')
                    if val_gen_samples > 0:
                        gb = min(val_gen_samples, v_sel.numel())
                        vb = v_sel[:gb]
                        cv = cond_all[vb] if cond else None
                        ggen = torch.Generator().manual_seed(seed + 999)
                        xg = torch.randn(gb, 3, data.Nx, data.Nz,
                                         generator=ggen).to(dev, dtype)
                        dtg = 1.0 / val_gen_steps
                        for si in range(val_gen_steps):
                            tg = torch.full((gb,), si / val_gen_steps,
                                            device=dev, dtype=dtype)
                            xg = xg + ema(xg, tg, cv) * dtg
                        fg = xg * sigma
                        rg = steady_residual_rel(fg[:, 0], fg[:, 1],
                                                 fg[:, 2] + zc,
                                                 Ra_all[vb], Pr_all[vb],
                                                 data.dx, data.dz)
                        val_gen = float(rg['combined'].mean())
                ema.train()
                history['val'].append((it, val, val_res, val_gen))
                vr = f' val_res={val_res:.4e}' if val_res == val_res else ''
                vg = f' val_gen_res={val_gen:.4e}' if val_gen == val_gen else ''
                print(f'[train]   val(ema fm)={val:.4e} (best {best_val:.4e})'
                      f'{vr}{vg}  [fixed {cnt} fields]', flush=True)
                improved = val < best_val - 1e-6
                if improved:
                    best_val = val; bad = 0
                    torch.save({'state': ema.state_dict(),
                                'cfg': dict(hidden=hidden, modes_x=modes_x,
                                            modes_z=modes_z, n_layers=n_layers,
                                            time_emb=time_emb, cond=cond),
                                'sigma': data.sigma.cpu(),
                                'Ra_range': data.Ra_range, 'Pr_range': data.Pr_range,
                                'best_val': best_val, 'it': it}, ckpt_out)
                else:
                    bad += 1
                torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                            'opt': opt.state_dict(), 'sched': sched.state_dict(),
                            'it': it + 1, 'best_val': best_val,
                            'history': history, 'arch': ARCH_VERSION},
                           resume_path)
                stop = bad >= patience
            stop = ddp_broadcast_flag(stop, src=0, device=dev)
            if stop:
                if main:
                    print(f'[train] early stop at it={it} (best val {best_val:.4e})')
                break

    if main:
        print(f'[train] done. best val {best_val:.4e}. saved EMA -> {ckpt_out}')
    return (ckpt_out, ema) if return_model else ckpt_out


# ============================================================================
#  9.  Load a trained model
# ============================================================================
def load_model(ckpt_path, device='cpu'):
    """Load either a PBFM checkpoint (this pipeline) or a PCFM/vanilla
    checkpoint produced by rb2d_pcfm_common.py. Both use the identical
    FlowFNO2d, so the PCFM plain-FM checkpoint IS the `vanilla` baseline of the
    eval spec and needs no retraining -- it just has a different wrapper:

        PCFM : {'state', 'cfg', 'cond', 'norm': {'std','Ra_rng','Pr_rng'}, 'info'}
        PBFM : {'state', 'cfg'(+cond,physics), 'sigma', 'Ra_range', 'Pr_range'}
    """
    ck = torch.load(find_file(ckpt_path) or ckpt_path, map_location=device,
                    weights_only=False)
    cfg = dict(ck['cfg'])
    cond = cfg.get('cond', ck.get('cond', False))
    m = FlowFNO2d(hidden=cfg['hidden'], modes_x=cfg['modes_x'],
                  modes_z=cfg['modes_z'], n_layers=cfg['n_layers'],
                  time_emb=cfg['time_emb'], in_ch=3, cond=cond).to(device)
    missing, unexpected = m.load_state_dict(ck['state'], strict=False)
    n_total = len(m.state_dict())
    if len(missing) > 0.2 * max(n_total, 1):
        raise RuntimeError(
            f'{ckpt_path} does not match the current network '
            f'({len(missing)}/{n_total} weights missing).\n'
            f'  missing e.g. {list(missing)[:3]}\n'
            f'  found   e.g. {list(unexpected)[:3]}\n'
            f'This checkpoint predates the PCFM-parity architecture '
            f'(sp/w/proj). Delete it and retrain -- loading it with strict=False '
            f'would leave the network RANDOMLY INITIALISED and silently produce '
            f'meaningless numbers.')
    if missing or unexpected:
        print(f'[load_model] {ckpt_path}: missing={list(missing)[:4]} '
              f'unexpected={list(unexpected)[:4]}')
    m.eval()

    # normalise the two wrapper formats into one
    if 'norm' in ck:                                   # PCFM-style
        nm = ck['norm']
        out = dict(cfg=dict(cfg, cond=cond),
                   sigma=nm['std'].clone().detach(),
                   Ra_range=tuple(nm['Ra_rng']), Pr_range=tuple(nm['Pr_rng']),
                   source='pcfm')
    else:                                              # PBFM-style
        out = dict(cfg=dict(cfg, cond=cond),
                   sigma=ck['sigma'], Ra_range=tuple(ck['Ra_range']),
                   Pr_range=tuple(ck['Pr_range']),
                   source='pbfm', physics=cfg.get('physics', True))
    out['cfg']['cond'] = cond
    return m, out


# ============================================================================
#  10.  PBFM sampling  (plain ODE integration; physics already in the weights)
# ============================================================================
@torch.no_grad()
def pbfm_sample(model, ck, data, Ra, Pr, K=40, n_step=50, seed=0,
                project=False, hard_cleanup=False, device='cpu', E=None):
    """
    Draw K samples at (Ra,Pr).

    DEFAULT = the true PBFM path: pure Euler ODE integration, NO projection and
    NO clean-up. Physics lives in the weights, so this is the number that should
    be reported as PBFM. Inference cost equals plain flow matching.

    Ablations (not the headline):
      hard_cleanup=True : one differentiation-free Poisson solve restoring
                          incompressibility + walls (cheap, non-iterative)
      project=True      : PCFM-style iterative IMEX relaxation (expensive)
    """
    torch.manual_seed(seed)
    dev = device
    sigma = ck['sigma'].to(dev).view(1, 3, 1, 1)
    cond = ck['cfg']['cond']
    cond_b = None
    if cond:
        a = math.log(ck['Ra_range'][0]); b = math.log(ck['Ra_range'][1])
        c = math.log(ck['Pr_range'][0]); d = math.log(ck['Pr_range'][1])
        cr = (math.log(Ra) - a) / (b - a); pr_ = (math.log(Pr) - c) / (d - c)
        cond_b = torch.tensor([[cr, pr_]], device=dev).expand(K, 2)

    Nx, Nz = data.Nx, data.Nz
    t_start = time.perf_counter()
    x = torch.randn(K, 3, Nx, Nz, device=dev)
    dt = 1.0 / n_step
    for i in range(n_step):
        tv = torch.full((K,), i / n_step, device=dev)
        x = x + model(x, tv, cond_b) * dt
    f = x * sigma                                     # denormalize -> physical

    u = f[:, 0]; w = f[:, 1]; Tp = f[:, 2]
    if E is None:
        E = build_eig(Nx, Nz, data.dx, data.dz, dev, torch.float64)

    if project:
        u, w, Tp = imex_project(u.double(), w.double(), Tp.double(),
                                [Ra] * K, [Pr] * K, E)
        u, w, Tp = u.float(), w.float(), Tp.float()
    elif hard_cleanup:
        u2, w2, Tp2 = enforce_hard(u.double(), w.double(), Tp.double(), E)
        u, w, Tp = u2.float(), w2.float(), Tp2.float()

    # Diagnostic residual in FLOAT32 -- this matches rb2d_pcfm_common.py, which
    # evaluates rho on the float32 sample fields. It agrees with float64 to
    # ~1e-7 relative, and it matters for the reported timing: a T4 runs FP64 at
    # 1/32 of FP32 (complex128 FFTs are far worse), so forcing .double() here
    # made `sec_per_sample` ~60x larger than the sampling actually costs and
    # would have inverted the PBFM-vs-PCFM inference-speed comparison.
    zc = (1.0 - data.z.to(dev))[None, :]
    Tfull = Tp + zc[None]
    t_ode = time.perf_counter() - t_start
    r = steady_residual_rel(u.float(), w.float(), Tfull.float(),
                            [Ra] * K, [Pr] * K, data.dx, data.dz)['combined']
    roll = wavenumber(w)
    fields_phys = torch.stack([u, w, Tp], dim=1)
    secs = time.perf_counter() - t_start
    return fields_phys, dict(roll=roll.cpu(), residual=r.float().cpu(),
                             seconds=secs, sec_per_sample=secs / max(K, 1),
                             seconds_ode=t_ode, sec_per_sample_ode=t_ode / max(K, 1))


# ============================================================================
#  11.  Threshold calibration (per-branch, from the bank)
# ============================================================================
def calibrate_thresholds(data, mult=1.5, q=0.99, ra_min=0.0, ra_max=float('inf')):
    """Per-branch acceptance threshold = mult * quantile_q(reference residuals).
       Matches heldout_branch_test.mode_calibrate."""
    by_n = {}
    for i in range(data.N):
        Ra, Pr = data.params[i]
        if Ra < ra_min or Ra > ra_max:
            continue
        n = data.rolls[i]
        f = data.fields[i:i + 1].double()
        u = f[:, 0]; w = f[:, 1]; Tp = f[:, 2]
        Tfull = Tp + (1.0 - data.z.double())[None, None, :]
        r = steady_residual_rel(u, w, Tfull, [Ra], [Pr], data.dx, data.dz)['combined']
        by_n.setdefault(n, []).append(float(r))
    thr, floor = {}, {}
    for n, v in by_n.items():
        t = torch.tensor(v)
        thr[n] = mult * float(t.quantile(q))
        floor[n] = float(t.median())
    return thr, floor


# ============================================================================
#  12.  Symmetry-aware alignment + ground-truth errors
# ============================================================================
def align_to(g, r):
    """Align generated g:(3,Nx,Nz) to reference r:(3,Nx,Nz) over the RB symmetry
    group: all circular x-shifts and the reflection (flip x, negate u_x). Returns
    the aligned copy of g with minimum MSE to r. Vectorized brute-force over the
    Nx shifts (O(Nx^2 Nz), Nx<=128 -> negligible)."""
    Nx = g.shape[1]
    gf = g.flip(1).clone(); gf[0] = -gf[0]
    # index[s, x] = (x + s) % Nx  -> shifted[s][x] = gg[(x+s)%Nx]
    idx = (torch.arange(Nx)[None, :] + torch.arange(Nx)[:, None]) % Nx
    best = None; bmse = float('inf')
    for gg in (g, gf):
        sh = gg[:, idx, :].permute(1, 0, 2, 3)       # (S,3,Nx,Nz)
        mse = ((sh - r[None]) ** 2).mean(dim=(1, 2, 3))
        s = int(mse.argmin())
        if float(mse[s]) < bmse:
            bmse = float(mse[s]); best = sh[s]
    return best


def data_errors(g_aligned, r):
    mse = float(((g_aligned - r) ** 2).mean())
    nrmse = float(((g_aligned - r).pow(2).sum().sqrt() /
                   r.pow(2).sum().sqrt().clamp_min(1e-12)))
    return mse, nrmse


def physics_L2(u, w, Tfull, Ra, Pr, dx, dz):
    """Buoyancy-scaled RMS physics L2 (dimensionless). u,w,Tfull:(Nx,Nz)."""
    r = steady_residual_rel(u[None], w[None], Tfull[None], [Ra], [Pr], dx, dz)
    return float((r['sq'][0] / 3.0).sqrt())


# ============================================================================
#  13.  Roll entropy
# ============================================================================
def roll_entropy(rolls, branches=(2, 3, 4, 5, 6)):
    """Branch diversity entropy in NATS (spec A.7: use ln, not log2 -- the paper
    reports nats and the maximum must equal ln(N)). Pass only the VALID samples."""
    import collections
    c = collections.Counter(int(x) for x in rolls)
    tot = sum(c.get(b, 0) for b in branches)
    if tot == 0:
        return 0.0
    ent = 0.0
    for b in branches:
        p = c.get(b, 0) / tot
        if p > 0:
            ent -= p * math.log(p)          # NATS
    return ent