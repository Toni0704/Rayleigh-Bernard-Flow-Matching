#!/usr/bin/env python3
"""
generate_rb3d_multisolution.py
================================================================================
Reference-solution generator for 3-D steady Rayleigh-Benard convection with a
CURATED SET OF COEXISTING PLANFORM BRANCHES per (Ra, Pr) parameter pair, for
training a multimodal generative PDE surrogate (the 3D upgrade of
generate_rb2d_multisolution.py).

WHAT CHANGES FROM 2D (and why)
--------------------------------------------------------------------------------
The 2D solver used a SCALAR streamfunction psi and SCALAR vorticity omega, which
made incompressibility free (u = curl psi) and eliminated pressure. In 3D there
is no scalar streamfunction: vorticity is a 3-vector and the vorticity equation
gains vortex stretching. This generator therefore works in PRIMITIVE VARIABLES
(u, v, w, T') with a pressure / Leray projection to enforce div(u)=0:

    d_t u = -(u.grad)u - grad p + Pr lap(u) + Pr Ra T' zhat       (momentum)
    d_t T'= -(u.grad)T' + w      + lap(T')                        (temperature)
    div u = 0                                                     (continuity)

with T' = T-(1-z) (so T'=0 at both walls). The buoyant base part (1-z)zhat is a
pure gradient and is absorbed into pressure; only Pr Ra T' zhat drives motion.

SCHEME (IMEX fractional step, fully batched over configs)
--------------------------------------------------------------------------------
Per pseudo-time step, per velocity component a in {u,v,w} and for T':
  (1) explicit advection (+ buoyancy in w, + base-coupling w in T'),
  (2) IMPLICIT diffusion  (I - dt*Pr*lap) a** = a^n + dt*expl,
  (3) Leray projection    lap(phi) = div(u**),  u = u** - grad(phi).
Steps (1)-(2) are IDENTICAL in spirit to the 2D solver; the diffusion solves are
homogeneous-Dirichlet Helmholtz problems (a=0 at walls, T'=0 at walls), even
SIMPLER than 2D (no Thom wall vorticity, no T=1 wall-data term). Step (3) is the
3D analog of the 2D "recover psi from Poisson" hard-constraint step.

DOUBLE/TRIPLE-SPECTRAL BASIS
--------------------------------------------------------------------------------
FFT in x AND y (both periodic) + the eigenbasis of the z second-difference
(computed once). Each linear solve becomes a per-mode DIAGONAL scaling with
eigenvalue (lam_z - kx^2 - ky^2), whose coefficient can differ PER CONFIG -- so
every LHS sample carries its own Ra, Pr, dt in one batched GPU tensor, exactly
as in 2D. Velocities/T' use the DIRICHLET z-eigenbasis (interior Nz-2 nodes);
pressure uses the NEUMANN z-eigenbasis (all Nz nodes, dp/dz=0 at walls).

MODES (planforms) -- the 3D analog of the 2D roll set {2,3,4,5,6}
--------------------------------------------------------------------------------
In 2D a branch is one integer (horizontal wavenumber). In 3D a branch is a
PLANFORM: a horizontal wavevector pattern. The default curated set (target 5):
    ('roll', 2, 0)     rolls along x, wavelength Gamma/2
    ('roll', 3, 0)     rolls along x, wavelength Gamma/3
    ('square', 2, 2)   squares
    ('square', 3, 3)   squares
    ('rect', 3, 2)     cross-rolls / rectangle
All are classical Boussinesq RB steady planforms; roll-vs-square stability is
Pr-dependent (squares favoured at low Pr, rolls at high Pr), so the SAME (Ra,Pr)
box that 2D used is where the interesting mode competition lives. The pilot
verifies which planforms robustly lock across the box (gate 2 below) and prunes
the set if needed.

PILOT MODE (--pilot)  -- run this FIRST
--------------------------------------------------------------------------------
Generates a small bank on a cheap grid and prints three GATES that must pass
before committing to a full run:
  GATE 1  correctness   : (a) linear onset near Ra_c ~= 1708 for the preferred
                          wavelength, and (b) an x-roll seeded invariant in y
                          reproduces a pure-2D roll (residual, peak) so the 3D
                          solver is checked against the trusted 2D physics.
  GATE 2  locking       : each seeded planform locks to its seeded wavevector
                          across the sampled (Ra,Pr) box.
  GATE 3  distinctness  : the planforms are genuinely distinct fields.
It also reports the steady divergence (continuity residual) achieved by the
projection -- if that is not at truncation level, the projection needs the
staggered-z upgrade (documented; not needed if the gate passes).

PERFORMANCE (T4-class GPUs)
--------------------------------------------------------------------------------
Five compounding optimisations vs the first working version (~2300 s / 32
configs at 64x64x49 on a T4):
  0. RELATIVE convergence a converged fp32 field sits at the rounding floor
                         (~1e-5 relative); the old absolute |df|/dt vs a fixed
                         3e-6 tol could never be met, so already-steady configs
                         marched to the step cap. Judging steadiness by
                         relative field change makes them exit at ~3-4k steps
                         instead of 175k -- the single biggest win, and it also
                         lets stall detection retire truly unsteady configs.
  1. fp32 default        a T4 runs fp64 at ~1/32 of fp32 throughput; the fp32
                         fixed point was validated to satisfy the steady
                         equations far below the verification gate (see
                         --precision help). ~10-30x.
  2. fused spectral ops  one batched FFT round trip for all of (u,v,w) in the
                         Leray projection and ONE for all four Helmholtz
                         solves (was 14+ single-field transforms/step). ~1.4x.
  3. saturation seeding  seeds at the measured amplitude 0.30*sqrt(Ra)*Pr^0.25
                         so the exponential growth phase (most of the march)
                         is skipped, + adaptive dt tracking |u|max. ~2-3x.
  4. batch compaction    converged configs LEAVE the batch; straggler cost
                         scales with stragglers.
  5. dual-GPU sharding   --dual-gpu spawns one INDEPENDENT worker per GPU on
                         interleaved config shards (zero inter-GPU traffic =>
                         a true ~2x; a DataParallel-style split of a single
                         batch would only pay sync overhead for nothing) and
                         merges the shard banks at the end.
Combined: expect roughly 40-100x vs the unoptimised fp64 run; check the
first chunk's printed timing + the per-chunk ETA.

OUTPUT (loadable with weights_only=False)
--------------------------------------------------------------------------------
    bank['param_points'] = [(Ra,Pr), ...]
    bank['mode_list'] = [('roll',2,0), ...]; bank['aspect']=(Gx,Gy)
    bank['grid'] = {Nx,Ny,Nz,x,y,z}
    bank['entries'][(Ra,Pr,mode_id)] = {
        'grid_u','grid_v','grid_w','grid_T':(Nx,Ny,Nz),   # native-grid fields
        'planform':tuple,'peak_u':float,'residual':float,'div':float,
        'converged':bool}
mode_id is the planform tuple; grid_T is FULL temperature. Velocities are stored
directly (unlike 2D which stored psi/omega) because there is no scalar potential.

USAGE
    python generate_rb3d_multisolution.py --pilot --device cuda:0
    python generate_rb3d_multisolution.py --pilot --device cpu     # tiny smoke
    python generate_rb3d_multisolution.py --n-params 300            # full run
ALWAYS verify a full bank with verify_rb3d_multisolution.py (Phase 2).
"""

import argparse
import math
import os
import time

import numpy as np
import torch


# ============================================================================
#  Latin Hypercube Sampling over (Ra, Pr)   [verbatim from 2D]
# ============================================================================
def lhs_unit(n, d, rng):
    cut = np.linspace(0.0, 1.0, n + 1)
    lo, hi = cut[:n], cut[1:]
    pts = np.empty((n, d))
    for j in range(d):
        u = rng.random(n)
        col = lo + u * (hi - lo)
        rng.shuffle(col)
        pts[:, j] = col
    return pts


def sample_params(n, Ra_range, Pr_range, log, seed):
    rng = np.random.default_rng(seed)
    U = lhs_unit(n, 2, rng)
    if log:
        ra = np.exp(np.log(Ra_range[0]) + U[:, 0] * (np.log(Ra_range[1]) - np.log(Ra_range[0])))
        pr = np.exp(np.log(Pr_range[0]) + U[:, 1] * (np.log(Pr_range[1]) - np.log(Pr_range[0])))
    else:
        ra = Ra_range[0] + U[:, 0] * (Ra_range[1] - Ra_range[0])
        pr = Pr_range[0] + U[:, 1] * (Pr_range[1] - Pr_range[0])
    return [(float(r), float(p)) for r, p in zip(ra, pr)]


# ============================================================================
#  Finite differences.  Field layout: (B, Nx, Ny, Nz).
#  x = axis 1 (periodic), y = axis 2 (periodic), z = axis 3 (walls).
# ============================================================================
def fd_ddx(f, dx):
    return (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dx)


def fd_ddy(f, dy):
    return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2 * dy)


def dz_central(f, dz):
    out = torch.empty_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * dz)
    out[:, :, :, 0] = (-3 * f[:, :, :, 0] + 4 * f[:, :, :, 1] - f[:, :, :, 2]) / (2 * dz)
    out[:, :, :, -1] = (3 * f[:, :, :, -1] - 4 * f[:, :, :, -2] + f[:, :, :, -3]) / (2 * dz)
    return out


def advect(a, u, v, w, dx, dy, dz):
    """(u.grad) a with FD-x/y (roll) + FD-z, matching the solver's advection."""
    return u * fd_ddx(a, dx) + v * fd_ddy(a, dy) + w * dz_central(a, dz)


# ---- spectral horizontal derivatives (for the divergence-free projection) ----
def _kxy(Nx, Ny, dx, dy, device, dtype):
    kx = 2.0 * math.pi * torch.fft.fftfreq(Nx, d=dx, device=device, dtype=dtype)
    ky = 2.0 * math.pi * torch.fft.rfftfreq(Ny, d=dy, device=device, dtype=dtype)
    return kx, ky


def sp_ddx(f, kx):
    fh = torch.fft.rfft2(f, dim=(1, 2))
    return torch.fft.irfft2(1j * kx[None, :, None, None] * fh, s=f.shape[1:3], dim=(1, 2))


def sp_ddy(f, ky):
    fh = torch.fft.rfft2(f, dim=(1, 2))
    return torch.fft.irfft2(1j * ky[None, None, :, None] * fh, s=f.shape[1:3], dim=(1, 2))


# ============================================================================
#  Vertical SINE / COSINE Galerkin  (the robust no-slip projection basis)
#  ------------------------------------------------------------------------
#  No-slip velocities and T' vanish at both walls -> expand in a SINE basis
#  sin(m*pi*z) (Dirichlet). Pressure has dp/dz=0 at walls -> COSINE basis
#  cos(m*pi*z) (Neumann). The pair is EXACTLY consistent:
#     d/dz sin(m pi z) =  m pi cos(m pi z)     (velocity -> pressure space)
#     d/dz cos(m pi z) = -m pi sin(m pi z)     (pressure -> velocity space)
#     d2/dz2 (either)  = -(m pi)^2 (same fn)   (diagonal Laplacian)
#  so div(grad phi) reproduces the discrete divergence EXACTLY and the
#  eigenvalues -(m pi)^2 - k^2 are strictly negative away from the (0,0)
#  mode -- no odd-even (checkerboard) decoupling, no ill-conditioning. This
#  is the standard Fourier-Galerkin projection for a plane layer.
#
#  Implemented with small dense (Nz x Nz) synthesis matrices (Nz is tiny);
#  analysis is a pinv least-squares fit on the collocation nodes.
# ============================================================================
def build_vertical(Nz, device, dtype):
    z = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    m = torch.arange(Nz, device=device, dtype=dtype)
    ang = math.pi * torch.outer(z, m)                       # (Nz,Nz)
    Ssin = torch.sin(ang)                                   # grid <- sine coeff
    Ccos = torch.cos(ang)                                   # grid <- cosine coeff
    Ssin_ana = torch.linalg.pinv(Ssin)                      # coeff <- grid (sine)
    Ccos_ana = torch.linalg.pinv(Ccos)                      # coeff <- grid (cosine)
    mpi = math.pi * m
    mpi2 = mpi ** 2                                         # (Nz,) Laplace eigvals
    # d/dz cosine-grid-field -> sine-grid-field
    DZ_cos2sin = Ssin @ torch.diag(-mpi) @ Ccos_ana
    # d/dz sine-grid-field -> cosine-grid-field
    DZ_sin2cos = Ccos @ torch.diag(mpi) @ Ssin_ana
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    VS = dict(Ssin=Ssin, Ccos=Ccos, Ssin_ana=Ssin_ana, Ccos_ana=Ccos_ana,
              mpi2=mpi2, DZ_cos2sin=DZ_cos2sin, DZ_sin2cos=DZ_sin2cos, Nz=Nz)
    # complex copies (avoid a real->complex cast of these matrices EVERY step)
    for k in ('Ssin', 'Ccos', 'Ssin_ana', 'Ccos_ana', 'DZ_cos2sin', 'DZ_sin2cos'):
        VS[k + '_c'] = VS[k].to(cdtype)
    return VS


def _apply_z(M, f):
    """Apply an (Nz,Nz) vertical matrix along the last axis of f (B,Nx,Ny,Nz)."""
    return torch.einsum('ij,bxyj->bxyi', M, f)


# ============================================================================
#  Implicit diffusion (sine basis) and Leray projection (sine/cosine)
# ============================================================================
def helmholtz_sine(rhs, VS, k2, coef):
    """Solve (I - coef*lap) a = rhs with a=0 at walls, sine-diagonal in z.
    rhs:(B,Nx,Ny,Nz); coef:(B,). Batched: stack multiple fields along B to
    amortise the FFTs (imex_step solves u,v,w,T' in ONE call of batch 4B)."""
    B, Nx, Ny, Nz = rhs.shape
    rh = torch.fft.rfft2(rhs, dim=(1, 2))                   # (B,Nx,Ky,Nz)
    a = torch.einsum('mn,bkln->bklm', VS['Ssin_ana_c'], rh)
    lam = VS['mpi2'][None, None, None, :] + k2[None, :, :, None]   # (1,Nx,Ky,Nz)
    denom = 1.0 + coef.view(B, 1, 1, 1) * lam               # (I - coef*(-(mpi^2)-k^2))
    a = a / denom
    out = torch.einsum('jm,bklm->bklj', VS['Ssin_c'], a)
    out = torch.fft.irfft2(out, s=(Nx, Ny), dim=(1, 2))
    out[:, :, :, 0] = 0.0
    out[:, :, :, -1] = 0.0
    return out


def leray_project(u, v, w, VS, k2, kx, ky):
    """Project (u,v,w) onto the divergence-free subspace: solve lap(phi)=div(u)
    in the cosine (Neumann) basis, then u <- u - grad(phi). Consistent
    sine/cosine derivatives -> interior divergence removed to solver tolerance.
    Walls re-imposed (the O(sqrt dt) projection slip -> 0 as the march
    converges to the steady state).
    FUSED: one forward + one inverse batched FFT for all three components
    (vs 10 single-field transforms in the naive version)."""
    B, Nx, Ny, Nz = u.shape
    st = torch.stack([u, v, w], 1).reshape(3 * B, Nx, Ny, Nz)
    H = torch.fft.rfft2(st, dim=(1, 2)).reshape(B, 3, Nx, -1, Nz)
    uh, vh, wh = H[:, 0], H[:, 1], H[:, 2]
    ikx = (1j * kx)[None, :, None, None].to(H.dtype)
    iky = (1j * ky)[None, None, :, None].to(H.dtype)
    div_h = (ikx * uh + iky * vh
             + torch.einsum('ij,bxyj->bxyi', VS['DZ_sin2cos_c'], wh))
    dc = torch.einsum('mn,bxyn->bxym', VS['Ccos_ana_c'], div_h)   # cosine coeff
    lam = -(VS['mpi2'][None, None, :] + k2[:, :, None])           # (Nx,Ky,Nz)
    sing = lam.abs() < 1e-12                                # (kx=ky=0, m=0)
    lam = torch.where(sing, torch.ones_like(lam), lam)
    phic = dc / lam[None]
    phic = torch.where(sing[None], torch.zeros_like(phic), phic)
    phi_h = torch.einsum('jm,bxym->bxyj', VS['Ccos_c'], phic)
    uh = uh - ikx * phi_h
    vh = vh - iky * phi_h
    wh = wh - torch.einsum('ij,bxyj->bxyi', VS['DZ_cos2sin_c'], phi_h)
    out = torch.fft.irfft2(torch.stack([uh, vh, wh], 1).reshape(3 * B, Nx, -1, Nz),
                           s=(Nx, Ny), dim=(1, 2)).reshape(B, 3, Nx, Ny, Nz)
    u, v, w = out[:, 0], out[:, 1], out[:, 2]
    for f in (u, v, w):
        f[:, :, :, 0] = 0.0
        f[:, :, :, -1] = 0.0
    return u.contiguous(), v.contiguous(), w.contiguous()


def divergence_rms(u, v, w, VS, kx, ky):
    div = sp_ddx(u, kx) + sp_ddy(v, ky) + _apply_z(VS['DZ_sin2cos'], w)
    I = slice(1, -1)
    # scale by the FULL velocity-gradient magnitude (symmetric in x/y/z) so a
    # roll with no x-variation is not divided by ~0.
    gu = sp_ddx(u, kx)[:, :, :, I]
    gv = sp_ddy(v, ky)[:, :, :, I]
    gw = _apply_z(VS['DZ_sin2cos'], w)[:, :, :, I]
    scale = torch.stack([gu, gv, gw]).pow(2).sum(0).flatten(1).mean(1).sqrt().clamp_min(1e-12)
    return (div[:, :, :, I].pow(2).flatten(1).mean(1).sqrt() / scale)


# ============================================================================
#  Planform seeding
# ============================================================================
def planform_field(mode, Gx, Gy, X, Y):
    """Horizontal planform P(x,y). mode=(kind,n,m).
    Orientation modes (Option B): ('roll',n,0) = x-rolls (vary in x),
    ('roll',0,m) = y-rolls (vary in y). Squares/rects kept for generality but
    are UNSTABLE in Boussinesq RB (pilot-verified) -- not in the default set."""
    kind, n, m = mode
    kx = 2.0 * math.pi * n / Gx
    ky = 2.0 * math.pi * m / Gy
    if kind == 'roll':
        return torch.cos(kx * X) if m == 0 else torch.cos(ky * Y)
    if kind == 'square':
        return torch.cos(kx * X) + torch.cos(ky * Y)
    if kind == 'rect':
        return torch.cos(kx * X) + torch.cos(ky * Y)
    raise ValueError(mode)


def seed_state(modes, Gx, Gy, Nx, Ny, Nz, amplitude, device, dtype,
               Ra_l=None, Pr_l=None):
    """Seed w and T' with the target planform x sin(pi z); u=v=0 initially.
    If amplitude is None (auto) seed NEAR SATURATION: pilot measurements give
    peak w ~= 0.30*sqrt(Ra)*Pr^0.25 across the box, so seeding there (T'~0.25)
    skips the exponential growth phase (~6-7 e-folds of the pseudo-time march)
    and drops straight into the target branch's basin -- big speedup, and
    empirically MORE robust locking (deeper in the basin than a tiny seed)."""
    B = len(modes)
    x = torch.arange(Nx, device=device, dtype=dtype) * (Gx / Nx)
    y = torch.arange(Ny, device=device, dtype=dtype) * (Gy / Ny)
    z = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    sz = torch.sin(math.pi * z)
    u = torch.zeros(B, Nx, Ny, Nz, device=device, dtype=dtype)
    v = torch.zeros_like(u)
    w = torch.zeros_like(u)
    Tp = torch.zeros_like(u)
    for b, mode in enumerate(modes):
        P = planform_field(mode, Gx, Gy, X, Y)
        if amplitude is None:
            aw = 0.30 * math.sqrt(Ra_l[b]) * max(Pr_l[b], 0.5) ** 0.25
            aT = 0.25
        else:
            aw = aT = amplitude
        w[b] = aw * P[:, :, None] * sz[None, None, :]
        Tp[b] = aT * P[:, :, None] * sz[None, None, :]
    return u, v, w, Tp


# ============================================================================
#  One IMEX step (batched, per-config Ra,Pr,dt)
# ============================================================================
def imex_step(u, v, w, Tp, VS, kx, ky, k2, dx, dy, dz, dt, Pr, Ra):
    B = u.shape[0]
    dtv = dt.view(-1, 1, 1, 1)
    Prv = Pr.view(-1, 1, 1, 1)
    Rav = Ra.view(-1, 1, 1, 1)
    # explicit advection (+ buoyancy in w, + base-coupling in T')
    eu = -advect(u, u, v, w, dx, dy, dz)
    ev = -advect(v, u, v, w, dx, dy, dz)
    ew = -advect(w, u, v, w, dx, dy, dz) + Prv * Rav * Tp
    eT = -advect(Tp, u, v, w, dx, dy, dz) + w
    # implicit diffusion, ALL FOUR fields in one batched sine solve (4B)
    rhs = torch.stack([u + dtv * eu, v + dtv * ev, w + dtv * ew, Tp + dtv * eT],
                      dim=1).reshape(4 * B, *u.shape[1:])
    coef = torch.stack([dt * Pr, dt * Pr, dt * Pr, dt], dim=1).reshape(4 * B)
    out = helmholtz_sine(rhs, VS, k2, coef).reshape(B, 4, *u.shape[1:])
    return out[:, 0].contiguous(), out[:, 1].contiguous(), \
        out[:, 2].contiguous(), out[:, 3].contiguous()


# ============================================================================
#  Solve a chunk of configs to steady state
#  --------------------------------------------------------------------------
#  CONVERGENCE by RELATIVE field change (see _march_phase) -- the correct test
#  at fp32, where the absolute pseudo-time rate |df|/dt bottoms out at the
#  rounding floor (~1e-5 relative) and can NEVER reach a fixed absolute tol.
#  BATCH COMPACTION: converged/stalled configs LEAVE the batch, so straggler
#  cost scales with stragglers only. ADAPTIVE dt: dt_i = cfl*min_d /
#  max(1.5*|u|max_i, 0.30*Umax_est_i), updated each check (bigger dt while the
#  flow is slow). NaN => retry with a globally halved dt scale.
#  STALL DETECTION: a genuinely time-dependent config (oscillatory convection
#  at low Pr / high Ra) never meets rel_tol; once its relative change stops
#  improving it is parked with converged=False instead of burning the budget.
# ============================================================================
def _march_phase(u, v, w, Tp, Ra, Pr, VS, kx, ky, k2, dims, cfl, Umax_est,
                 rel_tol, stall_patience, check_every, max_steps, adapt_dt,
                 dt_scale, verbose, tag):
    """March the batch to steady state; per-config exit with compaction.

    CONVERGENCE is judged by RELATIVE field change over the check interval:
        rel = max|field^(k) - field^(k-1)| / max|field^(k)|
    (dimensionless). This is the correct steady-state test at ANY precision:
    a converged fp32 field sits at rel ~ 1e-5 (rounding floor), rel_tol=1e-4
    flags it with margin; a genuinely TIME-DEPENDENT config (oscillatory
    convection at low Pr / high Ra) keeps rel ~ 1e-2..1e0 and never passes.
    [The earlier absolute |df|/dt vs a fixed 3e-6 tol could NEVER be met in
    fp32 -- converged fields looked 'stuck' and ran to max_steps.]

    STALL DETECTION: if rel stops improving (no 30% drop) for stall_patience
    checks and is still above rel_tol, the config is declared non-steady and
    parked with converged=False -- so unsteady corners exit fast instead of
    grinding to the step budget. Returns full-batch (U,V,W,T, resid, conv, s)
    or None on NaN (-> caller halves dt and retries)."""
    dx, dy, dz = dims
    B = u.shape[0]
    device = u.device
    min_d = min(dx, dy, dz)

    idx = torch.arange(B, device=device)                     # local -> global
    RU, RV, RW, RT = u.clone(), v.clone(), w.clone(), Tp.clone()
    resid = torch.full((B,), float('inf'), device=device, dtype=u.dtype)
    conv = torch.zeros(B, dtype=torch.bool, device=device)   # steady?
    best = torch.full((B,), float('inf'), device=device, dtype=u.dtype)
    stall = torch.zeros(B, dtype=torch.long, device=device)

    def umax_of(u, v, w):
        return torch.stack([u, v, w]).abs().flatten(2).amax(2).amax(0)

    umax_now = umax_of(u, v, w).clamp_min(1.0)
    dt = dt_scale * cfl * min_d / torch.maximum(1.5 * umax_now, 0.30 * Umax_est)
    prev = torch.cat([u.flatten(1), v.flatten(1), w.flatten(1), Tp.flatten(1)], 1)
    t0 = time.perf_counter()
    s = 0
    while s < max_steps and idx.numel() > 0:
        n_inner = min(check_every, max_steps - s)
        for _ in range(n_inner):
            u, v, w, Tp = imex_step(u, v, w, Tp, VS, kx, ky, k2,
                                    dx, dy, dz, dt, Pr, Ra)
            u, v, w = leray_project(u, v, w, VS, k2, kx, ky)
        s += n_inner
        cur = torch.cat([u.flatten(1), v.flatten(1), w.flatten(1), Tp.flatten(1)], 1)
        if not torch.isfinite(cur).all():
            return None                                      # blow-up -> retry
        fmax = cur.abs().amax(dim=1).clamp_min(1e-6)
        rel = (cur - prev).abs().amax(dim=1) / fmax          # RELATIVE change
        umax_now = umax_of(u, v, w).clamp_min(1.0)
        resid[idx] = rel
        improved = rel < 0.7 * best
        best = torch.minimum(best, rel)
        stall = torch.where(improved, torch.zeros_like(stall), stall + 1)
        done = rel < rel_tol
        stalled = (stall >= stall_patience) & ~done
        park = done | stalled
        conv[idx[done]] = True
        if park.any():                                       # park + compact
            gp = idx[park]
            RU[gp] = u[park]; RV[gp] = v[park]
            RW[gp] = w[park]; RT[gp] = Tp[park]
            keep = ~park
            u, v, w, Tp = u[keep], v[keep], w[keep], Tp[keep]
            Ra, Pr, Umax_est = Ra[keep], Pr[keep], Umax_est[keep]
            umax_now, best, stall = umax_now[keep], best[keep], stall[keep]
            cur, idx = cur[keep], idx[keep]
        if adapt_dt:
            dt = dt_scale * cfl * min_d / torch.maximum(1.5 * umax_now,
                                                        0.30 * Umax_est)
        elif park.any():
            dt = dt[keep]
        prev = cur
        if verbose and s % (check_every * 8) == 0 and idx.numel() > 0:
            print(f'      [{tag}] step {s}/{max_steps} active={idx.numel()}/{B} '
                  f'max_rel={float(rel.max()):.1e} '
                  f'{time.perf_counter() - t0:.0f}s', flush=True)
    if idx.numel() > 0:                                      # budget exhausted
        RU[idx] = u; RV[idx] = v; RW[idx] = w; RT[idx] = Tp
    if verbose:
        print(f'    [{tag}] {s} steps, steady {int(conv.sum())}/{B}, '
              f'{time.perf_counter() - t0:.1f}s', flush=True)
    return RU, RV, RW, RT, resid, conv, s


def solve_chunk(Ra_l, Pr_l, modes, Gx, Gy, Nx, Ny, Nz, amplitude,
                cfl, umax_c, umax_floor, T_final, tol, check_every,
                device, spec32, spec64, mode='fp32',
                rel_tol=1e-4, polish_rel_tol=1e-6, stall_patience=8,
                polish_steps=6000, polish_cfl=0.7, max_march=30000,
                settle_steps=0, settle_div=16.0,
                verbose=True, max_retries=3):
    """mode: 'fp32' (default; converge to the fp32 relative floor), 'two-phase'
    (fp32 march + fp64 polish to polish_rel_tol), 'fp64' (full fp64 march).
    Convergence is by RELATIVE field change; genuinely time-dependent configs
    are caught by stall detection and returned with converged=False."""
    B = len(Ra_l)
    dx = Gx / Nx; dy = Gy / Ny; dz = 1.0 / (Nz - 1)
    dims = (dx, dy, dz)

    for attempt in range(max_retries):
        scale = 0.5 ** attempt
        # ---------------- phase 1: march (fp32 unless mode='fp64') ----------
        d1 = torch.float64 if mode == 'fp64' else torch.float32
        VS1, kx1, ky1, k21 = spec64 if mode == 'fp64' else spec32
        rt1 = polish_rel_tol if mode == 'fp64' else rel_tol
        Ra1 = torch.as_tensor(Ra_l, device=device, dtype=d1)
        Pr1 = torch.as_tensor(Pr_l, device=device, dtype=d1)
        Umax1 = torch.clamp(umax_c * Ra1.sqrt() * Pr1.clamp(min=1).pow(0.25),
                            min=umax_floor)
        u, v, w, Tp = seed_state(modes, Gx, Gy, Nx, Ny, Nz, amplitude,
                                 device, d1, Ra_l, Pr_l)
        u, v, w = leray_project(u, v, w, VS1, k21, kx1, ky1)
        if verbose:
            print(f'    [chunk B={B}] phase1={d1} adaptive-dt '
                  f'max_steps={max_march} rel_tol={rt1:g} '
                  f'(attempt {attempt + 1})', flush=True)
        out = _march_phase(u, v, w, Tp, Ra1, Pr1, VS1, kx1, ky1, k21, dims,
                           cfl, Umax1, rt1, stall_patience, check_every,
                           max_march, True, scale, verbose,
                           'fp32' if d1 == torch.float32 else 'fp64')
        if out is None:
            if verbose:
                print(f'    chunk: blow-up in phase 1 -> halving dt '
                      f'(retry {attempt + 1})')
            continue
        u, v, w, Tp, resid, conv, _ = out

        # ---------------- phase 2: fp64 polish (only steady configs) --------
        if mode == 'two-phase':
            VS2, kx2, ky2, k22 = spec64
            Ra2 = torch.as_tensor(Ra_l, device=device, dtype=torch.float64)
            Pr2 = torch.as_tensor(Pr_l, device=device, dtype=torch.float64)
            Umax2 = torch.clamp(umax_c * Ra2.sqrt() * Pr2.clamp(min=1).pow(0.25),
                                min=umax_floor)
            u, v, w, Tp = (f.double() for f in (u, v, w, Tp))
            u, v, w = leray_project(u, v, w, VS2, k22, kx2, ky2)
            out = _march_phase(u, v, w, Tp, Ra2, Pr2, VS2, kx2, ky2, k22, dims,
                               polish_cfl, Umax2, polish_rel_tol, stall_patience,
                               2 * check_every, polish_steps, True, scale,
                               verbose, 'fp64-polish')
            if out is None:
                if verbose:
                    print(f'    chunk: blow-up in polish -> halving dt '
                          f'(retry {attempt + 1})')
                continue
            u, v, w, Tp, resid, conv, _ = out
        break
    else:
        raise FloatingPointError('chunk failed to stabilise after retries')

    # optional SETTLE pass: a short march at dt/settle_div. The fractional-
    # step projection leaves an O(sqrt(Pr*dt)) wall-slip layer in the momentum
    # balance (measured: full-field momentum residual drops ~5x settling at
    # dt/16); this shrinks that layer physically before the state is stored.
    if settle_steps > 0:
        VSf, kxf, kyf, k2f = spec64 if u.dtype == torch.float64 else spec32
        Raf = torch.as_tensor(Ra_l, device=device, dtype=u.dtype)
        Prf = torch.as_tensor(Pr_l, device=device, dtype=u.dtype)
        umax = torch.stack([u, v, w]).abs().flatten(2).amax(2).amax(0).clamp_min(1.0)
        dts = (cfl * min(dims) / (1.5 * umax)) / settle_div
        t0s = time.perf_counter()
        for _ in range(settle_steps):
            u, v, w, Tp = imex_step(u, v, w, Tp, VSf, kxf, kyf, k2f,
                                    dims[0], dims[1], dims[2], dts, Prf, Raf)
            u, v, w = leray_project(u, v, w, VSf, k2f, kxf, kyf)
        if verbose:
            print(f'    [settle] {settle_steps} steps @ dt/{settle_div:g} '
                  f'({time.perf_counter() - t0s:.1f}s)', flush=True)

    # final clean projection + full temperature (in the final dtype)
    VSf, kxf, kyf, k2f = spec64 if u.dtype == torch.float64 else spec32
    u, v, w = leray_project(u, v, w, VSf, k2f, kxf, kyf)
    z = torch.linspace(0.0, 1.0, Nz, device=device, dtype=u.dtype)
    Tfull = Tp + (1.0 - z)[None, None, None, :]
    Tfull[:, :, :, 0] = 1.0; Tfull[:, :, :, -1] = 0.0
    div = divergence_rms(u, v, w, VSf, kxf, kyf)
    return dict(u=u, v=v, w=w, T=Tfull, Tp=Tp,
                residual=resid.cpu(), converged=conv.cpu(),
                div=div.cpu())


# ============================================================================
#  Planform classifier (3D analog of roll_mode)
# ============================================================================
def classify_planform(w, Gx, Gy, kmax=8, ratio=0.5):
    """Dominant horizontal wavevector(s) of w at mid-height -> planform tuple.
    w:(B,Nx,Ny,Nz). Returns list of tuples ('roll'/'square'/'rect', n, m)."""
    B, Nx, Ny, Nz = w.shape
    mid = Nz // 2
    wm = w[:, :, :, mid]
    spec = torch.fft.rfft2(wm, dim=(1, 2)).abs()             # (B,Nx,Ky)
    spec[:, 0, 0] = 0.0
    out = []
    for b in range(B):
        s = spec[b].clone()
        s[kmax + 1:, :] = 0.0                                # ignore very high modes
        s[:, kmax + 1:] = 0.0
        # top peak
        flat = s.reshape(-1)
        i1 = int(flat.argmax())
        n1, m1 = i1 // s.shape[1], i1 % s.shape[1]
        n1 = n1 if n1 <= Nx // 2 else n1 - Nx                # fold negative kx
        a1 = float(s[n1 % Nx, m1])
        # second peak (zero out a neighbourhood of the first)
        s2 = s.clone()
        for dn in range(-1, 2):
            for dm in range(-1, 2):
                s2[(n1 + dn) % Nx, min(max(m1 + dm, 0), s.shape[1] - 1)] = 0.0
        i2 = int(s2.reshape(-1).argmax())
        n2, m2 = i2 // s.shape[1], i2 % s.shape[1]
        n2 = n2 if n2 <= Nx // 2 else n2 - Nx
        a2 = float(s2[n2 % Nx, m2])
        N1, M1 = abs(n1), abs(m1)
        N2, M2 = abs(n2), abs(m2)
        if a2 < ratio * a1:                                 # single wavevector
            if M1 == 0:
                out.append(('roll', N1, 0))
            elif N1 == 0:
                out.append(('roll', 0, M1))                 # y-roll
            else:
                out.append(('rect', N1, M1))
        else:                                               # two wavevectors
            pair = sorted([(N1, M1), (N2, M2)])
            (pa, pb), (pc, pd) = pair
            # square: (n,0)+(0,n); rect: (n,0)+(0,m)
            if {pa, pb} == {0} or {pc, pd} == {0}:
                out.append(('rect', max(N1, N2, M1, M2), min(N1 + M1, N2 + M2)))
            elif (pb == 0 and pc == 0 and pa == pd):
                out.append(('square', pa, pa))
            elif (pb == 0 and pc == 0):
                out.append(('rect', pa, pd))
            else:
                out.append(('rect', max(N1, N2), max(M1, M2)))
    return out


def peak_u_of(sol):
    return torch.stack([sol['u'].abs().flatten(1).amax(1),
                        sol['v'].abs().flatten(1).amax(1),
                        sol['w'].abs().flatten(1).amax(1)]).amax(0)


def distinctness(entries_fields):
    """entries_fields: list of (4,Nx,Ny,Nz) tensors -> min pairwise rel-L2."""
    F = torch.stack([f.flatten() for f in entries_fields]).double()
    norms = F.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = torch.cdist(F, F)
    denom = norms.sqrt() @ norms.sqrt().t()
    rel = d / denom
    k = len(entries_fields)
    off = rel[~torch.eye(k, dtype=torch.bool)]
    return float(off.min()) if k > 1 else float('inf')


# ============================================================================
#  Generation driver
# ============================================================================
def mode_id(m):
    return (str(m[0]), int(m[1]), int(m[2]))


def generate(args):
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    Gx, Gy = args.aspect_x, args.aspect_y
    Nx, Ny, Nz = args.nx, args.ny, args.nz
    dx = Gx / Nx; dy = Gy / Ny; dz = 1.0 / (Nz - 1)

    params = sample_params(args.n_params, (args.ra_min, args.ra_max),
                           (args.pr_min, args.pr_max), not args.linear, args.seed)
    modes = [tuple(m) for m in args.modes]
    total = len(params) * len(modes)
    print(f'[gen] device={device} precision={args.precision} '
          f'grid={Nx}x{Ny}x{Nz} aspect=({Gx},{Gy})')
    print(f'[gen] LHS({"log" if not args.linear else "lin"}) {len(params)} (Ra,Pr) '
          f'x {len(modes)} planforms = {total} states')
    print(f'[gen] Ra in [{args.ra_min:.0f},{args.ra_max:.0f}] '
          f'Pr in [{args.pr_min:g},{args.pr_max:g}] modes={modes}')

    def spec_pack(dtype):
        VS = build_vertical(Nz, device, dtype)
        kx, ky = _kxy(Nx, Ny, dx, dy, device, dtype)
        k2 = kx[:, None] ** 2 + ky[None, :] ** 2           # (Nx,Ky)
        return (VS, kx, ky, k2)
    spec32 = spec_pack(torch.float32)
    spec64 = spec_pack(torch.float64)

    bank = {
        'param_points': params, 'mode_list': [mode_id(m) for m in modes],
        'aspect': (Gx, Gy), 'Ra_range': (args.ra_min, args.ra_max),
        'Pr_range': (args.pr_min, args.pr_max),
        'sampling': 'LHS-log' if not args.linear else 'LHS-linear', 'seed': args.seed,
        'grid': {'Nx': Nx, 'Ny': Ny, 'Nz': Nz,
                 'x': (torch.arange(Nx) * dx).to(torch.float32),
                 'y': (torch.arange(Ny) * dy).to(torch.float32),
                 'z': torch.linspace(0.0, 1.0, Nz).to(torch.float32)},
        'method': 'imex-primitive-leray-eigenbasis',
        'entries': {},
    }
    if args.resume and os.path.exists(args.out):
        old = torch.load(args.out, map_location='cpu', weights_only=False)
        bank['entries'] = old.get('entries', {})
        print(f'[gen] resuming: {len(bank["entries"])} cached entries')

    configs = []
    for (Ra, Pr) in params:
        for m in modes:
            configs.append((float(Ra), float(Pr), m))
    configs.sort(key=lambda c: c[0] * (c[1] ** 0.5))
    # SHARD FIRST, then drop cached entries. Order matters: if the cached
    # filter ran first, each worker would interleave a DIFFERENT list (its own
    # resume state differs), so configs[k::K] would select different configs
    # per worker -- duplicating some and silently DROPPING others from the
    # merged bank (measured: up to 320 of 1500 missing in stress tests).
    # Sharding the full, deterministic list keeps the partition identical
    # across workers and across restarts.
    if args.shard:
        k, K = (int(x) for x in args.shard.split(':'))
        configs = configs[k::K]                  # interleave -> balanced Ra mix
        print(f'[gen] shard {k}/{K}: {len(configs)} configs assigned')
    n_assigned = len(configs)
    configs = [c for c in configs
               if (c[0], c[1], mode_id(c[2])) not in bank['entries']]
    if n_assigned != len(configs):
        print(f'[gen] {n_assigned - len(configs)} already cached, '
              f'{len(configs)} remaining on this worker')
    total = len(configs) + len(bank['entries'])
    t_start = time.perf_counter()
    done = len(bank['entries'])

    for i in range(0, len(configs), args.chunk):
        ch = configs[i:i + args.chunk]
        Ra_l = [c[0] for c in ch]; Pr_l = [c[1] for c in ch]; mm = [c[2] for c in ch]
        sol = solve_chunk(Ra_l, Pr_l, mm, Gx, Gy, Nx, Ny, Nz, args.amplitude,
                          args.cfl, args.umax_c, args.umax_floor, args.t_final,
                          args.tol, args.check_every, device, spec32, spec64,
                          mode=args.precision, rel_tol=args.rel_tol,
                          polish_rel_tol=args.polish_rel_tol,
                          stall_patience=args.stall_patience,
                          polish_steps=args.polish_steps,
                          polish_cfl=args.polish_cfl, max_march=args.max_march,
                          settle_steps=args.settle_steps,
                          settle_div=args.settle_div)
        planforms = classify_planform(sol['w'], Gx, Gy)
        peaks = peak_u_of(sol).cpu()
        n_bad = 0
        for b, (Ra, Pr, m) in enumerate(ch):
            entry = {
                'grid_u': sol['u'][b].cpu().to(torch.float32),
                'grid_v': sol['v'][b].cpu().to(torch.float32),
                'grid_w': sol['w'][b].cpu().to(torch.float32),
                'grid_T': sol['T'][b].cpu().to(torch.float32),
                'planform': tuple(int(x) if not isinstance(x, str) else x
                                  for x in planforms[b]),
                'peak_u': float(peaks[b]), 'residual': float(sol['residual'][b]),
                'div': float(sol['div'][b]), 'converged': bool(sol['converged'][b]),
            }
            key = (float(Ra), float(Pr), mode_id(m))
            if key not in bank['entries']:
                done += 1
            bank['entries'][key] = entry
            if planforms[b] != mode_id(m) or not entry['converged']:
                n_bad += 1
        torch.save(bank, args.out)
        tail = 'all good' if n_bad == 0 else f'{n_bad} flagged'
        el = time.perf_counter() - t_start
        frac = (i + len(ch)) / max(len(configs), 1)
        eta = el / frac * (1 - frac)
        print(f'  [{min(done,total):4d}/{total}] chunk B={len(ch)} ({tail})  '
              f'elapsed {el/60:.0f}m, ETA {eta/60:.0f}m', flush=True)

    dtm = time.perf_counter() - t_start
    nconv = sum(1 for e in bank['entries'].values() if e['converged'])
    nlock = sum(1 for (Ra, Pr, mid), e in bank['entries'].items()
                if e['planform'] == mid)
    print(f'\n[gen] done: {len(bank["entries"])} states '
          f'({nconv} converged, {nlock} locked to seeded planform), {dtm:.1f}s '
          f'-> {args.out}')
    return bank


# ============================================================================
#  PILOT GATES
# ============================================================================
def run_pilot_gates(bank, args):
    print('\n' + '=' * 74)
    print('PILOT GATES')
    print('=' * 74)
    Gx, Gy = bank['aspect']
    # ---- GATE 2: locking ----------------------------------------------------
    modes = bank['mode_list']
    lock = {tuple(m): [0, 0] for m in modes}
    for (Ra, Pr, mid), e in bank['entries'].items():
        lock.setdefault(tuple(mid), [0, 0])[1] += 1
        if e['planform'] == tuple(mid):
            lock[tuple(mid)][0] += 1
    print('\n[GATE 2] planform locking (locked / total across the box):')
    all_lock = True
    for m in modes:
        ok, tot = lock[tuple(m)]
        frac = ok / max(tot, 1)
        flag = 'OK' if frac >= 0.8 else 'WEAK'
        if frac < 0.8:
            all_lock = False
        print(f'    {str(tuple(m)):>18}: {ok:3d}/{tot:3d}  ({100*frac:5.1f}%)  {flag}')
    # ---- GATE 3: distinctness ----------------------------------------------
    groups = {}
    for (Ra, Pr, mid), e in bank['entries'].items():
        groups.setdefault((Ra, Pr), []).append(
            torch.stack([e['grid_u'], e['grid_v'], e['grid_w'], e['grid_T']]))
    mind = min((distinctness(v) for v in groups.values() if len(v) > 1),
               default=float('inf'))
    print(f'\n[GATE 3] distinctness: min pairwise rel-L2 anywhere = {mind:.3f} '
          f'({"OK" if mind > 0.30 else "FAIL"}; threshold 0.30)')
    # ---- divergence report --------------------------------------------------
    divs = [e['div'] for e in bank['entries'].values()]
    print(f'\n[divergence] projection continuity residual: '
          f'median={np.median(divs):.2e} max={np.max(divs):.2e}  '
          f'({"OK, at truncation" if np.max(divs) < 5e-2 else "HIGH -> needs staggered-z upgrade"})')
    # ---- convergence --------------------------------------------------------
    nconv = sum(1 for e in bank['entries'].values() if e['converged'])
    print(f'\n[convergence] {nconv}/{len(bank["entries"])} states reached the '
          f'pseudo-time tolerance')
    print('\n[GATE 1] correctness (onset & 2D-embedding) is checked separately '
          'by --check-onset / --check-2d (run those next).')
    print('=' * 74)
    verdict = all_lock and mind > 0.30 and np.max(divs) < 5e-2
    print('PILOT GATES 2-3: ' + ('PASS' if verdict else 'REVIEW NEEDED'))
    print('=' * 74)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='./datasets/rb3d_multisolution/refs_bank.pt')
    p.add_argument('--device', default=None)
    p.add_argument('--precision', choices=['fp32', 'two-phase', 'fp64'],
                   default='fp32',
                   help="fp32 (default; ~10-30x faster on T4. Validated: the "
                        "fp32 fixed point's temperature-eq residual is ~7e-5 "
                        "relative, 14x below the verify gate, and the momentum "
                        "system is satisfied identically to fp64 -- the "
                        "residual there is a scheme-level O(dt)+wall-layer "
                        "artifact common to both precisions), "
                        "two-phase (fp32 march + fp64 polish to tol=3e-6), "
                        "fp64 (original full-precision march)")
    p.add_argument('--polish-steps', dest='polish_steps', type=int, default=8000,
                   help='fp64 polish step budget per chunk (two-phase mode)')
    p.add_argument('--polish-cfl', dest='polish_cfl', type=float, default=0.8,
                   help='CFL during the fp64 polish (near-steady => can be '
                        'more aggressive than the transient CFL)')
    p.add_argument('--shard', default=None, metavar='k:K',
                   help="process only configs[k::K] -- run one shard per GPU "
                        "(e.g. '0:2' and '1:2') for embarrassingly-parallel "
                        "multi-GPU generation; merge with --merge-shards")
    p.add_argument('--dual-gpu', dest='dual_gpu', action='store_true',
                   help='spawn 2 shard workers (cuda:0, cuda:1), wait, merge')
    p.add_argument('--merge-shards', dest='merge_shards', nargs='+', default=None,
                   help='merge shard banks into --out and exit')
    p.add_argument('--cleanup-shards', dest='cleanup_shards', action='store_true',
                   help='delete shard files after the merged bank is written '
                        'AND verified (needed when shards+merged exceed the '
                        'disk quota, e.g. ~20.5 GB at 128x64x49 on Kaggle)')
    p.add_argument('--n-params', dest='n_params', type=int, default=300)
    p.add_argument('--ra-min', dest='ra_min', type=float, default=5000.0)
    p.add_argument('--ra-max', dest='ra_max', type=float, default=30000.0)
    p.add_argument('--pr-min', dest='pr_min', type=float, default=0.5)
    p.add_argument('--pr-max', dest='pr_max', type=float, default=7.0)
    p.add_argument('--linear', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--aspect-x', dest='aspect_x', type=float, default=8.0)
    p.add_argument('--aspect-y', dest='aspect_y', type=float, default=4.0)
    # Grid defaults are 8 pts/unit: 64x32x25 at aspect (8,4).
    # This generates in ~18 min/chunk on a T4 and is sufficient for SciML.
    # HPC/high-fidelity target: --nx 128 --ny 64 --nz 49 (16/unit, ~170 min/chunk).
    # Mid-range: --nx 96 --ny 48 --nz 33 (12/unit, ~55 min/chunk).
    p.add_argument('--nx', type=int, default=64)
    p.add_argument('--ny', type=int, default=32)
    p.add_argument('--nz', type=int, default=25)
    p.add_argument('--amplitude', type=float, default=None,
                   help='seed amplitude; default None = auto near-saturation '
                        '(0.30*sqrt(Ra)*Pr^0.25), which skips the growth phase')
    p.add_argument('--t-final', dest='t_final', type=float, default=25.0)
    p.add_argument('--tol', type=float, default=3e-6)
    p.add_argument('--check-every', dest='check_every', type=int, default=250)
    p.add_argument('--rel-tol', dest='rel_tol', type=float, default=1e-4,
                   help='steady-state RELATIVE field-change threshold for '
                        'the fp32 march (fp32 floor ~1e-5, so 1e-4 is safe)')
    p.add_argument('--polish-rel-tol', dest='polish_rel_tol', type=float,
                   default=1e-6, help='relative threshold for the fp64 polish')
    p.add_argument('--stall-patience', dest='stall_patience', type=int,
                   default=8, help='checks with no 30%% improvement before a '
                        'config is declared time-dependent (converged=False)')
    p.add_argument('--max-march', dest='max_march', type=int, default=30000,
                   help='hard per-chunk step cap (steady rolls need <8k)')
    p.add_argument('--settle-steps', dest='settle_steps', type=int, default=0,
                   help='extra small-dt settle steps after convergence to '
                        'shrink the O(sqrt(Pr*dt)) wall-slip momentum layer '
                        '(recommended 2500 for the final production bank; '
                        'measured ~5x reduction of the full-field momentum '
                        'residual at dt/16)')
    p.add_argument('--settle-div', dest='settle_div', type=float, default=16.0,
                   help='settle dt divisor (layer ~ sqrt(dt) => /16 -> ~4x)')
    p.add_argument('--cfl', type=float, default=0.30)
    p.add_argument('--umax-c', dest='umax_c', type=float, default=0.5)
    p.add_argument('--umax-floor', dest='umax_floor', type=float, default=30.0)
    p.add_argument('--chunk', type=int, default=32)
    # curated planform set (kind:n:m)
    # Option B (pilot-validated): x-rolls {2,3,4} + y-rolls {2,3} in a
    # rectangular box -> genuinely-3D coexisting distinct planforms.
    p.add_argument('--modes', nargs='+', default=['roll:2:0', 'roll:3:0',
                                                  'roll:4:0', 'roll:0:2',
                                                  'roll:0:3'])
    p.add_argument('--no-resume', dest='resume', action='store_false')
    p.add_argument('--pilot', action='store_true',
                   help='small bank on a cheap grid + gate report')
    p.add_argument('--smoke', action='store_true',
                   help='tiny CPU smoke test (a few steps, 2 points)')
    args = p.parse_args()
    # parse modes "kind:n:m" -> (kind,int,int)
    parsed = []
    for tok in args.modes:
        k, n, m = tok.split(':')
        parsed.append((k, int(n), int(m)))
    args.modes = parsed
    if args.pilot:
        args.n_params = 24
        args.nx, args.ny, args.nz = 64, 32, 25
        args.polish_steps = 4000
        args.t_final = 25.0
        args.chunk = 20
        if args.out == './datasets/rb3d_multisolution/refs_bank.pt':
            args.out = './datasets/rb3d_multisolution/pilot_bank.pt'
    if args.smoke:
        args.n_params = 2
        args.ra_min, args.ra_max = 8000.0, 12000.0
        args.pr_min, args.pr_max = 1.0, 2.0
        args.nx, args.ny, args.nz = 48, 24, 17
        args.t_final = 3.0
        args.chunk = 10
        args.modes = [('roll', 2, 0), ('roll', 0, 2)]
        if args.out == './datasets/rb3d_multisolution/refs_bank.pt':
            args.out = './datasets/rb3d_multisolution/smoke_bank.pt'
    return args


def merge_shard_banks(paths, out, cleanup=False, expect=None):
    """Merge shard banks into `out`.

    DISK: the merged file is a FULL EXTRA COPY of the dataset (~9.6 GB at
    128x64x49). Writing it while the shards still exist needs
    shards + merged (~20.5 GB), which overflows a 20 GB Kaggle quota.

      cleanup=False : write merged, keep shards.   peak = shards + merged
      cleanup=True  : load shards into RAM, DELETE them, then write merged.
                      peak = max(shards, merged). The bank is fully in memory
                      before any file is removed, and free space is checked
                      first, so the delete is safe.

    Better still: skip the merge entirely -- `verify_rb3d_multisolution.py` and
    `prepare_rb3d_splits.py` both accept several bank files, so you can pass
    the shards directly and never pay for the duplicate.
    """
    import shutil
    bank = torch.load(paths[0], map_location='cpu', weights_only=False)
    for p in paths[1:]:
        b = torch.load(p, map_location='cpu', weights_only=False)
        bank['entries'].update(b['entries'])
        del b
    n = len(bank['entries'])
    if expect is None:
        try:
            expect = len(bank['param_points']) * len(bank['mode_list'])
        except (KeyError, TypeError):
            expect = None

    shard_bytes = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    outdir = os.path.dirname(os.path.abspath(out)) or '.'
    free = shutil.disk_usage(outdir).free
    need = int(1.05 * shard_bytes)                 # merged ~= sum of shards

    if cleanup:
        if free + shard_bytes < need:
            raise RuntimeError(
                f'not enough disk even after removing shards: '
                f'{(free + shard_bytes)/1e9:.2f} GB available, '
                f'{need/1e9:.2f} GB needed. Skip the merge and pass the shard '
                f'files directly to verify/prepare (both accept several banks).')
        # the whole bank is in RAM now -- removing the shards is safe
        for p in paths:
            os.remove(p)
            print(f'[merge] removed {p} (bank held in memory)')
        free = shutil.disk_usage(outdir).free
    elif free < need:
        raise RuntimeError(
            f'not enough disk to write the merged bank: {free/1e9:.2f} GB free, '
            f'{need/1e9:.2f} GB needed. Either rerun with --cleanup-shards '
            f'(deletes shards first) or skip merging and pass the shard files '
            f'directly to verify/prepare (both accept several banks).')

    torch.save(bank, out)
    chk = torch.load(out, map_location='cpu', weights_only=False)
    assert len(chk['entries']) == n, 'merged bank verification failed'
    del chk
    print(f'[merge] {n} entries from {len(paths)} shards -> {out}')
    if expect is not None and n != expect:
        print(f'[merge] WARNING: expected {expect} entries, got {n}. '
              f'{expect - n} configs are missing -- rerun generation (resume '
              f'will fill the gaps).')
    return bank


def launch_dual_gpu(args):
    """Spawn one worker per GPU with interleaved shards; wait; merge.
    Two INDEPENDENT processes -> zero inter-GPU traffic -> true ~2x (this
    workload is embarrassingly parallel across configs; a DataParallel-style
    split would only add sync overhead for nothing)."""
    import subprocess, sys
    n = torch.cuda.device_count()
    if n < 2:
        print(f'[dual-gpu] only {n} GPU(s) visible -- running single-GPU')
        return False
    base, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False; continue
        if a == '--dual-gpu':
            continue
        if a in ('--shard', '--device', '--out'):
            skip = True; continue
        base.append(a)
    shard_paths = [args.out + f'.shard{i}' for i in range(2)]
    procs = []
    for i in range(2):
        cmd = [sys.executable, os.path.abspath(__file__), *base,
               '--shard', f'{i}:2', '--device', f'cuda:{i}',
               '--out', shard_paths[i]]
        print(f'[dual-gpu] worker {i}: {" ".join(cmd)}', flush=True)
        procs.append(subprocess.Popen(cmd))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        raise RuntimeError(f'[dual-gpu] worker exit codes {rcs}')
    merge_shard_banks(shard_paths, args.out, cleanup=args.cleanup_shards)
    return True


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.merge_shards:
        merge_shard_banks(args.merge_shards, args.out,
                          cleanup=args.cleanup_shards)
        return
    if args.dual_gpu and launch_dual_gpu(args):
        bank = torch.load(args.out, map_location='cpu', weights_only=False)
        if args.pilot or args.smoke:
            run_pilot_gates(bank, args)
        return
    bank = generate(args)
    if args.pilot or args.smoke:
        run_pilot_gates(bank, args)


if __name__ == '__main__':
    main()