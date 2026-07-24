#!/usr/bin/env python3
"""
rb3d_pcfm_sampler.py
================================================================================
FAITHFUL PCFM sampling for steady 3D Rayleigh-Benard, following
"Physics-Constrained Flow Matching" (Utkarsh, Cai, et al., NeurIPS 2025),
Algorithm 1 / Sec 3.3.

WHAT THIS REPLACES
--------------------------------------------------------------------------------
The earlier `RB3DRelaxer` ran the *generator's IMEX time-marching solver* for
600-1500 steps on each sample. That is NOT PCFM. It is a pseudo-time solver:
  * it moves a sample arbitrarily far, to the nearest ATTRACTOR;
  * it can only ever land on STABLE fixed points (unstable ones repel it);
  * it dominated runtime (the paper reports projection = 1-3% of sampling).

PCFM instead applies a *minimal* Gauss-Newton correction that root-finds the
constraint residual h(u)=0. It moves the sample as little as possible, and --
crucially -- Newton does not care about dynamical stability, so it can land on
UNSTABLE steady branches too. That property is what makes the "did the model
find a solution outside the training set?" question answerable.

ALGORITHM (per flow substep tau -> tau', mirrors Algorithm 1)
--------------------------------------------------------------------------------
  1. shoot:    u1 = ODESolve(u, v_theta, tau, 1)          [cheap Euler]
  2. project:  u_proj = u1 - J^T (J J^T)^{-1} h(u1)       [1 Gauss-Newton step]
  3. reverse:  u_hat  = u_proj + (tau'-1) * (u_proj - u0) [OT interpolant, Eq 5]
               (NOT a reverse ODE solve -- the paper avoids that for stability)
  4. u <- u_hat
  final:       one more projection at tau=1                [Eq 7]
Per the paper's Appendix H we use ONE Newton update per step and lambda=0 (no
relaxed-penalty guidance) by default.

THE CONSTRAINT h(u)  (the "middle ground" design)
--------------------------------------------------------------------------------
Splitting the constraints by their mathematical character:

  LINEAR, enforced EXACTLY, outside Newton:
      div(u) = 0   and   no-slip / Dirichlet walls
    These are handled by the exact Leray projector `leray_project`, which is an
    orthogonal projection onto the divergence-free, wall-satisfying subspace.
    Putting them into the Newton system would waste iterations on constraints we
    can satisfy in closed form. Every Newton iterate is re-projected, so the
    iteration lives entirely on that subspace.

  NONLINEAR, enforced by Gauss-Newton:
      h(u) = [ P_div (steady momentum residual) ,  steady temperature residual ]
    with P_div the same Leray projection applied to the residual -- this removes
    grad(p) exactly (pressure is curl-free), so no pressure unknown is needed.

Here m = dim h ~ n, so the paper's dense m x m Schur solve is not applicable.
We instead solve the Gauss-Newton normal equations
      (J J^T) y = h(u),      u <- u - J^T y
MATRIX-FREE by Conjugate Gradients, using JVP for (J v) and VJP for (J^T w).
Each CG iteration costs two residual-sized autodiff passes -- no assembly, no
n x n storage. `--gn-cg` (default 12) caps the CG iterations, `--gn-iters`
(default 1) the Newton updates per flow step, matching the paper.

proj_start
--------------------------------------------------------------------------------
Projecting from tau=0 is wasteful: early flow states are near-noise and their
residual is meaningless. `--proj-start` (default 0.6) begins the correction only
once tau >= proj_start, i.e. over the last 40% of the trajectory, plus the final
projection. Set --proj-start 0.0 for the fully-interleaved variant, or 1.0 for
"final projection only" (Eq 7 alone).

USAGE (as a library)
    from rb3d_pcfm_sampler import PCFMProjector, pcfm_sample
    proj = PCFMProjector(Nx, Ny, Nz, aspect, device)
    fields, diag = pcfm_sample(model, ck, data, Ra_l, Pr_l, proj, ...)
"""

import math
import time

import torch

from generate_rb3d_multisolution import build_vertical, _kxy, leray_project


# ============================================================================
#  Differentiable steady-residual operators (torch, batched, autograd-friendly)
#  layout: (B, Nx, Ny, Nz); x,y periodic; z walls
# ============================================================================
def _kx_ky(Nx, Ny, Gx, Gy, device, dtype):
    kx = 2 * math.pi * torch.fft.fftfreq(Nx, d=Gx / Nx, device=device, dtype=dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(Ny, d=Gy / Ny, device=device, dtype=dtype)
    return kx, ky


def ddx(f, kx):
    return torch.fft.ifft(1j * kx[None, :, None, None]
                          * torch.fft.fft(f, dim=1), dim=1).real


def ddy(f, ky):
    return torch.fft.ifft(1j * ky[None, None, :, None]
                          * torch.fft.fft(f, dim=2), dim=2).real


def ddz(f, dz):
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * dz)
    out[:, :, :, 0] = (-3 * f[:, :, :, 0] + 4 * f[:, :, :, 1]
                       - f[:, :, :, 2]) / (2 * dz)
    out[:, :, :, -1] = (3 * f[:, :, :, -1] - 4 * f[:, :, :, -2]
                        + f[:, :, :, -3]) / (2 * dz)
    return out


def d2z(f, dz):
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1]
                          + f[:, :, :, :-2]) / dz ** 2
    return out


def build_sine(Nz, device, dtype):
    """Sine basis for the vertical Laplacian, MATCHING the generator's implicit
    diffusion operator (exact eigenvalues -(m*pi)^2). Using the FD d2z here
    instead leaves an O(1) operator-mismatch residual on a true steady state
    (measured: rel 0.27), which the projector could never remove -- Newton would
    be chasing a target that the data itself does not satisfy."""
    zz = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    m = torch.arange(Nz, device=device, dtype=dtype)
    S = torch.sin(math.pi * torch.outer(zz, m))
    return S, torch.linalg.pinv(S), (math.pi * m) ** 2


def lap_sine(f, kx, ky, sine):
    """Horizontal spectral + sine-spectral vertical (generator-consistent)."""
    S, Spi, mpi2 = sine
    fh = torch.fft.fft2(f, dim=(1, 2))
    k2 = (kx[:, None] ** 2 + ky[None, :] ** 2)[None, :, :, None]
    lxy = torch.fft.ifft2(-k2 * fh, dim=(1, 2)).real
    a = torch.einsum('mn,bxyn->bxym', Spi, f)
    lz = torch.einsum('jm,bxym->bxyj', S, -(mpi2[None, None, None, :]) * a)
    return lxy + lz


def lap(f, kx, ky, dz):
    fh = torch.fft.fft2(f, dim=(1, 2))
    k2 = (kx[:, None] ** 2 + ky[None, :] ** 2)[None, :, :, None]
    lxy = torch.fft.ifft2(-k2 * fh, dim=(1, 2)).real
    return lxy + d2z(f, dz)


class SteadyResidual:
    """h(u) for steady RB. Returns the stacked, pressure-free residual.
    Incompressibility + walls are NOT in h -- they are enforced exactly by the
    Leray projector, so h only carries the nonlinear steady equations.

    WALL MASK: the reference data is produced by a fractional-step scheme whose
    momentum balance carries an O(sqrt(Pr*dt)) slip layer in the first few wall
    planes (measured: ~98% of the raw momentum-residual energy lives there, even
    for a fully converged steady state). The verifier therefore gates on the
    BULK momentum residual. We mask the same `wall_cut` planes out of the
    momentum rows of h, so Gauss-Newton enforces the physics the data actually
    satisfies rather than chasing a scheme artifact. The temperature equation
    has no such layer and is kept everywhere.
    """

    def __init__(self, Nx, Ny, Nz, aspect, device, dtype=torch.float32,
                 wall_cut=3):
        Gx, Gy = aspect
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        self.dx, self.dy = Gx / Nx, Gy / Ny
        self.dz = 1.0 / (Nz - 1)
        self.wall_cut = wall_cut
        self.kx, self.ky = _kx_ky(Nx, Ny, Gx, Gy, device, dtype)
        self.sine = build_sine(Nz, device, dtype)
        self.VS = build_vertical(Nz, device, dtype)
        gkx, gky = _kxy(Nx, Ny, Gx / Nx, Gy / Ny, device, dtype)
        self.gkx, self.gky = gkx, gky
        self.k2 = gkx[:, None] ** 2 + gky[None, :] ** 2
        self.z = torch.linspace(0, 1, Nz, device=device, dtype=dtype)
        self.k2h = (self.kx[:, None] ** 2 + self.ky[None, :] ** 2)  # (Nx,Ny)
        mask = torch.ones(Nz, device=device, dtype=dtype)
        if wall_cut > 0:
            mask[:wall_cut] = 0.0
            mask[Nz - wall_cut:] = 0.0
        self.wmask = mask.view(1, 1, 1, Nz)

    def project_div(self, u, v, w):
        """Exact orthogonal projection onto {div u = 0, u|walls = 0}."""
        return leray_project(u, v, w, self.VS, self.k2, self.gkx, self.gky)

    def __call__(self, state, Ra, Pr, scales=None):
        """state: (B,4,Nx,Ny,Nz) = (u,v,w,T'). Returns (B,4,Nx,Ny,Nz) residual.
        Uses the SAME operators the generator marched to zero (FD-roll advection
        in x,y + sine-spectral vertical Laplacian), so a true steady state has
        near-zero h. Momentum residual is Leray-projected -> grad(p) eliminated,
        then wall-masked (see class docstring)."""
        u, v, w, Tp = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        kx, ky, dz, sine = self.kx, self.ky, self.dz, self.sine
        dx, dy = self.dx, self.dy
        Prv = Pr.view(-1, 1, 1, 1)
        Rav = Ra.view(-1, 1, 1, 1)

        def fdx(a):
            return (torch.roll(a, -1, 1) - torch.roll(a, 1, 1)) / (2 * dx)

        def fdy(a):
            return (torch.roll(a, -1, 2) - torch.roll(a, 1, 2)) / (2 * dy)

        def adv(a):
            return u * fdx(a) + v * fdy(a) + w * ddz(a, dz)

        Ru = adv(u) - Prv * lap_sine(u, kx, ky, sine)
        Rv = adv(v) - Prv * lap_sine(v, kx, ky, sine)
        Rw = adv(w) - Prv * lap_sine(w, kx, ky, sine) - Prv * Rav * Tp
        # kill grad(p): project the momentum residual to the div-free subspace
        Ru, Rv, Rw = self.project_div(Ru, Rv, Rw)
        Ru = Ru * self.wmask
        Rv = Rv * self.wmask
        Rw = Rw * self.wmask
        RT = adv(Tp) - w - lap_sine(Tp, kx, ky, sine)
        # ROW SCALING, done right. In physical units the momentum block
        # (~Pr*Ra*T' ~ 1e5) drowns the temperature block (~lap T' ~ 1e2), so
        # Gauss-Newton minimising |h| corrects ONLY momentum: measured on a
        # perturbed steady state, rm dropped 0.93 -> 0.24 while rt sat
        # untouched at 1.00 -- exactly the evaluation failure, since physL2
        # counts both. Worse, on a clean GT input the momentum-dominated
        # objective let GN CORRUPT temperature (rt 0.004 -> 0.39) to buy a
        # marginal momentum gain through the Pr*Ra buoyancy coupling. The two
        # blocks are therefore normalised to comparable size. CRITICAL DETAIL:
        # the normalisers must be FROZEN across a Newton step and its line
        # search (passed via `scales`); an earlier attempt recomputed them
        # from each candidate state, so the line search compared |h| values
        # with DIFFERENT denominators and rejected every step.
        if scales is None:
            s_M = (Prv * Rav * Tp).flatten(1).norm(dim=1) \
                .clamp_min(1e-9).detach().view(-1, 1, 1, 1)
            s_T = lap_sine(Tp, kx, ky, sine).flatten(1).norm(dim=1) \
                .clamp_min(1e-9).detach().view(-1, 1, 1, 1)
        else:
            s_M, s_T = scales
        self._sM, self._sT, self._Pr = s_M, s_T, Prv.detach()
        return torch.stack([Ru / s_M, Rv / s_M, Rw / s_M, RT / s_T], dim=1)


# ============================================================================
#  Gauss-Newton projection, matrix-free (CG on the normal equations)
# ============================================================================
def _precondition(res, r):
    """Apply M^{-1} r where M approximates J J^T by its dominant LINEAR block:
    momentum rows  ~ (Pr * lap / s_M)^2,  temperature row ~ (lap / s_T)^2,
    both DIAGONAL in the Fourier(x,y) x sine(z) basis with eigenvalue
    lam = (kx^2 + ky^2 + (m*pi)^2). This is what makes Gauss-Newton usable at
    high resolution: cond(J J^T) grows like k_max^4 (measured: raw CG with 8
    iterations produced steps so poor the line search rejected ALL of them at
    128x64x49), and dividing by lam^2 collapses exactly that growth. The
    advection/buoyancy couplings are lower-order and left to CG."""
    S, Spi, mpi2 = res.sine
    k2h = res.k2h[None, :, :, None]                       # (1,Nx,Ny,1)
    lam = k2h + mpi2[None, None, None, :]                 # (1,Nx,Ny,Nz) eigen
    lam = lam.clamp_min(float(mpi2[1]))                   # floor at first mode
    sM, sT, Pr = res._sM, res._sT, res._Pr
    out = torch.empty_like(r)
    Spi_c = Spi.to(torch.complex64 if r.dtype == torch.float32
                   else torch.complex128)
    S_c = S.to(Spi_c.dtype)
    for c in range(4):
        scale = Pr if c < 3 else torch.ones_like(Pr)      # (B,1,1,1)
        rh = torch.fft.fft2(r[:, c], dim=(1, 2))
        a = torch.einsum('mn,bxyn->bxym', Spi_c, rh)      # sine coeffs
        a = a / ((scale.to(a.dtype)) ** 2 * (lam.to(a.dtype)) ** 2)
        rh = torch.einsum('jm,bxym->bxyj', S_c, a)
        out[:, c] = torch.fft.ifft2(rh, dim=(1, 2)).real
    return out


class PCFMProjector:
    """u_proj = u - J^T (J J^T)^{-1} h(u), one Gauss-Newton step (Eq. 4).
    J J^T is applied matrix-free: (J J^T) y = J (J^T y) via VJP then JVP."""

    def __init__(self, Nx, Ny, Nz, aspect, device, dtype=torch.float32,
                 cg_iters=12, cg_tol=1e-6, damping=1e-4, wall_cut=3,
                 precond=False):
        """`damping` is RELATIVE: the Levenberg term added to J J^T is
        damping * lam_est, where lam_est estimates lam_max(J J^T) at the
        current state via power iteration. A FIXED absolute damping is a trap:
        after row-scaling h, the whole JJ^T spectrum shrinks by ~1e12, and the
        old absolute 1e-6 became LARGER than lam_max (measured 4.9e-6 at
        96x48x33) -- the CG solve was damping-dominated and the 'Gauss-Newton'
        direction was garbage that no line search could rescue."""
        self.res = SteadyResidual(Nx, Ny, Nz, aspect, device, dtype,
                                  wall_cut=wall_cut)
        self.cg_iters, self.cg_tol, self.damping = cg_iters, cg_tol, damping
        # The spectral preconditioner's diagonal-Laplacian model over-predicts
        # lam_max(J J^T) by ~2 orders (wall mask + Leray deflate the top of
        # the spectrum), which mis-scales the Krylov space and measurably
        # SLOWS descent (0.92->0.95 with it vs 0.92->0.56 without, one GN
        # step, cg=8). Off by default; kept for experimentation.
        self.precond = precond
        self.device = device

    def _hvp_ops(self, state, Ra, Pr, scales=None):
        state = state.detach().requires_grad_(True)

        def f(s):
            return self.res(s, Ra, Pr, scales=scales)

        h = f(state)

        def Jt(w):                                    # VJP: J^T w
            return torch.autograd.grad(h, state, grad_outputs=w,
                                       retain_graph=True)[0]

        def J(v):                                     # JVP via double-backward
            dummy = torch.zeros_like(h, requires_grad=True)
            g = torch.autograd.grad(h, state, grad_outputs=dummy,
                                    create_graph=True)[0]
            return torch.autograd.grad(g, dummy, grad_outputs=v,
                                       retain_graph=True)[0]
        return state, h.detach(), J, Jt

    @staticmethod
    def _dot(a, b):
        return (a * b).flatten(1).sum(1).view(-1, 1, 1, 1, 1)

    def project(self, state, Ra, Pr, n_newton=1):
        """Return the projected state and the residual norms before/after."""
        s = state
        r0 = None
        for _ in range(max(n_newton, 1)):
            with torch.no_grad():
                _ = self.res(s, Ra, Pr)          # computes + stashes scales
                frozen = (self.res._sM, self.res._sT)
            st, h, J, Jt = self._hvp_ops(s, Ra, Pr, scales=frozen)
            if r0 is None:
                r0 = h.flatten(1).norm(dim=1)
            # estimate lam_max(J J^T) at this state (3 power iterations) so the
            # Levenberg damping can be scaled RELATIVE to the spectrum
            with torch.no_grad():
                hn = h.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1)
            v = h / hn.clamp_min(1e-30)
            lam_est = None
            for _ in range(3):
                w = J(Jt(v))
                lam_est = w.flatten(1).norm(dim=1).view(-1, 1, 1, 1, 1)
                v = w / lam_est.clamp_min(1e-30)
            damp = self.damping * lam_est.clamp_min(1e-30)
            # solve (J J^T + damp I) y = h  by PRECONDITIONED CG, with the
            # spectral (Laplacian-diagonal) preconditioner -- see _precondition
            y = torch.zeros_like(h)
            r = h.clone()
            zv = _precondition(self.res, r) if self.precond else r.clone()
            p = zv.clone()
            rz = self._dot(r, zv)
            for _ in range(self.cg_iters):
                Ap = J(Jt(p)) + damp * p
                denom = self._dot(p, Ap).clamp_min(1e-30)
                alpha = rz / denom
                y = y + alpha * p
                r = r - alpha * Ap
                if float(self._dot(r, r).max().sqrt()) < self.cg_tol:
                    break
                zv = _precondition(self.res, r) if self.precond else r
                rz_new = self._dot(r, zv)
                p = zv + (rz_new / rz.clamp_min(1e-30)) * p
                rz = rz_new
            delta = Jt(y)                                   # J^T y
            # BACKTRACKING LINE SEARCH: a full GN step computed from an
            # under-converged CG direction can overshoot the linearisation
            # and INCREASE |h| (observed at 128x64x49, where cond(J J^T) is
            # ~16x the coarse grid's). Try alpha = 1, 1/2, 1/4, 1/8 and accept
            # the first that reduces the residual norm; if none does, keep the
            # state unchanged -- the projection must never make a sample worse.
            base = st.detach()
            h0 = h.flatten(1).norm(dim=1)

            def _apply(alpha):
                t = (base - alpha * delta).detach()
                u, v, w = self.res.project_div(t[:, 0].contiguous(),
                                               t[:, 1].contiguous(),
                                               t[:, 2].contiguous())
                Tp = t[:, 3]
                Tp[:, :, :, 0] = 0.0
                Tp[:, :, :, -1] = 0.0
                return torch.stack([u, v, w, Tp], dim=1)

            s = base
            best = h0
            with torch.no_grad():
                for alpha in (1.0, 0.5, 0.25, 0.125):
                    cand = _apply(alpha)
                    hn = self.res(cand, Ra, Pr, scales=frozen)\
                        .flatten(1).norm(dim=1)
                    if bool((hn < best).all()):
                        s, best = cand, hn
                        break
        with torch.no_grad():
            r1 = self.res(s, Ra, Pr).flatten(1).norm(dim=1)
        return s, r0, r1


# ============================================================================
#  PCFM sampling loop (Algorithm 1)
# ============================================================================
@torch.no_grad()
def _velocity(model, x, t, cond):
    return model(x, t, cond)


def pcfm_sample(model, ck, data, Ra_l, Pr_l, projector, n_step=50,
                proj_start=0.6, n_newton=1, seed=0, final_projection=True,
                vanilla=False):
    """Euler flow + interleaved Gauss-Newton projection (PCFM).
    vanilla=True  -> plain flow matching, no correction at all (the ablation).
    Returns (fields (B,4,Nx,Ny,Nz) in PHYSICAL units, diag dict)."""
    device = next(model.parameters()).device
    B = len(Ra_l)
    t0 = time.perf_counter()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    x = torch.randn(B, 4, data.Nx, data.Ny, data.Nz, generator=gen).to(device)
    u0 = x.clone()                                    # kept for the OT reverse
    cond = data.norm_params(torch.tensor(Ra_l, dtype=torch.float64),
                            torch.tensor(Pr_l, dtype=torch.float64)) \
        .float().to(device)
    cond = cond if ck['cond'] else None
    scale = ck['scale'].to(device).view(1, 4, 1, 1, 1)
    Ra = torch.tensor(Ra_l, device=device, dtype=torch.float32)
    Pr = torch.tensor(Pr_l, device=device, dtype=torch.float32)

    ts = torch.linspace(0, 1, n_step + 1, device=device)
    n_proj = 0
    for i in range(n_step):
        tau, tau_p = ts[i], ts[i + 1]
        dt = float(tau_p - tau)
        # ---- plain Euler flow update
        v = _velocity(model, x, tau.expand(B), cond)
        x = x + dt * v
        if vanilla or float(tau_p) < proj_start:
            continue
        # ---- 1. shoot to tau=1 with one cheap Euler step
        x1 = x + (1.0 - float(tau_p)) * _velocity(model, x, tau_p.expand(B), cond)
        # ---- 2. Gauss-Newton projection in PHYSICAL units
        with torch.enable_grad():
            phys = x1 * scale
            phys_p, _, _ = projector.project(phys, Ra, Pr, n_newton=n_newton)
        x1_proj = (phys_p / scale).detach()
        n_proj += 1
        # ---- 3. reverse with the OT displacement interpolant (Eq. 5)
        #        u_hat(tau') = u_proj + (tau'-1) * (u_proj - u0)
        x = x1_proj + (float(tau_p) - 1.0) * (x1_proj - u0)

    # ---- final full projection (Eq. 7)
    if final_projection and not vanilla:
        with torch.enable_grad():
            phys = x * scale
            try:
                phys, r0, r1 = projector.project(phys, Ra, Pr,
                                                 n_newton=n_newton, final=True)
            except TypeError:
                phys, r0, r1 = projector.project(phys, Ra, Pr,
                                                 n_newton=n_newton)
        n_proj += 1
    else:
        phys = x * scale
        with torch.no_grad():
            r0 = r1 = projector.res(phys, Ra, Pr).flatten(1).norm(dim=1)

    diag = dict(seconds=time.perf_counter() - t0, n_projections=n_proj,
                res_before=r0.detach().cpu(), res_after=r1.detach().cpu())
    return phys.detach().cpu(), diag


# ============================================================================
#  PTC projection: pseudo-transient continuation as the h(u)=0 solver
# ============================================================================
class PTCProjector:
    """Projection onto the steady manifold by PSEUDO-TRANSIENT CONTINUATION:
    a short, fixed budget of IMEX+Leray pseudo-time steps from the current
    iterate. PTC is a standard Newton globalisation for stiff nonlinear
    elliptic systems -- the implicit diffusion step is exact for precisely the
    k^2-stiff part that defeats a few-iteration matrix-free CG.

    WHY THIS EXISTS ALONGSIDE PCFMProjector (Gauss-Newton):
      * measured at 96x48x33 and 128x64x49, matrix-free GN with a feasible
        CG budget (8-15 iterations) corrects the momentum block but leaves the
        temperature residual essentially untouched (rt frozen at ~1.0), and
        the interleaved loop then ends WORSE than vanilla in physL2;
      * the IMEX relaxer -- the same operator that GENERATED the data -- fixes
        all three blocks together, cheaply (a few FFTs per step, no autodiff).
    TRADE-OFF, stated honestly: pseudo-time flows to ATTRACTORS, so this
    projection can only land on stable branches; Gauss-Newton in principle
    reaches unstable ones (root-finding is stability-blind) but needs a far
    stronger inner solver at this resolution. For novelty claims involving
    unstable branches, use --projector gn with a large --cg-iters and expect
    it to be slow; for evaluation, PTC is the right tool.

    The fixed SMALL step budget keeps this a LOCAL projection (it moves the
    sample to the nearest attractor basin's floor, not across basins), and the
    interleaved PCFM loop structure (shoot -> project -> OT-reverse) is
    unchanged -- only the inner solver of the projection differs.
    """

    def __init__(self, Nx, Ny, Nz, aspect, device, dtype=torch.float32,
                 steps=200, final_steps=600, cfl=0.30, wall_cut=3, **_):
        from rb3d_pcfm_common import RB3DRelaxer

        class _View:
            pass
        v = _View()
        v.Nx, v.Ny, v.Nz, v.aspect = Nx, Ny, Nz, tuple(aspect)
        self.relaxer = RB3DRelaxer(v, device)
        self.steps, self.final_steps, self.cfl = steps, final_steps, cfl
        # diagnostics-compatible residual (same as the GN projector uses)
        self.res = SteadyResidual(Nx, Ny, Nz, aspect, device, dtype,
                                  wall_cut=wall_cut)
        self.device = device

    def project(self, state, Ra, Pr, n_newton=1, final=False):
        """state: (B,4,Nx,Ny,Nz) physical units. Returns (state', r0, r1) with
        r0/r1 the row-scaled |h| before/after (diagnostics only)."""
        with torch.no_grad():
            r0 = self.res(state, Ra, Pr).flatten(1).norm(dim=1)
            u, v, w, Tp = (state[:, 0].contiguous(), state[:, 1].contiguous(),
                           state[:, 2].contiguous(), state[:, 3].contiguous())
            n = self.final_steps if final else self.steps
            u, v, w, Tp, _ = self.relaxer.relax(
                u, v, w, Tp, Ra.tolist(), Pr.tolist(), steps=n, cfl=self.cfl)
            out = torch.stack([u, v, w, Tp], dim=1)
            r1 = self.res(out, Ra, Pr).flatten(1).norm(dim=1)
        return out, r0, r1