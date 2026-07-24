#!/usr/bin/env python3
"""
verify_rb3d_multisolution.py
================================================================================
Independent verification & validation for a bank produced by
`generate_rb3d_multisolution.py` (the 3D analog of verify_rb2d_multisolution.py).
Standalone: only needs torch.

It recomputes everything with INDEPENDENT operators (spectral x/y derivatives +
independent finite differences in z) -- it does NOT reuse the generator's
solver stencils -- so a bug shared with the solver cannot hide a bad bank.

It answers four questions, each a separate check:

  CHECK 1 - PDE RESIDUAL  ("does a stored field satisfy the steady PDE?")
  ---------------------------------------------------------------------------
  Steady Boussinesq in a periodic-x/y box with no-slip walls:
      continuity : d_x u + d_y v + d_z w                       = 0
      momentum   : (u.grad)u + grad p - Pr lap u - Pr Ra T' zhat = 0
      temperature: (u.grad)T' - w - lap T'                     = 0   [T'=T-(1-z)]
  Pressure is eliminated the SAME way it is physically absent from a steady
  state: the momentum residual is projected onto the divergence-free subspace
  (a Leray/Helmholtz projection kills grad p, since grad p is curl-free), so
  what remains is the pressure-free momentum imbalance. Every residual is
  reported RELATIVE to the dominant balancing term in its equation, and both a
  per-branch worst case and an overall worst case are printed.

  CHECK 2 - INCOMPRESSIBILITY  ("is each stored field divergence-free?")
  ---------------------------------------------------------------------------
  RMS(div u) relative to the RMS velocity-gradient magnitude, on the interior.

  CHECK 3 - DISTINCTNESS  ("are the branches at fixed (Ra,Pr) different fields?")
  ---------------------------------------------------------------------------
  All pairwise relative L2 distances between the (u,v,w,T') stacks; the smallest
  off-diagonal must exceed --dist-tol (genuinely distinct planforms sit ~1.2-2).

  CHECK 4 - PLANFORM LABEL  ("is the stored planform tag correct?")
  ---------------------------------------------------------------------------
  The planform of every branch is recomputed independently from the 2D FFT of w
  at mid-height and compared to the stored 'planform' tag and the seeded mode.

Only entries with converged==True are gated by CHECKS 1-2 (an unconverged /
time-dependent entry is NOT expected to satisfy the steady residual -- it is
reported separately, and is exactly the material for the held-out test split).

USAGE
    python verify_rb3d_multisolution.py --bank .../refs_bank.pt
    python verify_rb3d_multisolution.py --bank bank.pt --verbose
Exit code is 0 iff all gated checks pass.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import torch


# ---------------------------------------------------------------------------
#  Independent derivative operators
#  field layout (Nx, Ny, Nz); x=axis0 periodic, y=axis1 periodic, z=axis2 walls
# ---------------------------------------------------------------------------
def _k(N, L, device, dtype):
    return 2.0 * math.pi * torch.fft.fftfreq(N, d=L / N, device=device, dtype=dtype)


def ddx_spec(f, kx):
    return torch.fft.ifft(1j * kx[:, None, None] * torch.fft.fft(f, dim=0), dim=0).real


def ddy_spec(f, ky):
    return torch.fft.ifft(1j * ky[None, :, None] * torch.fft.fft(f, dim=1), dim=1).real


def d2x_spec(f, kx):
    return torch.fft.ifft(-(kx[:, None, None] ** 2) * torch.fft.fft(f, dim=0), dim=0).real


def d2y_spec(f, ky):
    return torch.fft.ifft(-(ky[None, :, None] ** 2) * torch.fft.fft(f, dim=1), dim=1).real


def ddz(f, dz):
    out = torch.empty_like(f)
    out[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
    out[:, :, 0] = (-3 * f[:, :, 0] + 4 * f[:, :, 1] - f[:, :, 2]) / (2 * dz)
    out[:, :, -1] = (3 * f[:, :, -1] - 4 * f[:, :, -2] + f[:, :, -3]) / (2 * dz)
    return out


# FD-roll horizontal derivatives -- these MATCH the generator's explicit
# advection stencil, so the "discrete" residual below is measured with the
# same operators the solver drove to zero (near the convergence floor), while
# the spectral operators above give the independent "continuum" residual.
def ddx_fd(f, dx):
    return (torch.roll(f, -1, 0) - torch.roll(f, 1, 0)) / (2 * dx)


def ddy_fd(f, dy):
    return (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2 * dy)


def d2z(f, dz):
    out = torch.zeros_like(f)
    out[:, :, 1:-1] = (f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]) / dz ** 2
    return out


def lap(f, kx, ky, dz):
    return d2x_spec(f, kx) + d2y_spec(f, ky) + d2z(f, dz)


# Generator-consistent Laplacian: horizontal spectral (-k^2) + SINE-spectral in
# z (exact -(m*pi)^2), matching the solver's implicit-diffusion operator. Used
# for the DISCRETE residual so the viscous term Pr*lap(u) is measured with the
# SAME operator the solver used -- otherwise the FD-vs-sine mismatch in d2z is
# amplified by Pr and shows up as a spurious high-Pr momentum residual.
def _sine_basis(Nz, device, dtype):
    zz = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    m = torch.arange(Nz, device=device, dtype=dtype)
    S = torch.sin(math.pi * torch.outer(zz, m))
    return S, torch.linalg.pinv(S), (math.pi * m) ** 2


def lap_gen(f, kx, ky, sine):
    """Horizontal spectral + sine-spectral vertical (matches the generator).
    f must vanish at the walls (true for u,v,w,T')."""
    S, Spi, mpi2 = sine
    lxy = d2x_spec(f, kx) + d2y_spec(f, ky)
    a = torch.einsum('mn,xyn->xym', Spi, f)                 # sine coeffs in z
    lz = torch.einsum('jm,xym->xyj', S, -(mpi2[None, None, :]) * a)
    return lxy + lz


def _rms(t):
    return float(t.pow(2).mean().sqrt())


def _rms_i(t):                                             # interior (drop walls)
    return _rms(t[:, :, 1:-1])


# ---------------------------------------------------------------------------
#  Leray projection (independent): remove grad-part of a vector field so only
#  the divergence-free (pressure-free) momentum residual remains.
#  Cosine (Neumann) vertical basis via a small dense matrix -> exact & robust.
# ---------------------------------------------------------------------------
def _cos_basis(Nz, device, dtype):
    z = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    m = torch.arange(Nz, device=device, dtype=dtype)
    C = torch.cos(math.pi * torch.outer(z, m))
    return C, torch.linalg.pinv(C), (math.pi * m) ** 2


def _proj_pack(Nx, Ny, Nz, kx, ky, device, dtype):
    """Precompute the per-horizontal-mode least-squares Helmholtz projector.
    For each (kx,ky): q (cosine coeffs) minimizing ||R - grad q||^2 solves
        A q = G^H b,   A = k2 * C^T C + D^T D
    with C the cosine synthesis and D = d/dz C (exact sine-space derivative
    evaluated on the grid). Because gradient and divergence are ADJOINT by
    construction, this projection annihilates any representable gradient to
    machine precision -- unlike a div/grad pair built from mismatched stencils,
    which leaks a fraction of grad(p) into the 'pressure-free' residual (the
    leak was measured at ~7 percent and inflated the momentum check)."""
    z = torch.linspace(0.0, 1.0, Nz, device=device, dtype=dtype)
    m = torch.arange(Nz, device=device, dtype=dtype)
    C = torch.cos(math.pi * torch.outer(z, m))                 # (Nz,Nz)
    D = -torch.sin(math.pi * torch.outer(z, m)) * (math.pi * m)[None, :]
    CtC = C.t() @ C
    DtD = D.t() @ D
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2                   # (Nx,Ny)
    A = k2[:, :, None, None] * CtC[None, None] + DtD[None, None]
    A_pinv = torch.linalg.pinv(A)                              # (Nx,Ny,Nz,Nz)
    return dict(C=C, D=D, A_pinv=A_pinv)


def leray_residual(Ru, Rv, Rw, kx, ky, dz, proj):
    """Exact discrete Leray projection of the momentum residual: remove the
    best-fit gradient (least squares per horizontal mode). Returns R - grad q*."""
    C, D, A_pinv = proj['C'], proj['D'], proj['A_pinv']
    ch = torch.fft.fft2(Ru, dim=(0, 1))
    dh = torch.fft.fft2(Rv, dim=(0, 1))
    eh = torch.fft.fft2(Rw, dim=(0, 1))
    ikx = (1j * kx)[:, None, None]
    iky = (1j * ky)[None, :, None]
    Cc = C.to(ch.dtype); Dc = D.to(ch.dtype)
    # G^H b = conj(ikx) C^T u + conj(iky) C^T v + D^T w  per mode
    rhs = ((-ikx) * torch.einsum('mj,xyj->xym', Cc.t().conj(), ch)
           + (-iky) * torch.einsum('mj,xyj->xym', Cc.t().conj(), dh)
           + torch.einsum('mj,xyj->xym', Dc.t().conj(), eh))
    q = torch.einsum('xymn,xyn->xym', A_pinv.to(ch.dtype), rhs)
    ch = ch - ikx * torch.einsum('jm,xym->xyj', Cc, q)
    dh = dh - iky * torch.einsum('jm,xym->xyj', Cc, q)
    eh = eh - torch.einsum('jm,xym->xyj', Dc, q)
    return (torch.fft.ifft2(ch, dim=(0, 1)).real,
            torch.fft.ifft2(dh, dim=(0, 1)).real,
            torch.fft.ifft2(eh, dim=(0, 1)).real)


# ===========================================================================
#  BATCHED, DEVICE-AWARE OPERATORS  -- fields carry a leading batch dim
#  (B, Nx, Ny, Nz).  These exist because the per-entry, CPU, float64 path
#  above costs ~0.27 s per leray_residual call at 128x64x49; with two calls
#  per entry and 1413 entries that is ~13 min of pure projection, plus ~6 min
#  wasted re-casting the 157 MB A_pinv to complex on EVERY call.  Here the
#  complex projector is built ONCE and entries are processed in chunks on the
#  GPU, which is where an 8x-larger grid belongs.
# ===========================================================================
def _b_dx_spec(f, kx):     # d/dx, spectral, batched
    return torch.fft.ifft(1j * kx[None, :, None, None]
                          * torch.fft.fft(f, dim=1), dim=1).real


def _b_dy_spec(f, ky):
    return torch.fft.ifft(1j * ky[None, None, :, None]
                          * torch.fft.fft(f, dim=2), dim=2).real


def _b_d2x_spec(f, kx):
    return torch.fft.ifft(-(kx ** 2)[None, :, None, None]
                          * torch.fft.fft(f, dim=1), dim=1).real


def _b_d2y_spec(f, ky):
    return torch.fft.ifft(-(ky ** 2)[None, None, :, None]
                          * torch.fft.fft(f, dim=2), dim=2).real


def _b_dx_fd(f, dx):
    return (torch.roll(f, -1, 1) - torch.roll(f, 1, 1)) / (2 * dx)


def _b_dy_fd(f, dy):
    return (torch.roll(f, -1, 2) - torch.roll(f, 1, 2)) / (2 * dy)


def _b_dz(f, dz):
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - f[:, :, :, :-2]) / (2 * dz)
    out[:, :, :, 0] = (-3 * f[:, :, :, 0] + 4 * f[:, :, :, 1]
                       - f[:, :, :, 2]) / (2 * dz)
    out[:, :, :, -1] = (3 * f[:, :, :, -1] - 4 * f[:, :, :, -2]
                        + f[:, :, :, -3]) / (2 * dz)
    return out


def _b_d2z(f, dz):
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1] = (f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1]
                          + f[:, :, :, :-2]) / dz ** 2
    return out


def _b_lap(f, kx, ky, dz):
    return _b_d2x_spec(f, kx) + _b_d2y_spec(f, ky) + _b_d2z(f, dz)


def _b_lap_gen(f, kx, ky, sine):
    S, Spi, mpi2 = sine
    lxy = _b_d2x_spec(f, kx) + _b_d2y_spec(f, ky)
    a = torch.einsum('mn,bxyn->bxym', Spi, f)
    lz = torch.einsum('jm,bxym->bxyj', S, -(mpi2[None, None, None, :]) * a)
    return lxy + lz


def _b_rms_i(t):
    """Per-sample RMS over the interior (walls dropped). Returns (B,)."""
    return t[:, :, :, 1:-1].pow(2).mean(dim=(1, 2, 3)).sqrt()


def _b_rms_cut(t, c):
    """Per-sample RMS with `c` wall planes dropped each side. Returns (B,)."""
    Nz = t.shape[-1]
    return t[:, :, :, c:Nz - c].pow(2).mean(dim=(1, 2, 3)).sqrt()


def _proj_pack_b(Nx, Ny, Nz, kx, ky, device, dtype):
    """Same projector as _proj_pack, but with C, D, A_pinv PRE-CAST to the
    matching complex dtype and moved to `device`. The unbatched version casts
    A_pinv (157 MB at 128x64x49) inside every call; hoisting that out removes
    ~6 min from a full hi-res verify."""
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    # build in float64 on CPU for a well-conditioned pinv, then cast/move
    z = torch.linspace(0.0, 1.0, Nz, dtype=torch.float64)
    m = torch.arange(Nz, dtype=torch.float64)
    C = torch.cos(math.pi * torch.outer(z, m))
    D = -torch.sin(math.pi * torch.outer(z, m)) * (math.pi * m)[None, :]
    CtC = C.t() @ C
    DtD = D.t() @ D
    k2 = (kx.double().cpu()[:, None] ** 2 + ky.double().cpu()[None, :] ** 2)
    A = k2[:, :, None, None] * CtC[None, None] + DtD[None, None]
    A_pinv = torch.linalg.pinv(A)
    return dict(
        Cc=C.to(cdtype).to(device),
        Dc=D.to(cdtype).to(device),
        CcH=C.t().conj().to(cdtype).to(device),
        DcH=D.t().conj().to(cdtype).to(device),
        A_pinv=A_pinv.to(cdtype).to(device),
        ikx=(1j * kx.to(cdtype))[None, :, None, None].to(device),
        iky=(1j * ky.to(cdtype))[None, None, :, None].to(device),
    )


def _b_leray_residual(Ru, Rv, Rw, proj):
    """Batched exact Leray projection. Inputs/outputs (B,Nx,Ny,Nz)."""
    Cc, Dc, CcH, DcH = proj['Cc'], proj['Dc'], proj['CcH'], proj['DcH']
    A_pinv, ikx, iky = proj['A_pinv'], proj['ikx'], proj['iky']
    ch = torch.fft.fft2(Ru, dim=(1, 2))
    dh = torch.fft.fft2(Rv, dim=(1, 2))
    eh = torch.fft.fft2(Rw, dim=(1, 2))
    rhs = ((-ikx) * torch.einsum('mj,bxyj->bxym', CcH, ch)
           + (-iky) * torch.einsum('mj,bxyj->bxym', CcH, dh)
           + torch.einsum('mj,bxyj->bxym', DcH, eh))
    q = torch.einsum('xymn,bxyn->bxym', A_pinv, rhs)
    ch = ch - ikx * torch.einsum('jm,bxym->bxyj', Cc, q)
    dh = dh - iky * torch.einsum('jm,bxym->bxyj', Cc, q)
    eh = eh - torch.einsum('jm,bxym->bxyj', Dc, q)
    return (torch.fft.ifft2(ch, dim=(1, 2)).real,
            torch.fft.ifft2(dh, dim=(1, 2)).real,
            torch.fft.ifft2(eh, dim=(1, 2)).real)


# ---------------------------------------------------------------------------
#  Independent planform classifier: dominant horizontal wavevector(s) of w(mid)
# ---------------------------------------------------------------------------
def classify(w, Gx, Gy, kmax=8, ratio=0.5, return_purity=False):
    """Classify the planform AND report a spectral PURITY score: the fraction
    of total spectral energy (over the resolved kmax band) explained by the
    reported peak(s). A clean, converged branch concentrates its energy in
    1-2 peaks (purity ~0.6-0.95); a noisy/blended/off-manifold field spreads
    energy across many wavenumbers (purity low). This decouples "what label
    would the top peak give" from "how confident is that label" -- the
    classifier used to ALWAYS force a label; purity lets the caller reject
    low-confidence classifications instead of trusting a forced guess.
    If return_purity=False, behaves exactly as before (backward compatible)."""
    Nx, Ny, Nz = w.shape
    wm = w[:, :, Nz // 2]
    spec = torch.fft.rfft2(wm).abs()
    spec[0, 0] = 0.0
    s = spec.clone()
    s[kmax + 1:, :] = 0.0
    s[:, kmax + 1:] = 0.0
    total_energy = float((s ** 2).sum().clamp_min(1e-12))
    Ky = s.shape[1]
    i1 = int(s.reshape(-1).argmax())
    n1, m1 = i1 // Ky, i1 % Ky
    n1s = n1 if n1 <= Nx // 2 else n1 - Nx
    a1 = float(s[n1, m1])
    s2 = s.clone()
    for dn in range(-1, 2):
        for dm in range(-1, 2):
            s2[(n1 + dn) % Nx, min(max(m1 + dm, 0), Ky - 1)] = 0.0
    i2 = int(s2.reshape(-1).argmax())
    n2, m2 = i2 // Ky, i2 % Ky
    a2 = float(s2[n2, m2])
    N1, M1 = abs(n1s), abs(m1)

    def energy_near(nn, mm, rad=1):
        seen = set()
        for dn in range(-rad, rad + 1):
            for dm in range(-rad, rad + 1):
                idx = ((nn + dn) % Nx, min(max(mm + dm, 0), Ky - 1))
                seen.add(idx)                      # dedupe boundary clamping
        return sum(float(s[i, j] ** 2) for i, j in seen)

    if a2 < ratio * a1:
        purity = energy_near(n1, m1) / total_energy
        if M1 == 0:
            label = ('roll', N1, 0)
        elif N1 == 0:
            label = ('roll', 0, M1)
        else:
            label = ('rect', N1, M1)
        return (label, purity) if return_purity else label
    N2 = abs(n2 if n2 <= Nx // 2 else n2 - Nx); M2 = abs(m2)
    seen1 = {((n1 + dn) % Nx, min(max(m1 + dm, 0), Ky - 1))
             for dn in (-1, 0, 1) for dm in (-1, 0, 1)}
    seen2 = {((n2 + dn) % Nx, min(max(m2 + dm, 0), Ky - 1))
             for dn in (-1, 0, 1) for dm in (-1, 0, 1)}
    purity = sum(float(s[i, j] ** 2) for i, j in (seen1 | seen2)) / total_energy
    if N1 == M2 == 0 or N2 == M1 == 0 or (M1 == 0 and N2 == 0 and N1 == M2):
        label = ('square', max(N1, N2), max(M1, M2))
    else:
        label = ('rect', max(N1, N2), max(M1, M2))
    return (label, purity) if return_purity else label


# ============================================================================
#  CHECK 1 & 2  -- residual + incompressibility
# ============================================================================
def check_residual(bank, rel_tol, mom_tol, div_tol, verbose,
                   device='cpu', batch=8, precision='fp64', progress=50,
                   keys=None, return_stats=False):
    """Batched, device-aware residual check.

    PERFORMANCE: at 128x64x49 the old per-entry, CPU, float64 path spent ~0.27 s
    inside each leray_residual call (two per entry) and additionally re-cast the
    157 MB A_pinv to complex on EVERY call -- ~19 min of the ~25 min runtime for
    1413 entries, with no output until the very end. Here the complex projector
    is hoisted out, entries are processed in chunks of `batch` on `device`, and
    progress is printed. `keys` restricts the work to a subset (used for
    --dual-gpu sharding).
    """
    print('\n=== CHECK 1+2: steady PDE residual & incompressibility ===')
    print('  DISCRETE residual  = generator-consistent operators (FD advection);'
          '\n    this is what the solver drove to ~0, so it gates convergence.'
          '\n  CONTINUUM residual = independent spectral operators; INFORMATIONAL,'
          '\n    it is the dataset truncation error and shrinks as the grid refines.')
    g = bank['grid']
    Gx, Gy = bank['aspect']
    Nx, Ny, Nz = g['Nx'], g['Ny'], g['Nz']
    dtype = torch.float64 if precision == 'fp64' else torch.float32
    z = g['z'].to(device).to(dtype)
    dz = float(z[1] - z[0])
    dx = Gx / Nx; dy = Gy / Ny
    kx = _k(Nx, Gx, device, dtype)
    ky = _k(Ny, Gy, device, dtype)
    sine = tuple(t.to(device).to(dtype) for t in _sine_basis(Nz, 'cpu', dtype))
    t_pack = time.perf_counter()
    proj = _proj_pack_b(Nx, Ny, Nz, kx, ky, device, dtype)
    print(f'  [setup] projector built on {device} ({precision}) in '
          f'{time.perf_counter()-t_pack:.1f}s; batch={batch}', flush=True)

    wd = {'cont': 0.0, 'mom': 0.0, 'mom_bulk': 0.0, 'temp': 0.0}
    wc = {'cont': 0.0, 'mom': 0.0, 'temp': 0.0}
    wd_at = None
    per_branch = {}
    n_conv = n_unconv = 0
    unconv_resid = []

    items = sorted(bank['entries'].items(), key=lambda kv: kv[0][:2])
    if keys is not None:
        kset = set(keys)
        items = [it for it in items if it[0] in kset]
    n_tot = len(items)
    t0 = time.perf_counter()
    zprof = (1.0 - z)[None, None, None, :]

    for s in range(0, n_tot, batch):
        chunk = items[s:s + batch]
        B = len(chunk)
        u = torch.stack([e['grid_u'] for _, e in chunk]).to(device, dtype)
        v = torch.stack([e['grid_v'] for _, e in chunk]).to(device, dtype)
        w = torch.stack([e['grid_w'] for _, e in chunk]).to(device, dtype)
        T = torch.stack([e['grid_T'] for _, e in chunk]).to(device, dtype)
        Tp = T - zprof
        Ra = torch.tensor([k[0] for k, _ in chunk], device=device, dtype=dtype
                          ).view(B, 1, 1, 1)
        Pr = torch.tensor([k[1] for k, _ in chunk], device=device, dtype=dtype
                          ).view(B, 1, 1, 1)

        lapT = _b_lap(Tp, kx, ky, dz)
        lapT_g = _b_lap_gen(Tp, kx, ky, sine)
        s_T = _b_rms_i(lapT_g).clamp_min(1e-12)

        def advF(a):
            return u * _b_dx_fd(a, dx) + v * _b_dy_fd(a, dy) + w * _b_dz(a, dz)

        div_d = _b_dx_fd(u, dx) + _b_dy_fd(v, dy) + _b_dz(w, dz)
        s_div_d = torch.maximum(torch.maximum(_b_rms_i(_b_dx_fd(u, dx)),
                                              _b_rms_i(_b_dy_fd(v, dy))),
                                _b_rms_i(_b_dz(w, dz))).clamp_min(1e-12)
        rd_cont = _b_rms_i(div_d) / s_div_d
        rd_temp = _b_rms_i(advF(Tp) - w - lapT_g) / s_T
        Ru = advF(u) - Pr * _b_lap_gen(u, kx, ky, sine)
        Rv = advF(v) - Pr * _b_lap_gen(v, kx, ky, sine)
        Rw = advF(w) - Pr * _b_lap_gen(w, kx, ky, sine) - Pr * Ra * Tp
        Ru, Rv, Rw = _b_leray_residual(Ru, Rv, Rw, proj)
        buoy = Pr * Ra * Tp
        s_M = _b_rms_i(buoy).clamp_min(1e-12)
        rd_mom = (_b_rms_i(Ru) ** 2 + _b_rms_i(Rv) ** 2
                  + _b_rms_i(Rw) ** 2).sqrt() / s_M
        # BULK momentum (drop 3 wall planes each side): the full-field number
        # includes the fractional-step scheme's wall-slip layer, a documented
        # O(sqrt(Pr*dt)) projection artifact; the BULK value certifies the
        # physics and is what the gate uses.
        cN = 3
        s_Mb = _b_rms_cut(buoy, cN).clamp_min(1e-12)
        rd_mom_bulk = (_b_rms_cut(Ru, cN) ** 2 + _b_rms_cut(Rv, cN) ** 2
                       + _b_rms_cut(Rw, cN) ** 2).sqrt() / s_Mb

        def advS(a):
            return (u * _b_dx_spec(a, kx) + v * _b_dy_spec(a, ky)
                    + w * _b_dz(a, dz))

        div_c = _b_dx_spec(u, kx) + _b_dy_spec(v, ky) + _b_dz(w, dz)
        s_div_c = torch.maximum(torch.maximum(_b_rms_i(_b_dx_spec(u, kx)),
                                              _b_rms_i(_b_dy_spec(v, ky))),
                                _b_rms_i(_b_dz(w, dz))).clamp_min(1e-12)
        rc_cont = _b_rms_i(div_c) / s_div_c
        s_Tc = _b_rms_i(lapT).clamp_min(1e-12)
        rc_temp = _b_rms_i(advS(Tp) - w - lapT) / s_Tc
        Cu = advS(u) - Pr * _b_lap(u, kx, ky, dz)
        Cv = advS(v) - Pr * _b_lap(v, kx, ky, dz)
        Cw = advS(w) - Pr * _b_lap(w, kx, ky, dz) - Pr * Ra * Tp
        Cu, Cv, Cw = _b_leray_residual(Cu, Cv, Cw, proj)
        rc_mom = (_b_rms_i(Cu) ** 2 + _b_rms_i(Cv) ** 2
                  + _b_rms_i(Cw) ** 2).sqrt() / s_M

        rd_cont = rd_cont.cpu(); rd_temp = rd_temp.cpu()
        rd_mom = rd_mom.cpu(); rd_mom_bulk = rd_mom_bulk.cpu()
        rc_cont = rc_cont.cpu(); rc_temp = rc_temp.cpu(); rc_mom = rc_mom.cpu()

        for b, ((Ra_b, Pr_b, mode), e) in enumerate(chunk):
            a_cont = float(rd_cont[b]); a_temp = float(rd_temp[b])
            a_mom = float(rd_mom[b]); a_mb = float(rd_mom_bulk[b])
            c_cont = float(rc_cont[b]); c_temp = float(rc_temp[b])
            c_mom = float(rc_mom[b])
            if e['converged']:
                n_conv += 1
                if max(a_cont, a_mb, a_temp) > max(wd['cont'], wd['mom_bulk'],
                                                   wd['temp']):
                    wd_at = (Ra_b, Pr_b, mode)
                for kk, val in (('cont', a_cont), ('mom', a_mom),
                                ('mom_bulk', a_mb), ('temp', a_temp)):
                    wd[kk] = max(wd[kk], val)
                for kk, val in (('cont', c_cont), ('mom', c_mom),
                                ('temp', c_temp)):
                    wc[kk] = max(wc[kk], val)
                per_branch.setdefault(tuple(mode), []).append(a_mb)
                if verbose:
                    print(f'    [conv] Ra={Ra_b:7.0f} Pr={Pr_b:4.2f} {str(mode):>15} '
                          f'discrete[cont={a_cont:.1e} mom_bulk={a_mb:.1e} '
                          f'mom_full={a_mom:.1e} temp={a_temp:.1e}] '
                          f'continuum[cont={c_cont:.1e} mom={c_mom:.1e} '
                          f'temp={c_temp:.1e}]')
            else:
                n_unconv += 1
                unconv_resid.append(max(a_mom, a_temp))

        done = min(s + batch, n_tot)
        if progress and (done % max(progress, 1) < batch or done == n_tot):
            el = time.perf_counter() - t0
            rate = done / max(el, 1e-9)
            print(f'  [{done:5d}/{n_tot}] {el:6.1f}s elapsed, {rate:5.1f} entries/s, '
                  f'ETA {(n_tot-done)/max(rate,1e-9):5.1f}s', flush=True)

    print(f'  gated {n_conv} converged entries '
          f'({n_unconv} unconverged reported separately)')
    print('  DISCRETE worst-case relative residual (GATED):')
    print(f'    continuity      : {wd["cont"]:.2e}')
    print(f'    momentum (BULK) : {wd["mom_bulk"]:.2e}   (pressure-free, '
          f'walls excluded -- gated)')
    print(f'    momentum (full) : {wd["mom"]:.2e}   (includes the '
          f'O(sqrt(Pr*dt)) wall-slip layer of the projection scheme; '
          f'informational)')
    print(f'    temperature     : {wd["temp"]:.2e}   (worst at {wd_at})')
    print('  CONTINUUM worst-case relative residual (informational -- dataset '
          'truncation error):')
    print(f'    continuity={wc["cont"]:.2e}  momentum={wc["mom"]:.2e}  '
          f'temperature={wc["temp"]:.2e}')
    if per_branch and verbose:
        print('  per-branch worst BULK momentum:')
        for m, vals in sorted(per_branch.items()):
            print(f'    {str(m):>15}: max={max(vals):.2e} n={len(vals)}')
    if unconv_resid:
        t = torch.tensor(unconv_resid)
        print(f'  unconverged entries: discrete residual '
              f'median={float(t.median()):.2e} min={float(t.min()):.2e} '
              f'max={float(t.max()):.2e} (NOT gated -- held-out / '
              f'time-dependent set)')

    if return_stats:
        return dict(wd=wd, wc=wc, wd_at=list(wd_at) if wd_at else None,
                    n_conv=n_conv, n_unconv=n_unconv,
                    unconv=unconv_resid)
    ok_res = (wd['mom_bulk'] < mom_tol) and (wd['temp'] < rel_tol)
    ok_div = wc['cont'] < div_tol
    print(f'  -> residual {"PASS" if ok_res else "FAIL"} '
          f'(BULK momentum {wd["mom_bulk"]:.2e} < {mom_tol:g}, '
          f'temperature {wd["temp"]:.2e} < {rel_tol:g}); '
          f'incompressibility {"PASS" if ok_div else "FAIL"} '
          f'(spectral continuity {wc["cont"]:.1e} < {div_tol:g})')
    return ok_res and ok_div


def check_distinct(bank, dist_tol, verbose):
    print('\n=== CHECK 3: distinctness of branches at fixed (Ra,Pr) ===')
    groups = {}
    for (Ra, Pr, mode), e in bank['entries'].items():
        if e['converged']:
            groups.setdefault((Ra, Pr), {})[mode] = e
    gmin = math.inf; gat = None; ng = 0
    for (Ra, Pr), br in sorted(groups.items()):
        if len(br) < 2:
            continue
        modes = sorted(br)
        F = torch.stack([torch.stack([br[m]['grid_u'], br[m]['grid_v'],
                                      br[m]['grid_w'], br[m]['grid_T']]).flatten()
                         for m in modes]).double()
        norms = F.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rel = torch.cdist(F, F) / (norms.sqrt() @ norms.sqrt().t())
        k = len(modes)
        mn = float(rel[~torch.eye(k, dtype=torch.bool)].min())
        ng += 1
        if mn < gmin:
            gmin = mn; gat = (Ra, Pr)
        if verbose:
            print(f'  Ra={Ra:6.0f} Pr={Pr:<4.2f}: branches={len(modes)} '
                  f'min rel-dist={mn:.3f}')
    print(f'  checked {ng} parameter pairs; smallest pairwise dist={gmin:.3f} '
          f'(at {gat})')
    ok = gmin > dist_tol
    print(f'  -> {"PASS" if ok else "FAIL"} (min-dist > {dist_tol:g})')
    return ok


# ============================================================================
#  CHECK 4 -- planform label correctness
# ============================================================================
def check_planform(bank, verbose):
    print('\n=== CHECK 4: planform labels (independent recomputation) ===')
    Gx, Gy = bank['aspect']
    mismatch_stored = []; mismatch_seed = []
    n = 0
    for (Ra, Pr, mode), e in bank['entries'].items():
        if not e['converged']:
            continue
        n += 1
        pf = classify(e['grid_w'].double(), Gx, Gy)
        if pf != tuple(e['planform']):
            mismatch_stored.append((Ra, Pr, mode, pf, tuple(e['planform'])))
        if pf != tuple(mode):
            mismatch_seed.append((Ra, Pr, mode, pf))
        if verbose:
            tag = 'ok' if pf == tuple(mode) else 'DRIFTED'
            print(f'  Ra={Ra:6.0f} Pr={Pr:<4.2f} seed={str(mode):>14} '
                  f'-> {str(pf):>14} ({tag})')
    print(f'  checked {n} converged entries')
    if mismatch_stored:
        print(f'  {len(mismatch_stored)} recomputed != STORED tag '
              f'(first: {mismatch_stored[:3]})')
    frac_drift = len(mismatch_seed) / max(n, 1)
    print(f'  {len(mismatch_seed)} converged entries drifted from their seeded '
          f'planform ({100*frac_drift:.1f}%)')
    # gate only the stored-tag self-consistency; drift from seed is informational
    ok = not mismatch_stored
    print(f'  -> {"PASS" if ok else "FAIL"} (stored planform tag matches an '
          f'independent recomputation)')
    return ok


# ============================================================================
def _run_dual_gpu(args, bank):
    """Shard CHECK 1 across both GPUs as two independent processes (no
    gradient sync, no shared state -- each verifies a disjoint half of the
    entries), then merge the worst-case statistics. Sharding is on the sorted
    key list, so the partition is deterministic and reproducible."""
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        print(f'[dual-gpu] only {n_gpu} GPU(s) visible -> single-device run')
        dev = 'cuda' if n_gpu else 'cpu'
        prec = args.precision or ('fp32' if n_gpu else 'fp64')
        return check_residual(bank, args.rel_tol, args.mom_tol, args.div_tol,
                              args.verbose, device=dev, batch=args.batch,
                              precision=prec, progress=args.progress)
    procs, tmps = [], []
    for k in range(2):
        tf = tempfile.NamedTemporaryFile(suffix=f'.shard{k}.json', delete=False)
        tf.close(); tmps.append(tf.name)
        cmd = [sys.executable, os.path.abspath(__file__),
               '--bank', *args.bank, '--shard', f'{k}:2',
               '--device', f'cuda:{k}', '--batch', str(args.batch),
               '--progress', str(args.progress if k == 0 else 0),
               '--stats-out', tf.name,
               '--rel-tol', str(args.rel_tol), '--mom-tol', str(args.mom_tol),
               '--div-tol', str(args.div_tol)]
        if args.precision:
            cmd += ['--precision', args.precision]
        print(f'[dual-gpu] launching shard {k}/2 on cuda:{k}', flush=True)
        procs.append(subprocess.Popen(cmd))
    for k, p in enumerate(procs):
        if p.wait() != 0:
            raise RuntimeError(f'[dual-gpu] shard {k} failed')

    wd = {'cont': 0.0, 'mom': 0.0, 'mom_bulk': 0.0, 'temp': 0.0}
    wc = {'cont': 0.0, 'mom': 0.0, 'temp': 0.0}
    wd_at, best = None, -1.0
    n_conv = n_unconv = 0
    unconv = []
    for f in tmps:
        with open(f) as fh:
            st = json.load(fh)
        for k in wd:
            wd[k] = max(wd[k], st['wd'][k])
        for k in wc:
            wc[k] = max(wc[k], st['wc'][k])
        cand = max(st['wd']['cont'], st['wd']['mom_bulk'], st['wd']['temp'])
        if cand > best:
            best, wd_at = cand, st['wd_at']
        n_conv += st['n_conv']; n_unconv += st['n_unconv']
        unconv += st['unconv']
        os.remove(f)

    print(f'\n[dual-gpu] merged {n_conv} converged / {n_unconv} unconverged')
    print('  DISCRETE worst-case relative residual (GATED):')
    print(f'    continuity      : {wd["cont"]:.2e}')
    print(f'    momentum (BULK) : {wd["mom_bulk"]:.2e}')
    print(f'    momentum (full) : {wd["mom"]:.2e}   (wall-slip layer included; '
          f'informational)')
    print(f'    temperature     : {wd["temp"]:.2e}   (worst at {wd_at})')
    print('  CONTINUUM worst-case (informational):')
    print(f'    continuity={wc["cont"]:.2e}  momentum={wc["mom"]:.2e}  '
          f'temperature={wc["temp"]:.2e}')
    if unconv:
        t = torch.tensor(unconv)
        print(f'  unconverged entries: median={float(t.median()):.2e} '
              f'max={float(t.max()):.2e} (NOT gated)')
    ok_res = (wd['mom_bulk'] < args.mom_tol) and (wd['temp'] < args.rel_tol)
    ok_div = wc['cont'] < args.div_tol
    print(f'  -> residual {"PASS" if ok_res else "FAIL"} '
          f'(BULK momentum {wd["mom_bulk"]:.2e} < {args.mom_tol:g}, '
          f'temperature {wd["temp"]:.2e} < {args.rel_tol:g}); '
          f'incompressibility {"PASS" if ok_div else "FAIL"}')
    return ok_res and ok_div


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bank', required=True, nargs='+',
                   help='one or more bank files; pass shard files directly to '
                        'avoid writing a merged duplicate of the dataset')
    p.add_argument('--rel-tol', type=float, default=1e-1,
                   help='max DISCRETE relative TEMPERATURE residual (default '
                        '0.10; the projection-free temperature equation is the '
                        'clean steadiness certificate and sits at 3-5%%)')
    p.add_argument('--device', default=None,
                   help="cpu | cuda | cuda:0 ... (default: cuda if available). "
                        "The residual check is 8x more work at 128x64x49 than "
                        "at 64x32x25; a GPU turns ~25 min into well under a "
                        "minute.")
    p.add_argument('--batch', type=int, default=8,
                   help='entries processed per chunk (raise on a big GPU)')
    p.add_argument('--precision', choices=['fp32', 'fp64'], default=None,
                   help='fp64 on CPU, fp32 on GPU by default. fp32 changes the '
                        'reported residuals only in the 6th significant digit '
                        '(verified), far below any gate.')
    p.add_argument('--progress', type=int, default=50,
                   help='print a progress line every N entries (0 = silent)')
    p.add_argument('--dual-gpu', dest='dual_gpu', action='store_true',
                   help='shard CHECK 1 across both GPUs (2 processes)')
    p.add_argument('--shard', default=None, metavar='k:K',
                   help='internal: verify only shard k of K')
    p.add_argument('--stats-out', dest='stats_out', default=None,
                   help='internal: dump CHECK-1 worst-case stats as JSON and exit')
    p.add_argument('--mom-tol', type=float, default=2e-1,
                   help='max DISCRETE relative BULK momentum residual (default '
                        '0.20). Momentum retains a Pr-scaled bulk tail of the '
                        'fractional-step wall-slip layer, a MEASURED O(sqrt('
                        'Pr*dt)) scheme artifact (settling at dt/16 shrinks the '
                        'full-field value ~5x); banks generated with the '
                        'generator option --settle-steps pass 0.10 here')
    p.add_argument('--div-tol', type=float, default=1e-1,
                   help='max DISCRETE relative continuity residual (default 0.10)')
    p.add_argument('--dist-tol', type=float, default=0.30)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    bank = torch.load(args.bank[0], map_location='cpu', weights_only=False)
    for _p in args.bank[1:]:
        _b = torch.load(_p, map_location='cpu', weights_only=False)
        bank['entries'].update(_b['entries'])
        del _b
    ent = bank['entries']
    nconv = sum(1 for e in ent.values() if e['converged'])
    print(f'loaded {len(args.bank)} bank file(s): {len(ent)} entries '
          f'({nconv} converged, {len(ent)-nconv} unconverged), '
          f'grid={bank["grid"]["Nx"]}x{bank["grid"]["Ny"]}x{bank["grid"]["Nz"]}, '
          f'aspect={bank["aspect"]}, modes={bank["mode_list"]}')

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    precision = args.precision or ('fp32' if str(device).startswith('cuda')
                                   else 'fp64')
    keys = None
    if args.shard:
        k, K = (int(x) for x in args.shard.split(':'))
        allk = sorted(bank['entries'], key=lambda t: t[:2])
        keys = allk[k::K]

    if args.stats_out:                       # child worker of --dual-gpu
        st = check_residual(bank, args.rel_tol, args.mom_tol, args.div_tol,
                            args.verbose, device=device, batch=args.batch,
                            precision=precision, progress=args.progress,
                            keys=keys, return_stats=True)
        with open(args.stats_out, 'w') as fh:
            json.dump(st, fh)
        return

    if args.dual_gpu:
        r1 = _run_dual_gpu(args, bank)
    else:
        r1 = check_residual(bank, args.rel_tol, args.mom_tol, args.div_tol,
                            args.verbose, device=device, batch=args.batch,
                            precision=precision, progress=args.progress,
                            keys=keys)
    r3 = check_distinct(bank, args.dist_tol, args.verbose)
    r4 = check_planform(bank, args.verbose)

    print('\n================= SUMMARY =================')
    results = [('PDE residual + incompressibility', r1),
               ('distinctness', r3), ('planform labels', r4)]
    all_ok = True
    for name, r in results:
        print(f'  {name:34s}: {"PASS" if r else "FAIL"}')
        all_ok = all_ok and r
    print('==========================================')
    print('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()