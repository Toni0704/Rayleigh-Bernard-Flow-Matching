#!/usr/bin/env python3
"""
rb3d_classify.py
================================================================================
Harmonic-aware, purity-gated planform classification for 3D Rayleigh-Benard.

WHY THIS EXISTS
--------------------------------------------------------------------------------
The original classifier had two defects that make any "the model produced a
planform outside the training set" claim untrustworthy:

  (1) NO HARMONIC AWARENESS.  A finite-amplitude roll is nonlinear: w contains
      the fundamental k and its harmonics 2k, 3k, ...  The old code took the
      two largest spectral peaks and, if the second exceeded half the first,
      declared a two-mode planform.  For a roll(3,0) with a strong 2nd harmonic
      at (6,0) it returned ('rect', 6, 0) -- a label that looks like a novel
      mode but is just one roll with its own harmonic.  (This is precisely the
      rect(6,0) seen in evaluation.)  Here, before a second peak is admitted as
      an independent fundamental, we check whether it is COLLINEAR and an
      INTEGER MULTIPLE of the first; if so it belongs to the first mode's
      harmonic family and is folded into it.

  (2) NO ABSTENTION.  argmax always returns something, so a blended or noisy
      off-manifold field always got a confident label.  We now report a
      spectral PURITY -- the fraction of in-band energy explained by the
      reported mode families -- and return ('unknown', 0, 0) when purity falls
      below a threshold CALIBRATED ON THE TRAINING BANK (default: the 5th
      percentile of the training branches' own purity).  "Unknown" means "less
      spectrally coherent than 95% of real training branches".

Together these turn a forced guess into a decision with a stated confidence,
which is the minimum required to claim a planform is genuinely off-manifold.

WHAT A NOVELTY CLAIM STILL REQUIRES (see novelty_audit.py)
--------------------------------------------------------------------------------
A high-purity label outside the training set is NECESSARY, not SUFFICIENT.
Passing a residual tolerance does not make a field a PDE solution: the discrete
operator admits solutions the continuum does not, and a coarse grid + loose
tolerance is a weak filter.  The full evidence chain is
    purity gate -> strict fp64 residual -> distinctness under the symmetry
    group -> persistence under grid refinement -> independent solver seeding.
"""

import math

import torch


# ---------------------------------------------------------------------------
def _spectrum(w, kmax):
    """Horizontal power spectrum of w, summed over z (more robust than a single
    mid-plane slice), restricted to the resolved band |k| <= kmax."""
    Nx, Ny, Nz = w.shape
    W = torch.fft.rfft2(w, dim=(0, 1))                 # (Nx, Ky, Nz)
    P = (W.abs() ** 2).sum(dim=2)                      # (Nx, Ky)
    P[0, 0] = 0.0                                      # drop the mean
    Ky = P.shape[1]
    band = torch.zeros_like(P)
    for n in range(Nx):
        ns = n if n <= Nx // 2 else n - Nx
        if abs(ns) <= kmax:
            band[n, :min(kmax + 1, Ky)] = P[n, :min(kmax + 1, Ky)]
    return band, Nx, Ky


def _signed(n, Nx):
    return n if n <= Nx // 2 else n - Nx


def _family_indices(n, m, Nx, Ky, n_harm=4, rad=0):
    """Index set of the mode (n,m) and its harmonics j*(n,m), j=1..n_harm,
    folded onto the rfft grid and including the conjugate partner (-n, m).
    rad=0 by default: on a periodic grid the integer modes are exact deltas, so
    a +-1 window would merge ADJACENT fundamentals (k=3 would swallow k=4) and
    report a blended two-roll state as a pure roll with purity 1.0."""
    idx = set()
    for j in range(1, n_harm + 1):
        for sgn in (1, -1):
            nn, mm = sgn * j * n, j * m
            if abs(mm) > Ky - 1:
                continue
            for dn in range(-rad, rad + 1):
                for dm in range(-rad, rad + 1):
                    a = (nn + dn) % Nx
                    b = mm + dm
                    if 0 <= b <= Ky - 1:
                        idx.add((a, b))
    return idx


def _is_harmonic_of(n2, m2, n1, m1, tol=1e-9):
    """True if (n2,m2) is collinear with (n1,m1) and an integer multiple of it.
    This is what distinguishes a roll's own harmonic from a second fundamental.
    """
    if n1 == 0 and m1 == 0:
        return False
    # collinearity: cross product zero
    if abs(n1 * m2 - n2 * m1) > tol:
        return False
    # integer multiple (>=2; j=1 is the fundamental itself)
    if n1 != 0:
        j = n2 / n1
    else:
        j = m2 / m1
    return abs(j - round(j)) < 1e-6 and abs(round(j)) >= 2


def classify_purity(w, Gx, Gy, kmax=8, ratio=0.5, purity_min=None,
                    n_harm=4):
    """Return (label, purity, info).

    label   : ('roll', n, 0) | ('roll', 0, m) | ('rect', n, m) |
              ('square', n, m) | ('unknown', 0, 0)
    purity  : fraction of in-band spectral energy captured by the reported
              mode family/families (harmonics included).
    info    : dict with the fundamentals found and their energies.

    If purity_min is given and purity < purity_min, the label is downgraded to
    ('unknown', 0, 0) -- the classifier ABSTAINS rather than forcing a guess.
    """
    P, Nx, Ky = _spectrum(w.double(), kmax)
    total = float(P.sum().clamp_min(1e-30))

    # --- first fundamental -------------------------------------------------
    i1 = int(P.reshape(-1).argmax())
    n1, m1 = i1 // Ky, i1 % Ky
    n1s = _signed(n1, Nx)
    fam1 = _family_indices(n1s, m1, Nx, Ky, n_harm)
    E1 = sum(float(P[i, j]) for i, j in fam1)

    # --- look for a SECOND FUNDAMENTAL (not a harmonic of the first) --------
    Q = P.clone()
    for i, j in fam1:
        Q[i, j] = 0.0
    n2s = m2 = None
    E2 = 0.0
    while True:
        i2 = int(Q.reshape(-1).argmax())
        a2 = float(Q.reshape(-1)[i2])
        if a2 <= 0:
            break
        c2, d2 = i2 // Ky, i2 % Ky
        c2s = _signed(c2, Nx)
        if _is_harmonic_of(c2s, d2, n1s, m1):
            # a harmonic that leaked outside the family window: absorb it
            for i, j in _family_indices(c2s, d2, Nx, Ky, 1):
                Q[i, j] = 0.0
            continue
        # is it a genuine, comparable second fundamental?
        peak1 = float(P[n1, m1])
        if a2 < ratio ** 2 * peak1:            # compare POWER (ratio is on amp)
            break
        n2s, m2 = c2s, d2
        fam2 = _family_indices(n2s, m2, Nx, Ky, n_harm)
        # union, not sum: harmonic families can share indices (e.g. the 2nd
        # harmonic of k=3 and the 3rd of k=2 both land on k=6), and double
        # counting would push purity above 1.
        E2 = sum(float(P[i, j]) for i, j in (fam2 - fam1))
        break

    N1, M1 = abs(n1s), abs(m1)
    if n2s is None:                                    # single fundamental
        purity = E1 / total
        if M1 == 0:
            label = ('roll', N1, 0)
        elif N1 == 0:
            label = ('roll', 0, M1)
        else:
            label = ('oblique', N1, M1)
        info = dict(fundamentals=[(N1, M1)], E=[E1 / total])
    else:
        N2, M2 = abs(n2s), abs(m2)
        purity = (E1 + E2) / total
        x1, y1 = (M1 == 0), (N1 == 0)
        x2, y2 = (M2 == 0), (N2 == 0)
        if x1 and y2:                                  # x-roll + y-roll
            label = ('square', N1, M2) if N1 == M2 else ('rect', N1, M2)
        elif y1 and x2:                                # y-roll + x-roll
            label = ('square', N2, M1) if N2 == M1 else ('rect', N2, M1)
        elif (x1 and x2) or (y1 and y2):
            # two fundamentals on the SAME axis: a superposition of two rolls
            # of different wavenumber. This is NOT a standard planform and is
            # generically NOT a steady solution -- flag it, never silently
            # relabel it as a higher-k roll (the old code's rect(6,0) failure).
            a, b = sorted((max(N1, M1), max(N2, M2)))
            label = ('mixed', a, b)
        else:
            label = ('oblique', max(N1, N2), max(M1, M2))
        info = dict(fundamentals=[(N1, M1), (N2, M2)],
                    E=[E1 / total, E2 / total])

    info['purity'] = purity
    if purity_min is not None and purity < purity_min:
        return ('unknown', 0, 0), purity, info
    return label, purity, info


# ---------------------------------------------------------------------------
def calibrate_purity(bank, quantile=0.05, kmax=8):
    """Purity threshold from the TRAINING bank: the `quantile` of the real
    branches' own purity. A candidate less coherent than 95% of genuine
    training branches is not trustworthy enough to label."""
    z = torch.linspace(0, 1, bank['grid']['Nz'])
    vals, per_mode = [], {}
    for (Ra, Pr, mode), e in bank['entries'].items():
        if not e.get('converged', True):
            continue
        _, pur, _ = classify_purity(e['grid_w'].float(), *bank['aspect'],
                                    kmax=kmax)
        vals.append(pur)
        per_mode.setdefault(tuple(mode), []).append(pur)
    t = torch.tensor(vals)
    thr = float(torch.quantile(t, quantile))
    return thr, {m: (float(torch.tensor(v).mean()), len(v))
                 for m, v in per_mode.items()}


def harmonic_report(w, Gx, Gy, kmax=8, n_harm=4):
    """Diagnose whether a field's spectrum is a fundamental + harmonics (one
    roll) or genuinely multi-modal. Prints the energy in each harmonic of the
    dominant wavevector."""
    P, Nx, Ky = _spectrum(w.double(), kmax)
    total = float(P.sum().clamp_min(1e-30))
    i1 = int(P.reshape(-1).argmax())
    n1, m1 = i1 // Ky, i1 % Ky
    n1s = _signed(n1, Nx)
    rows = []
    for j in range(1, n_harm + 1):
        idx = _family_indices(j * n1s, j * m1, Nx, Ky, n_harm=1)
        E = sum(float(P[i, k]) for i, k in idx)
        rows.append((j, (j * n1s, j * m1), E / total))
    return dict(fundamental=(n1s, m1), harmonics=rows, total=total)
