# Evaluation & Paper-Update Spec — Multimodal PCFM

This document compiles every decision we reached about **what to measure**, **how to measure it**, and **how to write it up**, into one self-contained reference. It has two independent instruction sets:

- **Part A — for the code chat:** exactly what to evaluate, for every model/sampling configuration, so the numbers are directly comparable across all PDEs.
- **Part B — for the paper chat:** exactly what to change in `manual.tex`, with anchor phrases (the LaTeX already contains most of this; it mainly needs generalizing and one real fix).

The single governing principle behind all of it: **every reported number must mean the same thing in every PDE.** A metric that is comparable across Bratu, Allen–Cahn, and Rayleigh–Bénard is worth more than a metric that is individually "better" but system-specific.

---

## 0. The configurations being compared

There are two axes. **Method** (how physics is enforced) and **conditioning** (whether the parameter is an input).

| Config | Trained model | Sampling-time projection | Physics enforced… |
|---|---|---|---|
| **vanilla** (cond) | plain flow-matching | **none** | nowhere (baseline) |
| **PBFM** (cond) | flow-matching **+ soft physics loss** | **none** | at training time (soft) |
| **PCFM** (cond) | plain flow-matching (same net as vanilla) | **yes** (final projection) | at sampling time (hard) |
| **PCFM** (uncond) | plain flow-matching, no parameter input | **yes** (final projection) | at sampling time (hard) |

Key point for whoever writes the eval: **vanilla and PBFM are sampled identically (no projection); they differ only in the trained checkpoint.** PCFM reuses the *same* plain checkpoint as vanilla and adds a projection step. Conditioning just controls whether the parameter (λ / ε / Ra) is fed to the network.

The **same metric suite** (Section A.4–A.8) is run on all four. That identical treatment is the whole point — it is what makes "PCFM valid% vs PBFM valid%" a fair sentence.

---

# PART A — FOR THE CODE CHAT: what to evaluate

## A.1 Sampling scheme (how to draw the K samples for each config)

For each parameter value in a sweep (see A.2), draw `K` samples (use `K ≥ 40`; `K=200` for the coverage/entropy plots):

- **vanilla / PBFM:** integrate the learned flow from noise to `t=1`, **no projection** (`project=False`).
- **PCFM:** integrate the flow to `t=1`, then apply **one** robust final projection onto the discrete PDE residual (`project=True, per_step=False`).

**Two hard rules for PCFM sampling — both were regressions we already hit:**

1. **Free-flow, then a single final projection. Never per-step projection from small `t`.** Per-step projection from early `t` drags every trajectory onto whichever branch is nearest the noise draw and *erases the multimodality*. (This is the `proj_start=0.0, per_step=True` bug that collapsed Bratu 2D to one branch.)
2. **Genuine per-sample projection. No reference cache / no homotopy snapping.** A cache keyed by `(param, branch)` that returns a precomputed reference field is not a projection — it makes NRMSE artificially ~`1e-6` and routes samples to a single branch. Disable it (`USE_PROJ_CACHE=False`). Each sample must be projected from its own free-flowed field.

Classify each sample's branch **after** projection, from the field itself (A.6).

## A.2 Parameter sweep

Evaluate strictly **inside the multi-solution regime** so all branches genuinely exist:

- **Bratu:** `λ ∈ [2.0, λ_c − 0.1]` (below the fold; both lower/upper branches exist).
- **Allen–Cahn:** `ε ∈ [ε_lo, ~0.022]` with `ε_lo = 0.007` (kept just above the mode-4 pitchfork so the stored set is complete: 7 branches).
- **Rayleigh–Bénard:** the Ra range already used for RB2D/RB3D.

Use ~6–16 parameter points; report metrics pooled over the sweep and, where useful, per-parameter.

## A.3 References — **the single most important correctness rule**

The validity floor (A.5) is calibrated from reference residuals, so the references must be generated **independently of the residual evaluator**. Concretely:

> **A reference must never be produced by driving down the very discrete residual you then evaluate on it.**

If you Newton-solve a Bratu field on the *same* `N×N` operator that `ρ` later evaluates, its residual is machine epsilon (~`1e-14`) — that is *solver tolerance*, not physics — and the calibrated floor degenerates to zero, so **nothing** can ever be classified valid. This is exactly why Bratu 2D reported 0% valid.

Correct recipes (all give a genuine, comparable discretization floor of order `1e-3`–`1e-6`):

- **Finer-grid solve, restricted down:** solve on a `(2N−1)×(2N−1)` grid, then take every other point onto the working `N`-grid. (Bratu, RB.)
- **Continuum solve, interpolated:** solve the BVP/ODE with an adaptive continuum solver, then interpolate onto the working grid. (Allen–Cahn already does this — which is precisely why AC's floor is a healthy `~7e-7` and AC validity works.)

The rule is uniform: *reference = an independently obtained, higher-fidelity solution represented on the working grid.* Apply it to **all** PDEs. Once it holds everywhere, **no `max()`/`τ` floor-guard is needed or wanted** — a guard would make the threshold mean two different things in two different PDEs, which breaks comparability.

## A.4 Relative physics residual `ρ` (the headline, comparable number)

For every PDE, per field `u`:

```
r_j(u) = rms(R_j(u)) / rms(D_j(u))       # rms over INTERIOR grid points
ρ(u)   = max_j r_j(u)                    # worst-of-equations aggregate
physL2(u) = sqrt( mean_j r_j(u)^2 )      # root-mean aggregate (secondary)
```

`R_j` is the residual of governing equation `j`; `D_j` is its **dominant balancing term**. Normalizing by `D_j` makes `ρ` dimensionless and therefore comparable across PDEs. Per system:

| PDE | Residual `R` | Dominant term `D` | `m` |
|---|---|---|---|
| Bratu | `Δu + λ e^u` | `λ e^u` | 1 |
| Allen–Cahn | `−ε u'' + (u³ − u)` | `u³ − u` (reaction) | 1 |
| Rayleigh–Bénard | steady vorticity residual | buoyancy `Pr·Ra·∂ₓT` | 1 (report physL2 too) |

For single-equation problems (Bratu, AC) `ρ = physL2 = r_1`. **Report `ρ` for every PDE** (not an absolute residual) so both the residual column and the validity column are comparable.

**Degenerate branch (Allen–Cahn trivial `u ≡ 0`):** `R ≡ 0` and `D ≡ 0`, so `ρ` is `0/0`. Define `ρ := 0` for the trivial branch — it solves the PDE exactly and is valid by construction. Do **not** score it with a range-normalized NRMSE (that produces the spurious "trivial valid 0").

## A.5 Floor, threshold, validity (identical formula everywhere)

Per branch `n`, from the **independent** references of A.3:

```
floor_n = median{ ρ(f) : f in reference branch n }
θ_n     = 1.5 · quantile_0.99{ ρ(f) : f in reference branch n }
valid(u) ⟺ ρ(u) < θ_{n(u)}
validity = 100 · (# valid samples) / K
```

Also report `ρ / floor_n` (interpretation: `~1` = as consistent as the reference is on the grid; `≫1` = off-manifold; `≪1` = discretely more consistent than the sampled reference, which is normal for a projected PCFM field).

**No absolute cutoff, no `max(·, τ)`.** The calibration from properly-generated references is the only bar. This is the branch-free, single-meaning criterion.

**Robustness note to expect and report:** vanilla/PBFM residuals sit `~10⁵` above PCFM residuals, so the valid/invalid verdict is insensitive to the exact quantile or inflation constant — the criterion is not a tuned knob.

## A.6 Branch classification `n(u)` (read from the field)

| PDE | Rule |
|---|---|
| Bratu | `lower` if `u.max() < α_fold` else `upper` (α_fold = center value at the fold) |
| Allen–Cahn | `trivial` if `|u|.max() < tol`; else mode `n =` (#interior sign changes)+1, sign from the projection onto `sin(nπx)` → e.g. `twobump_pos` |
| Rayleigh–Bénard | roll number / planform from the dominant horizontal wavenumber |

## A.7 Coverage and diversity (entropy in **nats**)

Over the **valid** samples:

```
coverage = |distinct branches recovered| / N
H = − Σ_k p_k · ln(p_k)          # NATS (natural log), max = ln(N)
```

`N` = number of coexisting branches: **Bratu N=2, RB N=5, Allen–Cahn N=2M+1** (odd; `M` active modes; **7** on the AC band: trivial, ±u₁, ±u₂, ±u₃).

**Use `ln`, not `log2`.** The paper reports nats; a `log2` (bits) entropy is a factor `1/ln2 ≈ 1.443` too large and will not match `ln N`.

## A.8 Symmetry-aware fidelity NRMSE (secondary)

```
g* = argmin_{σ∈G} ‖σ·g − r‖₂ ,   NRMSE = ‖g* − r‖₂ / ‖r‖₂
```

Align over the PDE's symmetry group `G` before measuring error:

| PDE | Symmetry group `G` |
|---|---|
| Rayleigh–Bénard | horizontal translation (via FFT cross-correlation) + reflection |
| Bratu | identity only (Dirichlet walls fix the phase; branches are not sign-related) |
| Allen–Cahn | reflection `u → −u` only (Dirichlet ⇒ no translation) |

NRMSE is a *fidelity* diagnostic, reported alongside — it is **not** the validity criterion.

## A.9 Output table (one schema, all configs, all PDEs)

Produce one row per configuration with these columns:

```
model(cond/uncond) | method(vanilla/PBFM/PCFM) | ρ (median) | ρ/floor | valid% | coverage | H(nats) | NRMSE%
```

Minimum rows requested: **cond-PCFM, uncond-PCFM, cond-vanilla, cond-PBFM** (add uncond-vanilla / uncond-PBFM if cheap — it strengthens the conditioning ablation). Save to CSV per PDE.

## A.10 Plots (same three for every PDE)

1. **Branch-coverage histogram** at a fixed parameter (counts per branch, colored by branch), for vanilla vs PCFM.
2. **Solution-family reconstruction:** samples' branch amplitude vs parameter, colored by branch, over the reference curves (the bifurcation diagram for Bratu/AC; planform map for RB).
3. **Residual distribution:** histogram / box of `ρ` for vanilla vs PBFM vs PCFM, with the floor `θ_n` marked.

## A.11 Expected outcomes (sanity checks)

- **vanilla, PBFM:** `ρ` far above the floor (`ρ/floor ≫ 1`), low validity. PBFM should beat vanilla but stay well above the floor (soft training-time physics cannot enforce the constraint exactly).
- **PCFM:** `ρ` at/below the floor (`ρ/floor ≲ 1`), validity near 100%, all branches covered, entropy near `ln N`.
- **cond vs uncond PCFM:** both should be valid and cover branches; conditioning mainly sharpens fidelity, not validity (the projection fixes the parameter regardless).
- **Allen–Cahn is the reference implementation** — it already produces exactly this (uncond PCFM 96.9% / cond PCFM 99.7% valid, 7 branches, ~1.6–1.8 nats, vanilla 0%). Match its structure for the others.

---

# PART B — FOR THE PAPER CHAT: what to write in `manual.tex`

The metrics section (`\label{sec:metrics}`) already defines `ρ` (`eq:rho`), the floor (`eq:floor`), the threshold (`eq:threshold`), validity (`eq:valid`), coverage (`eq:coverage`), and symmetry-NRMSE (`eq:nrmse`) — and they are already the uniform, nats-based, calibrated-floor scheme we want. The edits below **generalize** them from RB-centric wording to all PDEs, add the one genuinely missing idea (reference independence), and fix the branch count. Each edit gives a find-anchor and the replacement.

## B.1 Add the reference-independence requirement (the one real gap)

**Where:** the `\subsection{Validity needs a calibrated, floor-relative criterion}` (`\label{sec:floor}`).
**Find** the sentence beginning `The relaxation projector cannot push a sample below the residual that the reference solver itself attains on the grid, and that floor is not zero.`
**Append after it:**
```latex
Two conditions make this floor a meaningful, comparable quantity rather than an artefact.
First, the reference must be obtained \emph{independently of the residual evaluator}---a
higher-fidelity solve (a finer grid restricted to the working grid, or a continuum solution
interpolated onto it)---never a field produced by driving down the very discrete residual we
then report on it, which would collapse the floor to solver tolerance ($\sim$machine epsilon)
and make every generated sample fail. Second, the residual is measured relatively
(Eq.~\eqref{eq:rho}), so the floor reflects genuine discretization error in units comparable
across systems. Under these two conditions the same calibrated bar is applied unchanged to every
PDE; no system-specific absolute cutoff is introduced.
```
**Why:** this is the fix for the Bratu 0%-valid failure and the justification for using one branch-free criterion everywhere. It also pre-empts the reviewer question "why is the floor nonzero for Bratu, which has no near-cancellation?" — because the reference is an independent finer-grid solution, not a same-grid Newton solve.

## B.2 Generalize the dominant term `D` beyond RB

**Where:** `\paragraph{Relative physics residual.}` in `sec:metrics`.
**Find** `the dominant balancing term of equation $j$ (for the vorticity equation, the buoyancy term $\mathrm{Pr}\,\mathrm{Ra}\,\partial_x T$).`
**Replace with:**
```latex
the dominant balancing term of equation $j$: the buoyancy term $\mathrm{Pr}\,\mathrm{Ra}\,\partial_x T$
for the vorticity equation, the reaction term $\lambda e^{u}$ for Bratu, and the double-well
reaction $u^3-u$ for Allen--Cahn.
```

## B.3 Handle the degenerate (trivial) branch

**Where:** end of `\paragraph{Relative physics residual.}` (after the `physL2` discussion).
**Find** `For a single-equation problem such as Bratu, $m=1$ and $\rho$ reduces to the normalised residual $\|\mathcal{R}(u)\|$.`
**Append after it:**
```latex
When a branch is the trivial state $u\equiv 0$ (Allen--Cahn), both $\mathcal{R}$ and $\mathcal{D}$
vanish identically; we set $\rho:=0$ there, since $u\equiv0$ solves the equation exactly and is
valid by construction, and we do not score it with the range-normalised fidelity of
Eq.~\eqref{eq:nrmse}.
```

## B.4 Fix the branch count `N` and generalize `n(u)`

**Where:** `\paragraph{Coverage and diversity.}`.
**Find** `let $n(u)$ be the mode (roll number / planform) read from the dominant horizontal wavenumber of $u$, let $N$ be the number of coexisting branches ($N=2$ for Bratu, $N=5$ for RB),`
**Replace with:**
```latex
let $n(u)$ be the branch label read from the field itself---the roll number/planform from the
dominant horizontal wavenumber (RB), the peak amplitude relative to the fold value (Bratu), or the
interior sign-change count and sign (Allen--Cahn)---let $N$ be the number of coexisting branches
($N=2$ for Bratu, $N=5$ for RB, and $N=2M+1$ for Allen--Cahn with $M$ active modes, i.e.\ seven on
our band),
```

## B.5 Generalize the NRMSE symmetry group per PDE

**Where:** end of `\paragraph{Symmetry-aware fidelity.}`.
**Find** `where the optimal translation is obtained in one FFT cross-correlation and the two reflections are enumerated (identical fields under a random shift/reflection align to $0\%$).`
**Append after it:**
```latex
 The group $G$ is system-specific: horizontal translation (one FFT cross-correlation) plus
reflection for Rayleigh--B\'enard, the reflection $u\mapsto-u$ alone for Allen--Cahn (Dirichlet
walls leave no translation), and the identity for Bratu (Dirichlet walls fix the phase and its
branches are not sign-related).
```

## B.6 State that PBFM is sampled without projection (method clarity)

**Where:** in `\subsection{Vanilla vs.\ PBFM vs.\ PCFM}` (results) or the PBFM subsection (`sec:pbfm`), wherever the three methods are first contrasted operationally.
**Insert a sentence:**
```latex
At evaluation, vanilla and PBFM are sampled identically---free flow with no sampling-time
projection---and differ only in the trained velocity field (PBFM adds the soft physics penalty of
Eq.~\eqref{eq:pbfmloss} during training); PCFM reuses the plain velocity field and adds a single
final projection. All three are scored by the identical metric suite of Section~\ref{sec:metrics}.
```

## B.7 Results table schema

Make the results table(s) carry, for each system, the four configurations as rows with the columns of A.9: `ρ (median)`, `ρ/floor`, `valid%`, `coverage`, `H (nats)`, `NRMSE%`. Ensure the caption states that **every column is dimensionless and computed identically across PDEs**, that entropy is in **nats** (max `ln N`), and that references are independently obtained (cross-ref `sec:floor`). Fill the Allen–Cahn row from the working run; fill Bratu once the reference-generation fix (B.1 / Part A.3) is in and its floor is no longer degenerate.

## B.8 Carry-over narrative edits (already agreed, keep them)

These are not metric edits but must stay consistent with the above:
- Allen–Cahn benchmark paragraph reframed as a **symmetry-breaking pitchfork cascade** (Chafee–Infante), with the metastability/energy-landscape nod kept (abstract unchanged), the corrected equation `−ε u'' + u(u²−1)=0` on `[0,1]`, and the **seven** sign-split branches.
- The three "metastable equilibria" → "distinct/pitchfork-generated equilibria" softenings.
- "one benchmark" → "two benchmarks" for the unstable-branch recovery, and the added Allen–Cahn "access achieved" paragraph in the unstable-branch subsection.

---

## One-line summary

**Report a relative residual `ρ` for every PDE, calibrate a per-branch floor from independently-generated references, call a sample valid iff `ρ < 1.5·q₀.₉₉` of that floor, and measure coverage/entropy (nats) and symmetry-aligned NRMSE — the identical pipeline for vanilla, PBFM, and PCFM, conditioned or not.** The only substantive fix left is generating Bratu's references independently (finer grid restricted) so its floor stops collapsing to machine zero; Allen–Cahn already does this and is the template.
