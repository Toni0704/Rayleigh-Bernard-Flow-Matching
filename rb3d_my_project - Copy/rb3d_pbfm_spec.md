# RB3D PBFM — Design Spec

Add 3-D PBFM next to the existing `rb3d_pcfm_*` files in this folder (physics
enforced at training time via a residual loss, vs. PCFM's sampling-time
projection). Fill in the open items, then we start coding.

## Decided

1. **Architecture: import, don't copy.** `rb3d_pbfm_common.py` imports `FNO3d`
   / `RB3DData` from `rb3d_pcfm_common.py` (plus batched diff-ops from
   `generate_rb3d_multisolution.py` / `verify_rb3d_multisolution.py`) instead
   of a verbatim duplicate. Guarantees PCFM and PBFM share one architecture
   (your requirement) and means the PCFM plain-FM checkpoint doubles as the
   "vanilla" baseline — no separate vanilla run needed.
2. **Dual-T4 training.** Port the 2-D pipeline's multi-GPU scheme: no DDP
   wrapper (DDP's autograd hooks conflict with ConFIG needing two separate
   backward passes), instead each rank holds a full replica, gradients are
   flattened and reduced with one `all_reduce` per loss. `--batch` is the
   global batch (paper convention); each rank takes `batch/world_size` with a
   different data slice. Works under `torchrun` or standalone `mp.spawn`.
3. **Checkpointing.** Every `eval_every` iters: overwrite a `_resume.pt` (full
   optimizer/scheduler/EMA/iter state, so a timeout loses at most one interval)
   and, only on improvement, the best EMA-only checkpoint. Auto-resume on
   startup unless `--no-resume`. Tag with an `ARCH_VERSION`; a mismatched
   resume file is archived (`.stale`) and training restarts clean instead of
   crashing. Only rank 0 writes; the found/not-found decision is broadcast so
   ranks don't diverge.
4. **Evaluation.** No new evaluator — extend `evaluate_rb3d.py` with PBFM as a
   third method column (vanilla/PBFM/PCFM), following
   `evaluation_and_paper_spec.md` Part A exactly (same `ρ`/floor/valid%/
   coverage/entropy/NRMSE columns, PBFM sampled with `project=False`, no
   relaxation). **Flag to check later:** `calibrate_thresholds`'s reference
   fields come from the training bank itself, scored with operators that
   partly overlap the generator's own convergence operator — the same
   mismatch that caused Bratu's 0%-valid bug (spec §A.3 requires independent
   references). Worth a sanity check on the calibrated floor before trusting
   valid% numbers; not blocking training code.

## Open — please fill in

- **Conditioning:** cond-only (like 2-D PBFM, since PBFM has no sampling-time
  handle on Ra/Pr) — confirm, or want uncond too? `___`
- **Residual loss (the one genuinely new piece):** 3-D has no streamfunction,
  so the training-time residual must work on primitive `(u,v,w,T')` with a
  batched, differentiable Leray projection for momentum (adapting
  `evaluate_rb3d.py::Residual`, currently unbatched/float64/eval-only).
  - Aggregate for the loss: sum-of-squares of (momentum, temp, continuity)
    like RB2D, weighted by `t^p`? `___`
  - Match `evaluate_rb3d.py::Residual`'s operators exactly (so train-time loss
    == eval-time scorer)? `___`
- **Unroll depth / backprop mode:** 2-D goes up to unroll=4, `backprop='last'`.
  3-D's Leray solve is much costlier per call than 2-D's Poisson solve — start
  smaller (unroll=2, no curriculum) and grow only if needed? `___`
- **Sampling ablations:** keep `RB3DRelaxer` wired in as an optional
  `--project` ablation (headline path stays plain ODE, no relaxation)? Add a
  cheap non-iterative cleanup (single `leray_project`, no `imex_step`s), like
  2-D's `hard_cleanup`? `___`
- **Batch size / hidden / grid:** PCFM defaults to `batch=12, hidden=32` at
  128×64×49; PBFM's unroll + differentiable residual costs more per step. Size
  empirically once code runs, or do you already have a target? `___`
- **Hyperparameters** (lr, iters, warmup_iters, patience, early-stop metric
  `fm` vs `gen_res`): carry 2-D's defaults, or set your own? `___`
- **Filenames:** `rb3d_pbfm_common.py`, `train_rb3d_pbfm_conditioned.py`,
  `ckpt_rb3d_pbfm_cond.pt` — fine, or rename? `___`
- Anything else: `___`

## My read

Everything above "Open" is settled; the residual-loss definition (§Open,
2nd bullet) is the only structurally new code — the rest is a port of
`train_rb2d_pbfm_conditioned.py`'s recipe. Biggest real risk is compute cost:
a differentiable batched Leray projection inside an unrolled loop at
128×64×49 is much heavier than 2-D's FFT/FD Poisson solve, so unroll depth and
batch size will likely need empirical tuning even with both GPUs in use.
