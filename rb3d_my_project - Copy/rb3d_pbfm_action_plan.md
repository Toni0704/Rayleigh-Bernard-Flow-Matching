# RB3D PBFM — Action Plan

Defaults locked in for the items left open in `rb3d_pbfm_spec.md` (no reply
came back, so proceeding with the recommended option from each bullet there —
revisit any of these anytime by editing this file):

- Conditioning: **cond-only**.
- Residual loss: sum-of-squares of (Leray-projected momentum, temperature,
  continuity), `t^p`-weighted, operators matched to `evaluate_rb3d.py::Residual`
  exactly (train-time loss == eval-time scorer).
- Unroll: start at **unroll=2, `backprop='last'`, no curriculum**; revisit if
  residual loss needs more depth.
- Sampling: headline = plain ODE, no relaxation. Keep `RB3DRelaxer` as an
  opt-in `--project` ablation; add a single non-iterative `leray_project`
  cleanup as `--hard-cleanup`.
- Batch/hidden: start from PCFM's `batch=12, hidden=32`; in practice OOM'd at
  batch=12 (6/GPU) once physics loss turned on (ConFIG's two separate backward
  passes briefly hold both loss graphs at once) — running at **batch=6** (3/GPU)
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Hyperparameters: carry over 2-D PBFM's defaults (AdamW wd=0, β=(0.5,0.999),
  fixed LR 3e-4, warmup_iters=1000, ema_decay=0.999, patience=8×eval_every,
  **early-stop metric=`fm`** — corrected from an earlier `gen_res` note here,
  which was a misreading of 2-D's actual default; `gen_res` is offered as an
  option but `fm` is what 2-D actually defaults to and what 3-D now matches)
  unless training reveals a reason to change them.
- **EMA now uses 2-D's bias-corrected ramp** (`d = min(decay, (1+it)/(10+it))`)
  instead of a fixed decay from step 0 — a fixed decay keeps ~37% of the random
  init blended into the EMA at it=1000 (right when physics turns on), which let
  a diverging live model look deceptively good in the "best" checkpoint during
  the exact window that mattered. First real run (batch=6, pre-EMA-fix) showed
  training-time residual loss improving steadily while `val_gen_res` (physics
  residual of samples generated from pure noise, the actual deployment metric)
  diverged catastrophically after ~2000 iters and samples mode-collapsed to one
  planform — the EMA ramp plus switching early-stopping to `fm` (so a volatile
  `val_gen_res` doesn't drive checkpoint selection/stopping) are the fixes
  ported from 2-D to address this; re-running to confirm.
- Filenames as proposed: `rb3d_pbfm_common.py`, `train_rb3d_pbfm_conditioned.py`,
  `ckpt_rb3d_pbfm_cond.pt`.

## Steps

1. **`rb3d_pbfm_common.py`**
   - Import `FNO3d`, `RB3DData` from `rb3d_pcfm_common.py`.
   - **Residual — reuse, don't rewrite:** `rb3d_pcfm_sampler.SteadyResidual` is
     already a batched, differentiable, autograd-friendly `h(u)` (Leray-projected
     bulk momentum + temperature, correctly row-scaled, wall-masked) — built for
     PCFM's Gauss-Newton projector, but exactly the residual PBFM's training
     loss needs too. Import and call it directly instead of writing a new one
     from the unbatched/float64/eval-only `evaluate_rb3d.py::Residual`.
     `SteadyResidual` deliberately excludes continuity (PCFM enforces div=0 by
     exactly Leray-projecting the *state*, outside Newton) — PBFM's plain-ODE
     sampler has no such projection step, so add a continuity term via
     `divergence_rms` (`generate_rb3d_multisolution.py`, already batched/
     differentiable) and combine as sum-of-squares (momentum, temp, continuity)
     for the training loss aggregate.
   - Port the multi-GPU infra verbatim from `rb2d_pbfm_common.py`:
     `ddp_setup`/`ddp_cleanup`/`ddp_allreduce_mean_`/`ddp_broadcast_flag`,
     `_flat_grads`/`_unflat_to_grads`, `config_direction` (ConFIG), `launch`.
   - Write `train_pbfm_3d`: unroll → residual loss → ConFIG-combined backward,
     EMA, resume/best checkpointing (per spec §Decided-3), validation (FM loss
     + generation-residual, matching 2-D's `val_gen`/`val_res` split).
   - Write `pbfm_sample_3d`: plain ODE integration by default; optional
     `project`/`hard_cleanup` ablation paths using `RB3DRelaxer`/`leray_project`.
2. **`train_rb3d_pbfm_conditioned.py`**
   - CLI mirroring `train_rb2d_pbfm_conditioned.py` (multi-GPU flags, model/
     training hyperparams, `--quick` smoke-test mode), calling into
     `rb3d_pbfm_common.train_pbfm_3d`. Sanity-sample at the end like the 2-D
     script does.
3. **Smoke test:** `--quick` run (few iters, tiny grid/hidden, 1 GPU) to confirm
   the residual loss, ConFIG combination, and checkpoint/resume round-trip all
   work before a real run.
4. **Real training run** on both T4s, watching step time to decide whether
   unroll/batch need adjusting (per the "size empirically" note above).
5. **`evaluate_rb3d.py` extension:** add PBFM as a third method column
   (vanilla/PBFM/PCFM), per `evaluation_and_paper_spec.md` Part A. Includes the
   flagged sanity check on `calibrate_thresholds`' reference independence
   (spec §A.3) before trusting valid% numbers.

Next: start on step 1.
