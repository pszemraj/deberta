# Development Log

This file records dated investigations that need more context than a commit
message: what changed, what was measured, what failed, and what remained at
the end of each campaign.

## 2026-07-05 - FlashDeBERTa Packed Doc-Block Remediation

Active goal: make both packed doc-block FlashDeBERTa routes correct and
end-to-end faster than eager for the repo's packed RTD configs, with committed
tests plus benchmark/training artifacts.

### Current State

- Checkpoint commit: `e624482 fix: keep docblock flash correctness-first`.
- Route policy updated after dense correctness and speed fixes: default packed
  doc-block route now uses table-selected dense `docblock_bias` for the shipped
  `1024`/`2048`/`4096` packed lengths. Set
  `model.hf.flash.docblock_bias_seq_len=0` to force the segment-aware ragged
  `docblock` route for ablations or hardware retuning.
- CUDA access requires escalated permissions in this environment. Sandboxed
  `torch.cuda.is_available()` can be false even when the RTX 5090 is available.
  Local `AGENTS.md` now records the escalation rule.
- Escalated CUDA visibility works:
  - GPU: NVIDIA GeForce RTX 5090
  - Driver: 580.159.04
  - VRAM: 32607 MiB total, about 32109 MiB free at check time
  - Torch: `2.9.1+cu128`

### Implemented Since `e624482`

- Versioned the internal doc-block custom-op names from
  `deberta::flashdeberta_docblock` / `deberta::flashdeberta_docblock_backward`
  to `deberta::flashdeberta_docblock_v2` /
  `deberta::flashdeberta_docblock_backward_v2`.
  - Why: `torch.compile` reused a stale generated backward call with the old
    schema after the custom op began returning/saving packed aux tensors.
  - Evidence: flash profile initially failed with
    `flashdeberta_docblock_backward() is missing value for argument 'q_unpad'`.
    The versioned op profile ran through compiled backward.
- Added route-specific varlen backward tuning support:
  - `_varlen_backward_raw_impl(..., route="varlen")`
  - doc-block backward now calls it with `route="docblock"`.
  - Added a unit test to ensure doc-block backward resolves tuning rows from
    the `docblock` namespace rather than generic `varlen`.

### GPU Profile Evidence

Artifacts live under:
`local-scratch/benchmarks/flashdeberta/docblock_goal_20260705/`.

Packed `1024` doc-block eager profile:

- Directory: `rtd_eager_after_aux_patch/`
- Command shape: `tools/flashdeberta_rtd_profile.py ... --mode eager
  --warmup-steps 4 --profile-steps 8`
- Median step: `955.20 ms`
- Mean step: `1171.95 ms`
- Peak memory: `7.71 GiB`
- Median generator backward: `12.48 ms`
- Median discriminator backward: `14.32 ms`

Packed `1024` segment-aware flash profile after aux-save and v2 op:

- Directory: `rtd_flash_segment_after_aux_patch_v2op/`
- Command shape: `tools/flashdeberta_rtd_profile.py ... --mode flash
  --warmup-steps 4 --profile-steps 8`
- Median step: `1501.21 ms`
- Mean step: `1747.31 ms`
- Peak memory: `6.86 GiB`
- Median generator backward: `56.81 ms`
- Median discriminator backward: `91.21 ms`
- Current result: correct-running but too slow. This fails the speed gate.

Profiler hotspot for segment-aware flash:

- `_bwd_kv_dise_kernel_varlen`: `5.751 s` CUDA over the window, about `52.7%`
  of CUDA time.
- `deberta::flashdeberta_docblock_backward_v2`: `6.712 s` self CUDA over
  `1100` calls.
- Segment pack/unpack kernels are now secondary:
  - `_pack_segment_grad_and_delta_kernel`: `88.45 ms`
  - `_pack_segment_rows_pair_kernel`: `130.06 ms`
  - `_unpack_segment_rows_pair_kernel`: `144.29 ms`

Interpretation: shaving metadata or pack/unpack overhead will not make the
ragged route faster at `1024`; the upstream varlen KV backward kernel dominates.

### Latest Experiment

Probe file:
`local-scratch/benchmarks/flashdeberta/docblock_goal_20260705/docblock_bwd_16x16_override.json`

Purpose: test whether smaller doc-block-specific KV tiles help the `1024`
ragged route.

Override:

- `route=docblock`, `kind=bwd_kv`, `seq_bucket=1024_exact`:
  `BLOCK_M=16`, `BLOCK_N=16`, `stages=1`, `warps=2`
- `route=docblock`, `kind=bwd_q`, `seq_bucket=1024_exact`:
  `BLOCK_M=32`, `BLOCK_N=32`, `stages=2`, `warps=4`

Running command shape:

```bash
env TORCH_BLAS_PREFER_CUBLASLT=1 TOKENIZERS_PARALLELISM=false \
  conda run --name neobert --no-capture-output \
  python tools/flashdeberta_rtd_profile.py \
    configs/flashdeberta/pretrain_rtd_hf_deberta_v3pos_smol2stage4_1024_wp32k_v2_docblock.yaml \
    --mode flash \
    --warmup-steps 2 \
    --profile-steps 4 \
    --kernel-overrides-path local-scratch/benchmarks/flashdeberta/docblock_goal_20260705/docblock_bwd_16x16_override.json \
    --profile-dir local-scratch/benchmarks/flashdeberta/docblock_goal_20260705/rtd_flash_docblock_bwd_16x16_probe
```

Result:

- Directory: `rtd_flash_docblock_bwd_16x16_probe/`
- Median step: `1330.95 ms`
- Mean step: `1607.00 ms`
- Peak memory: `6.86 GiB`
- Median generator backward: `45.17 ms`
- Median discriminator backward: `72.86 ms`
- `_bwd_kv_dise_kernel_varlen`: `3.62 ms/call`, down from `5.23 ms/call`.

Interpretation: doc-block-specific KV tile tuning helps, but only narrows the
gap. Eager remains about `955 ms` median step on the same config, so the ragged
route is still slower than eager and fails the speed gate. The remaining hotspot
is still the generic upstream varlen backward kernel; the fix needs to be
kernel-level or replaced by a dense route that is both correct and faster.

Follow-up default-table validation:

- Added the same measured `docblock` `bwd_kv` / `bwd_q` rows to
  `src/deberta/modeling/flashdeberta_kernel_tuning.json`.
- Added tests that prove the default table resolves those rows and that the
  doc-block backward asks the table using `route="docblock"` instead of the
  generic `varlen` namespace.
- Directory: `rtd_flash_docblock_default_table_after_tuning/`
- Median step: `1333.10 ms`
- Mean step: `1605.22 ms`
- Peak memory: `6.86 GiB`
- `_bwd_kv_dise_kernel_varlen`: `3.62 ms/call`

Interpretation: the checked-in table matches the local override profile. This
is an ownership improvement and a real speedup over the previous `1501 ms`
flash median, but it is not enough for the speed gate.

### Correctness Coverage Status

Already present:

- `tools/audit_contracts.py --strict` includes CPU-safe flash metadata checks:
  segment descriptor round-trip, route-implies-mask encoding, fixed descriptor
  shape, non-flash pairwise mask.
- `tests/test_flashdeberta_patch.py` covers doc-block fallback mask rebuilds,
  route selection, no-Triton imports for pack/varlen modules, segment pack
  round-trips, fused segment grad/delta fallback, and the v2 doc-block aux
  contract.
- Existing parity tool exits non-zero on failure and covers gradients for word
  embeddings, relative embeddings, and layer-0 query/value projections.

Still missing against the updated goal:

- Leakage pytest must cover `S in {1024, 2048}` and all three consumers:
  eager/no-flash, flash `docblock`, and forced eager fallback.
- Parity matrix must cover `docblock` and `docblock_bias` at
  `S in {1024, 2048, 4096}` with at least three docs per row and ragged tails.
  Current parity coverage is weaker: `docblock` at `S=256` and optional
  `docblock_bias` at `S=1024`, two-doc synthetic rows.
- Fresh venv/no-Triton full import and test collection artifact still needs to
  be recorded.
- Baseline branch failure set versus final non-regression set still needs a
  committed summary artifact.

### Speed And Training Gates Still Open

- Gate 6 speed matrix is not met. Current `1024` ragged route is slower than
  eager, not `>=1.10x` tokens/sec.
- Current dense `docblock_bias` training still reproduces the bad learning
  trend on this code state:
  - Directory: `train300_flash_dense_current_ckpt/`
  - Log: `train300_flash_dense_current.log`
  - Config: packed 1024 doc-block config, `attention_impl=flash`,
    `docblock_bias_seq_len=1024`, compile enabled, `max_steps=300`
  - Step 300: `loss=11.4384`, `gen=7.3779`, `disc=0.4060`, `acc=0.8511`,
    `tok/s=31258.0`
  - This is consistent with the earlier 1000-step dense route collapse and is
    not acceptable even though the dense route is faster than the ragged route.
- Current eager control on the same code/config:
  - Directory: `train300_eager_current_ckpt/`
  - Log: `train300_eager_current.log`
  - Step 300: `loss=10.0244`, `gen=6.1700`, `disc=0.3854`, `acc=0.8617`,
    `tok/s=34915.4`
  - Dense flash is about `0.895x` eager tokens/sec for this actual training
    run, and the generator loss is materially worse. The issue is not just
    speed; the dense route is still corrupting learning.
  - Added `tools/flashdeberta_rtd_compare_step.py` to compare eager vs dense
    flash on one identical packed batch and identical initial weights.
- Need end-to-end packed config speed artifacts for `S in {1024, 2048, 4096}`
  with GPU name, driver, torch/triton versions, config hash, and tokens/sec for
  both eager and flash arms.
- Need one `>=500` step packed `2048` training run per arm with compressed
  metrics and final MLM loss/discriminator accuracy within 1% relative.

### Likely Next Fix Areas

- Dense `docblock_bias` route:
  - Found a concrete layout bug: `_flash_docblock_bias` returns `(B,H,S,D)`,
    but the adapter viewed it directly as `(B,S,H*D)` instead of transposing to
    `(B,S,H,D)` first.
  - Added a regression in
    `tests/test_flashdeberta_patch.py::test_flash_attention_docblock_bias_path_records_stats`
    with per-head/per-token sentinels that fails on the bad layout.
  - Paired one-step diagnostic before fix:
    `rtd_compare_docbias_step0.json`
    - generator active hidden mean abs `0.2196`
    - generator masked-logit mean abs `0.1366`
    - discriminator active-logit mean abs `0.0598`
    - corrupted token mismatches: `50`
  - Paired one-step diagnostic after layout fix:
    `rtd_compare_docbias_step0_after_layout_fix.json`
    - generator active hidden mean abs `0.0026`
    - generator masked-logit mean abs `0.00275`
    - discriminator active-logit mean abs `0.00103`
    - corrupted token mismatches: `1`
  - Dense flash 300-step validation after layout fix:
    - Directory: `train300_flash_dense_after_layout_fix_ckpt/`
    - Log: `train300_flash_dense_after_layout_fix.log`
    - Step 300: `loss=10.5376`, `gen=6.4850`, `disc=0.4053`,
      `acc=0.8576`, `tok/s=31233.0`
    - This is no longer degenerate like the pre-fix dense flash run
      (`gen=7.3779`, `acc=0.8511`), but it is still behind eager
      (`gen=6.1700`, `acc=0.8617`, `tok/s=34915.4`).
    - RTD prediction inspection after fix:
      `rtd_prediction_inspect_flash_dense_after_layout_fix300.json`
      - active probability mean `0.1353`, std `0.0560`
      - replaced probability mean `0.1512`, std `0.0674`, max `0.6602`
      - original probability mean `0.1328`, std `0.0534`
      - discriminator accuracy `0.8613`, generator loss `6.1681` on the
        inspection batch
    - Current eager 300-step inspection:
      `rtd_prediction_inspect_eager_current300.json`
      - replaced probability mean `0.1609`, std `0.0906`, max `0.5521`
      - discriminator accuracy `0.8655`, generator loss `5.7707`
    - The old pre-fix flash 1000-step inspection had replaced probabilities
      nearly collapsed around `0.168`. The layout fix removes that flat
      discriminator behavior, but the route still trails eager and the speed
      gate still fails.
    - Dense post-layout parity:
      `parity_docbias_after_layout_fix.log`
      - `docbias_1024`: pass
      - `docbias_2048`: pass; this was the previous failing dense case.
      - `docbias_4096`: pass
      - Checked tensors: last hidden state, word embedding gradient, relative
        embedding gradient, layer-0 query gradient, and layer-0 value gradient.
      - The largest dense output error now matches eager bf16 error against
        fp32 reference instead of exceeding the strict `3x` threshold.
- Ragged `docblock` route:
  - Current bottleneck is `_bwd_kv_dise_kernel_varlen`, not pack/unpack.
  - If tile tuning is insufficient, real fixes are kernel-level: bucket LUT,
    deterministic/segmented position-gradient accumulation, or a true
    doc-block-specialized backward kernel rather than generic varlen.

### Failed Speed Attempt: Dense Bias Bucket Reduce Reuse

Attempted to replace the dense `docblock_bias` backward's local
`scatter_add_` bucket reduction with the existing `_dense_bucket_reduce`
helper from `flashdeberta_dense_bias_op.py`.

- Correctness tests passed:
  `pytest tests/test_flashdeberta_patch.py -k "dense_bias or bucket_reduce or docblock_bias or position_bias" -q`
  reported `10 passed, 4 skipped, 48 deselected`.
- Profile directory: `rtd_flash_dense_after_bucket_reduce_profile/`
- Median step worsened from `1078.76 ms` to `1385.11 ms`.
- Mean step worsened from `1301.06 ms` to `1603.29 ms`.
- Peak memory increased from `5.93 GiB` to `6.47 GiB`.
- The `scatter_add_` hotspot disappeared, but was replaced by large
  `aten::gather`, `aten::cumsum`, and `aten::where` costs.

Decision: revert this attempt. The dense-route speed fix needs to move the
position-gradient bucket reduction into a Triton/custom-op path or into the
specialized attention backward itself. A PyTorch `cumsum/gather/where`
post-process is slower at the target 1024 shape.

### Dense Docblock Direct Positional-Gradient Backward

Implemented the next kernel-level attempt for dense `docblock_bias`: the exact
1024 specialized Triton backward now optionally accumulates `dpos_key` and
`dpos_query` directly from the KV/Q backward tiles with `tl.atomic_add`, instead
of materializing full dense `d_bias` and reducing it through PyTorch.

Correctness:

- `parity_docbias_1024_direct_dpos.log`: pass.
- `parity_docbias_direct_dpos.log`: pass for `docbias_1024`,
  `docbias_2048`, and `docbias_4096`.
- Checked tensors remained: last hidden state, word embedding gradient,
  relative embedding gradient, layer-0 query gradient, and layer-0 value
  gradient.

Profiles:

- GA=8 profile directory: `rtd_flash_dense_direct_dpos_profile/`
  - Median step improved from the post-layout `1078.76 ms` baseline to
    `804.60 ms`.
  - Peak memory improved from `5.93 GiB` to `5.76 GiB`.
  - `aten::scatter_add_` disappeared from the CUDA hotspot table.
  - Kernel time shifted into `_bwd_kv_kernel_docblock1024` and
    `_bwd_q_kernel_docblock1024`, as expected from direct atomics.
- Apples-to-apples GA=1 warm6 profile directory:
  `rtd_flash_dense_direct_dpos_warm6/`
  - Eager warm6 median step: `130.57 ms`.
  - Old dense flash warm6 median step: `200.61 ms`.
  - Direct dense flash warm6 median step: `111.08 ms`.
  - Direct dense flash is `1.18x` faster than eager at 1024 and uses
    `5.26 GiB` peak memory versus eager `7.26 GiB`.

Interpretation: the dense 1024 route now satisfies the 1024 speed target on
the local RTX 5090 and no longer shows the earlier degenerate learning
behavior. The remaining speed gate is the ragged `docblock` route at 2048 and
4096.

### 2048 And 4096 Packed Docblock Profiles

All profiles below used `block_cross_document_attention=true` and the packed
RTD configs now under `configs/flashdeberta/`. These are end-to-end RTD optimizer-step
measurements, not attention-only microbenchmarks.

Fresh 2048 pairwise eager baseline:

- Directory: `rtd_eager_docblock2048/`
- Median step: `1799.98 ms`
- Peak memory: `11.37 GiB`

Fresh 2048 ragged `docblock` flash:

- Directory: `rtd_flash_docblock2048/`
- Median step: `2298.84 ms`
- Peak memory: `6.87 GiB`
- Hotspot: `_bwd_kv_dise_kernel_varlen`, about `8.26 ms/call`.

Result: ragged 2048 saves memory but is slower than eager. It fails the speed
gate and confirms the same root cause seen at 1024: generic varlen backward is
not competitive for packed doc-block RTD.

2048 dense `docblock_bias` with direct positional-gradient backward:

- Directory: `rtd_flash_docblock_bias2048_direct_probe/`
- Median step: `1169.59 ms`
- Peak memory: `5.83 GiB`
- Speedup versus 2048 eager: `1.54x`
- `aten::scatter_add_` is gone from the CUDA hotspot table.
- Main kernels:
  - `_bwd_kv_kernel_docblock1024`: about `1.59 ms/call`
  - `_bwd_q_kernel_docblock1024`: about `1.42 ms/call`
  - `_dense_bias_fwd_kernel`: about `0.62 ms/call`

Result: dense 2048 is now both correct by parity and faster than eager by more
than the `1.25x` target. This route remains O(S^2) in the dense-bias forward,
so the route-policy question needs an explicit documented decision rather than
being hidden in the benchmark command.

Fresh 4096 pairwise eager baseline:

- Directory: `rtd_eager_docblock4096_probe/`
- Median step: `2887.31 ms`
- Peak memory: `18.44 GiB`

Fresh 4096 ragged `docblock` flash:

- Directory: `rtd_flash_docblock4096_probe/`
- Median step: `2647.20 ms`
- Peak memory: `6.90 GiB`
- Speedup versus 4096 eager: `1.09x`
- Hotspot: `_bwd_kv_dise_kernel_varlen`, about `10.83 ms/call`.

Result: ragged 4096 is faster than eager but misses the `1.25x` gate. The
generic varlen backward is still the bottleneck.

4096 dense `docblock_bias` before direct positional-gradient specialization:

- Directory: `rtd_flash_docblock_bias4096_warm_probe/`
- Median step: `2889.20 ms`
- Peak memory: `7.79 GiB`
- Hotspot: `aten::scatter_add_`, about `1.95 s` over the profiled window.
- Dense-bias forward cost: about `1.16 s` over the profiled window.

Result: dense 4096 without direct positional-gradient accumulation is roughly
tied with eager and fails the speed gate. The next concrete hypothesis is that
enabling the direct positional-gradient kernels for 4096 can remove the
`scatter_add_` cost and possibly bring median step time under the `2310 ms`
threshold needed for a `1.25x` speedup versus eager.

4096 dense `docblock_bias` with direct positional-gradient backward:

- Code change under test:
  - Widened the table-gated dense doc-block specialization from
    `{1024, 2048}` to `{1024, 2048, 4096}`.
  - Added `sm_120` tuning-table rows for `dense_bias/fwd` and
    `bias_docblock_specialized/{bwd,bwd_kv,bwd_q}` at `4096_plus`.
  - Added a resolver regression so missing rows fail in pytest instead of
    silently falling back to the PyTorch dense positional-gradient reduction.
- Correctness:
  - `parity_docbias_4096_direct_specialized.log`: pass.
  - Largest output error matched eager bf16 versus fp32 reference:
    `3.2894e-02`.
  - Checked gradient tensors passed: word embeddings, relative embeddings,
    layer-0 query projection, and layer-0 value projection.
- Profile directory: `rtd_flash_docblock_bias4096_direct_probe/`
- Median step: `1979.94 ms`
- Peak memory: `6.18 GiB`
- Speedup versus 4096 eager probe: `1.46x`
- Main kernels:
  - `_dense_bias_fwd_kernel`: about `1.77 ms/call`
  - `_bwd_q_kernel_docblock1024`: about `2.79 ms/call`
  - `_bwd_kv_kernel_docblock1024`: about `2.78 ms/call`

Result: the direct dense 4096 route now clears the `1.25x` probe target and
removes the PyTorch `scatter_add_` hotspot. This still needs the formal
`>=10` warmup / `>=50` step profiling window before it can count for the speed
gate.

### Route-Policy Decision

Before the layout and direct-position-gradient fixes, dense `docblock_bias`
could not be the default: it was degenerate in short RTD training and slower in
several end-to-end runs. That is no longer the current state:

- The adapter layout bug is fixed and protected by a sentinel regression.
- Dense parity now covers `docbias_1024`, `docbias_2048`, and `docbias_4096`.
- Dense direct-position-gradient probes beat eager at all shipped packed
  lengths:
  - `1024`: `111.08 ms` versus eager `130.57 ms` (`1.18x`)
  - `2048`: `1169.59 ms` versus eager `1799.98 ms` (`1.54x`)
  - `4096`: `1979.94 ms` versus eager `2887.31 ms` (`1.46x`)
- Ragged `docblock` remains correct but is still bottlenecked by generic
  varlen backward at `2048`/`4096`.

Decision: promote dense `docblock_bias` through the JSON route table for the
measured packed `1024`/`2048`/`4096` buckets. Keep ragged `docblock` as a
forced ablation/hardware-retuning route via
`model.hf.flash.docblock_bias_seq_len=0`.

## 2026-07-06 - FlashDeBERTa Docblock Validation Campaign (goal close-out)

Goal: validate the committed `feat/flash` HEAD (`486174a` + this campaign's
commits) end to end - correctness, speed gates, and training equivalence - and
make the final route-policy call. Artifacts:
`local-scratch/benchmarks/flashdeberta/resume_20260706/` (phase0/1/2/4 dirs,
each profiler `summary.json` carries GPU/driver/torch/triton/config-hash
provenance). Environment: RTX 5090, driver `580.159.04`, torch `2.9.1+cu128`.

### Timeline Finding That Reframed The Open Questions

Artifact mtimes prove the alarming Jul 5 training comparison was stale
evidence: the `train300_*` runs (13:19-13:38, flash `0.895x` tokens/sec and
gen `6.4850` vs eager `6.1700`) predate the first direct positional-gradient
backward artifacts (13:50-13:52) and the formal speed gates (14:19-14:55).
The bad training numbers measured the old `scatter_add_` dense backward; no
training run had ever executed the code now at HEAD.

### Phase 0 - Correctness Gate At HEAD

- Found and fixed two commit-split regressions that existed at `486174a`:
  a stale test fake missing the new `route` kwarg
  (`test_docblock_backward_narrows_fixed_capacity_saved_aux`), and
  `tools/generate_config_reference.py` lagging the hand-updated route-policy
  prose in `docs/guides/config-reference.md`.
- `tools/audit_contracts.py --strict`: 14 PASS, 0 WARN/SKIP/FAIL.
- Full parity matrix (`tools/flashdeberta_parity_test.py`): pass at HEAD,
  including strict `docblock`/`docbias` cases at `1024`/`2048`/`4096`.
- Suite green after fixes (later in the campaign: `546 passed, 4 skipped`
  with CUDA visible, `phase0/pytest_final_gpu.log`).

### Phase 1 - Formal Speed Gates At HEAD (10 warmup / 50 steps, GA=8)

| S | eager tok/s | flash tok/s | ratio | peak mem eager -> flash |
|---|---|---|---|---|
| 1024 | 34886 | 44805 | 1.28x | 7.71 -> 6.79 GiB |
| 2048 | 18439 | 25123 | 1.36x | 11.38 -> 7.89 GiB |
| 4096 | 11501 | 15335 | 1.33x | 18.46 -> 10.33 GiB |

Eager baselines reproduce Jul 5 within 0.4%; flash at `4096` matches the Jul 5
saved-bias probe within 0.03%. Flash `1024`/`2048` improved ~7% over the Jul 5
gates because the saved-bias aux change landed after those gates were recorded.

### Phase 2 - Throughput/Learning Discrepancy Resolved

- Long-window probe (20 warmup / 300 steps, flash `1024`): `44742` tok/s -
  identical to the 50-step gate, so the profiler methodology is sound.
- One-step compare (`tools/flashdeberta_rtd_compare_step.py`, dense route):
  matches the post-layout-fix reference (gen hidden mean-abs `2.6e-3`,
  1 corrupted-token mismatch, disc logits mean-abs `1.03e-3`).
- Fresh 300-step training pair on HEAD: eager reproduces Jul 5
  (`gen=6.1672`, `34841` tok/s); flash now `gen=6.1679` (+0.011%) at
  `44640` tok/s (`1.28x`). Both Jul 5 anomalies are gone; stale-evidence
  hypothesis confirmed. No further diagnostics needed.

### Phase 4 - Training-Equivalence Battery

Four sequential arms at packed `1024` x 1000 steps (wandb offline with
gradient watch, debug metrics JSONL), plus a `2048` x 500 confirmation pair
using the new committed `_docblock` configs. Final-step values:

| arm | loss | gen | disc | acc | tok/s |
|---|---|---|---|---|---|
| eager seed 42 | 7.5408 | 4.3959 | 0.3145 | 0.8846 | 34571 |
| eager seed 1337 | 6.9337 | 4.0594 | 0.2874 | 0.8942 | 34286 |
| flash seed 42 (a) | 7.6114 | 4.3954 | 0.3216 | 0.8833 | 44098 |
| flash seed 42 (b) | 7.4817 | 4.3906 | 0.3091 | 0.8855 | 44265 |

- Seed-noise floor (eager 42 vs 1337): ~8% relative on loss/gen.
- Atomics-noise floor (flash repeat spread): ~1.7% on loss, 0.11% on gen.
- Flash-vs-eager same seed: <=0.94% on every metric, with eager's final loss
  landing between the two flash repeats - flash is indistinguishable from its
  own run-to-run noise and an order of magnitude inside the seed floor.
- `2048` x 500 confirm: flash `gen=5.4189` vs eager `5.4075` (+0.21%), acc
  `0.8705` vs `0.8708`, `1.32x` tokens/sec. Passes the <=1% relative bar.
- Comparison tables/plots: `phase4/battery_1024_curves.png`,
  `phase4/confirm_2048_curves.png` via `resume_20260706/analyze_phase4.py`.

### Route-Policy Decision (final)

Keep dense `docblock_bias` as the JSON-table default for packed
`1024`/`2048`/`4096`; keep ragged `docblock` as the forced ablation and
non-`sm_120` fallback route. Two accepted caveats are now documented in
`docs/advanced/flash-attention.md`: flash gradients are not bitwise reproducible
(atomic positional-gradient accumulation - resume/drift tooling must not
assert bit-exact replay), and the dense route saves the `(B,H,S,S)` bias for
backward (memory-for-recompute trade; a recompute knob is a prerequisite for
contexts beyond `4096`).

### Validation Coverage Closed This Campaign

- CPU leakage tests parametrized over `S in {1024, 2048}`.
- New CUDA-gated gradient-isolation leakage test drives the real Triton
  `docblock` and `docblock_bias` routes (route stat counters prove no silent
  eager fallback) and asserts exactly zero cross-document hidden-state
  gradients.
- No-Triton import test now covers all six flash modules and
  snapshot/restores the flash module namespace so import state cached under
  `triton=None` cannot leak into later tests (the custom-op modules may
  legitimately report available via the process-global torch.library registry).
- Committed doc-block configs so packed
  doc-block benchmark arms are config-owned instead of CLI-override-owned.

### Still Open (deliberately deferred)

- Fresh-venv no-Triton environment artifact (low marginal value now that the
  in-process test covers 6/6 modules).
- Dense-bias recompute-in-backward knob (needed before >4096 contexts).
- Ragged `docblock` kernel ownership (bucket LUT, deterministic segmented
  position-gradient backward) - only worth it if a deterministic-gradient or
  non-`sm_120` requirement materializes.

### 2026-07-07 PR Review Follow-Up (both findings confirmed and fixed)

Two unresolved PR #5 review findings were verified and fixed:

- P1 (`rtd.py`): the RTD head gated global CLS conditioning on attention-mask
  rank alone, so the ragged flash `docblock` route (2D keep mask + segment
  metadata) added the packed row's first token to every token. The first fix
  prevented leakage by dropping CLS conditioning for packed rows, but that made
  packed and standalone examples optimize different classifiers. The 2026-07-10
  correction gives every packed segment its own CLS and maps every token to that
  context, preserving the standalone RTD head while keeping documents isolated.
- P2 (`flashdeberta_dense_bias_op.py` and `flashdeberta_bias_op.py`): the
  original conclusion was wrong: it treated the repo-local Triton builder as
  the semantic oracle, but both that builder and native eager attention used
  the reversed P2C bucket. The canonical term is
  `pos_query[key, bucket(query,key)]`, using the same signed `query - key`
  bucket as C2P. The 2026-07-10 correction realigned native eager, dense
  PyTorch, Triton forward, and both P2C backward reductions against an
  independent scalar definition and the Hugging Face implementation. The
  earlier 546-test parity result compared implementations sharing the same
  reversal and was not evidence of canonical DeBERTa semantics.

## 2026-07-07 - PR #5 Review Remediation Campaign

A multi-agent review of PR #5 surfaced two live defects plus a cluster of
verified duplication/altitude debt. All were fixed as atomic commits on
`feat/flash` (from `fix(flash): make kernel-override reconfiguration
idempotent` through `refactor(flash): consolidate duplicated op
infrastructure helpers`).

### Live defects

- Per-micro-step tuning-table reload: `configure_flashdeberta_kernel_overrides`
  cleared the payload cache unconditionally and runs at the top of
  `prepare_flash_attention_batch_metadata` every micro-step, so every step
  re-read and re-parsed `flashdeberta_kernel_tuning.json` from disk
  (empirically 20 calls -> 20 `read_text`s; affected effectively all runs
  because `model.hf.flash` is never `None`). Reconfiguration is now a no-op
  for an unchanged path, and `flash_seq_bucket` /
  `resolve_flash_kernel_config` are `functools.cache`d with coordinated
  invalidation (the per-layer fwd/bwd table scans ran inside the opaque
  custom-op wrappers, invisible to `torch.compile`).
- `docblock_bias_seq_len` cross-coupling: `_should_use_local_bias` gated the
  plain-batch dense local-bias route on the packed doc-block override, so the
  documented `docblock_bias_seq_len=0` ablation silently disabled the
  unrelated local-bias kernel for every plain batch (git history shows the
  gate was an independent `_LOCAL_BIAS_SEQ_LEN` constant collapsed onto the
  docblock field in `bd2132e`). Added a dedicated
  `model.hf.flash.local_bias_seq_len` field with null/0/exact-length
  semantics; both knobs are now documented as independent.

### Structural fixes (verified findings, behavior-preserving)

- One `_clear_flash_batch_metadata` helper replaces the six-statement pop
  block copy-pasted at five early returns in
  `prepare_flash_attention_batch_metadata`; the sixth (non-`hf_deberta_v2`)
  early return had already drifted and left stale `flash_doc_segment_*`
  tensors in rope doc-block batches - now cleared and contract-tested.
- `FlashBatchMeta.is_cross_document()` is the single predicate for "packed
  cross-document batch" (the e6fd72b CLS-leak fix had hand-rolled this check
  at one call site); the RTD head now uses it.
- Fixed-vs-varlen padded routing has one resolver (`flash_padding_route` in
  `flashdeberta_kernel_tuning.py`); the training hint path and the
  model-internal density-blind fallback both delegate, so retuning
  density-gated table rows can no longer silently diverge the two.
- The four `_forward_{dense,masked}_hs{0,1}` fast paths are one-line
  delegations to `_forward_*_resolved` (~280 duplicated lines removed);
  compile.py's fallback closures already proved the pattern compile-stable,
  and eager + `torch.compile` (aot_eager) parity was verified directly. The
  dense fast path's lack of a `flash_meta` parameter is now a documented
  contract (dense-route metadata is semantically empty).
- `_dense_bucket_index_tensor` now composes the eager backbone's
  `build_relative_position` instead of re-deriving the log-bucket formula;
  a reference test locks equality with the prior math in the shipped regime
  (`max_relative_positions >= seq_len`), and beyond it the shared math now
  saturates like eager (distant-negative offsets bucket to row 1, where the
  standalone formula drifted to row 0 - latent only, unreachable in shipped
  configs).
- Dropped the six dead flat `cfg.flash_*` mirrors (only `cfg.hf_flash` was
  ever consumed); consolidated `is_torch_compiling` (5 copies -> mask_utils),
  `lookup_registered_op` and `device_compute_capability` (5-6 copies each ->
  new `flashdeberta_op_utils.py`); tuning tools compose
  `compute_capability_key` with the shared resolver.

### Validation

- Full suite with CUDA visible: `552 passed, 4 skipped` (6 new regression
  tests over the campaign baseline; skips are the usual compiled
  position-bias environment gates).
- `tools/flashdeberta_parity_test.py`: full matrix passes at HEAD after the
  bucket-math consolidation and again after the helper consolidation.
- `tools/audit_contracts.py --strict`: 14 PASS, 0 WARN/SKIP/FAIL.
- No-triton import test still covers the flash module tree including the new
  `flashdeberta_op_utils`.

### Review findings intentionally not "fixed"

- Golden test pinning shipped `sm_120` table rows: deliberate per the
  2026-07-06 campaign notes; will churn on retune by design.
- Compiled dense route dropping `route_hint="dense"` metadata: confirmed
  inert (doc-block batches always carry a real mask and cannot reach the
  dense branch); resolved by documenting the tensor-only dense contract
  rather than threading a semantically empty object through Dynamo guards.

## 2026-07-07 - Redundancy/Dead-Code Consolidation Campaign

Read-only audit (7 parallel scope reviews + mechanical zero-consumer and
AST-clone scans over all 53k tracked Python lines) followed by a fix
campaign. All findings either resolved or explicitly deferred below.

### Dead code removed

- Flash op modules: `_select_rows_or_none` / `_docblock_fixed_forward_impl` /
  `_docblock_fixed_backward_impl` (git-confirmed debris from `bf0d6e7`'s
  row-partitioning removal), chained `_fixed_eager_backward_impl`,
  `flashdeberta_compiled_fixed_available`, `_varlen_density_bucket`,
  `active_flashdeberta_kernel_overrides_path`, `_normalize_route_hint`,
  `_flash_cfg_int`, `clear_segment_pack_host_cache` (~280 lines, all
  verified zero-consumer repo-wide including string/monkeypatch refs).
- `ConvLayer` + `_input_mask_for_conv` removed from the native backbone.
  The repo never sets `conv_kernel_size`; the only reach was loading an
  external DeBERTa-v2 xlarge/xxlarge checkpoint config, which is out of
  scope for this v3-focused repo. A conv-bearing config now fails with an
  explicit `ValueError` (new rejection test) instead of silently loading
  without conv weights. This also removes the unconditional
  `_input_mask_for_conv` computation that ran on every masked forward.

### Consolidations (production)

- Mask-shape zoo: `mask_utils` now owns `mask_to_2d_keep_mask`,
  `is_pairwise_mask`, `expand_keep_mask_to_4d`, `reduce_keep_mask_to_2d`.
  Six drifting copies across attention/native/rtd collapsed onto them;
  the rank-3 `(B,1,S)` broadcast divergence between the old copies is
  fixed (unreachable today, pinned by a new rank-matrix test).
- `training/compile.py`: 8 byte-identical routed masked entrypoints ->
  `_make_routed_masked_fn` factory (distinct function objects preserved
  for Dynamo; verified bitwise via an aot_eager dispatcher probe over
  dense + all 5 masked routes, hs0+hs1). Shared
  `_resolve_flash_seq_lengths_and_active_tokens` for the doc-block and
  padded branches.
- `cli.py` now calls `runtime._apply_backbone_defaults_and_validate_training_configs`
  instead of hand-copying the validator sequence (the CLI copy had already
  drifted: it skipped `_sync_legacy_train_aliases`).
- `data/collator.py`: unigram/ngram masking share
  `_resolve_masking_hyperparams` + `_apply_mask_replacement_policy`;
  verified bit-identical on a 6-case seeded golden (RNG draw order
  preserved).
- `entrypoint.py`: checkpoint-save (4 sites, 9 kwargs) and non-finite
  debug-artifact (5 sites, 14 kwargs) calls folded into closures;
  compile-scope resolution shared between dry-run and full run.
- Flash op/pack plumbing into `flashdeberta_op_utils`: `kernel_dtype_name`
  (5 copies), `lookup_existing_op_pair` (6 sites), `optional_triton_jit`
  (3 verbatim copies), `traceable_triton_kernel`, `flatten_padded_rows`,
  `can_use_triton_pack`. Optional pos-pair pack/unpack helpers added to
  both pack modules (6 plain sites); the 3 cache-aware/both-or-none inline
  variants intentionally kept (different semantics).
- `bias_op` backward: generic positional fallback now uses
  `dense_bias_op._dense_bucket_reduce` (cached contiguous-range reduction,
  same p2c transpose convention) instead of its own scatter-add; the two
  docblock1024 backward launchers share `_launch_docblock1024_backward`
  (owns the strictly-positional Triton argument order once). Gated on the
  full CUDA parity matrix (OK) plus the autograd-differential reduction test.
- `flashdeberta_attention`: doc-block scalars resolved once in `forward()`
  and threaded into `_flash_docblock` (was recomputed per call);
  `mask_utils` doc-block eye/CLS caches now bounded (8 entries, evict-half)
  matching the attention bucket-index cache policy.
- Native model: dynamic vs cached_bmm disentangled-bias variants (~90%
  identical) merged into one core with a per-term `use_bmm` branch;
  verified bitwise on a 12-case golden (3 kernels x 4 pos_att_type).
  `DebertaV2Intermediate` uses `activations.get_act_fn`.
- `config.py`: `_legacy_key_suggestion` derives from the executable
  `_LEGACY_MAP`s (the hand-copied table had drifted on `pretrained_*` rope
  paths); `apply_dotted_override` reuses `_replace_path`.
- `builder.py`: shared `_load_pretrained_backbone_or_raise` for the 4
  copy-pasted from_pretrained blocks.
- `tools/_bench_common.py`: sys.path bootstrap (6 copies), autocast (2),
  candidate parsing/env (2 each), real-loader chain (4), RTD model build
  (2), tuner sampling/timing loops extracted. The tune-script twins are
  now thin CLIs; two audited drifts fixed deliberately (bias_tune builds
  the model once per run, peak memory read per-device).
  `generate_config_reference.py` keeps its own tiny bootstrap so the docs
  generator does not import torch.
- `tools/audit_contracts.py`: the AST-exec machinery (hand-maintained
  globals dicts, itself the shadow-copy failure mode) replaced with plain
  imports; sibling checks already imported from the same modules.
- Tests: shared `capture_run_pretraining_kwargs` (was 10 byte-identical
  copies), `fake_torch_compile` (3), `checkpoint_saving_accelerator`
  (removes the only cross-test-module private import), resume suite
  adopts the `mock_checkpoint` fixture (6 hand-rolled scaffolds), the
  redundant `_has_nonfinite_grad_norm_any_rank` reduce-mocking trio
  collapsed to one parametrized translation test, two misplaced
  `build_backbone_configs` tests moved to `test_builder.py`, flash
  stats-test harness + strengthened single-route assertion, two smoke
  near-duplicate pairs parametrized, EMD trio scaffolding factored.

### Deferred at campaign close

- `entrypoint.py` decoupled vs joint training loops: genuinely different
  one- vs two-optimizer semantics, heavily tested; only the call
  boilerplate around them was deduplicated.
- `varlen_op` eager vs Triton-op path duplication: perf-motivated
  (identity-keyed mask-metadata cache is unsafe under compile), documented
  in the wrapper docstring; a merge needs dedicated parity work.
- Pack-kernel unification (prefix as segment-with-computed-offsets):
  correct in principle but hot-path; needs its own benchmarked change.
- `prefix_pack` rank4-strided kernels (~150 lines): suspected unreachable
  in production (all producers are contiguous); needs GPU instrumentation
  during a real run before deletion. Not asserted dead.
- `rtd_profile` decoupled/coupled windows mirror `entrypoint.py` step
  logic by hand; anyone changing the real step must mirror it or profiler
  numbers stop representing training.
- `norm.py` RMSNorm (stable FSDP2 param names), `get/set_input_embeddings`
  duplicated native/rope (HF API convention), builder pooler kwargs
  (export interop), config `__init__` kwargs-partition pattern, and the
  `_config_and_training_shared_imports` grab-bag.

### Validation

- Full suite with CUDA: 555 passed, 4 skipped (552/4 pre-campaign; net
  +3 from the conv-rejection, mask rank-matrix, and parametrized folds).
- `tools/flashdeberta_parity_test.py`: full matrix OK after the pack
  optional-pair change and again after the bias_op backward merges.
- Bitwise goldens: collator masking (6 cases), native disentangled bias
  (12 cases), compiled dispatcher probe (12 entrypoints, aot_eager).
- `tools/audit_contracts.py --strict`: 14 PASS, 0 WARN/SKIP/FAIL.
- API docs regenerated (`tools/generate_api_docs.py`); the diff was
  stale-doc catch-up for `flash_meta` parameters, not campaign changes.

## 2026-07-07 - Hardware Portability Hardening (capability-scoped routing)

The sm_120-only risk assessment concluded kernel tuning was self-gating but
route policies shipped sm_120-derived defaults globally: non-sm_120 GPUs
silently inherited dense `docblock_bias` (whose speed depends on
sm_120-gated backward specializations, and which saves the dense
`(B,H,S,S)` bias for backward) and the small-batch `local_bias` route.

- Route-policy rows now accept `compute_capability` (exact `sm_XX` beats
  wildcard `*`; later rows win within a tier). Shipped table: dense
  doc-block and local-bias defaults scoped to `sm_120`; wildcard packed
  doc-block default is ragged `docblock`; padded `fixed`/`varlen` split
  stays hardware-agnostic. sm_120 behavior is bit-for-bit the same policy.
- Route hints and the local-bias/varlen gates thread the batch device
  capability. Verified Dynamo-safe via 2-step compiled (aot_eager) flash
  smokes on the dense and packed-docblock configs.
- Override-table merge fix: route-policy namespaces append per row instead
  of replacing wholesale, so promoting one capability-scoped row for a new
  GPU cannot silently delete the shipped policy (the old semantics were a
  footgun for exactly that workflow).
- One-time logger warning on the first flash batch on a GPU with no
  measured rows, pointing at the new doc.
- New `docs/advanced/gpu-support.md`: capability table (sm_120/100/90/89/
  86/80), the no-eager-fallback-by-GPU guarantee, per-class expectations,
  knob and override-table recipes for promoting dense routes off sm_120,
  and why kernel tiles are capability-scoped. Config-reference guidance
  regenerated; flash-attention.md updated to the capability-scoped story.
- Validation: flash file 68 passed / 4 skipped with CUDA, full suite
  557/4, audit_contracts 14 PASS, full parity matrix OK, compiled smokes
  clean, doc links resolve (17 files).

## 2026-07-08 - Speed/Quality Gate Re-Validation At PR-Review HEAD

Question: does flash still beat eager after the 31 commits that landed on top
of the validated `486174a` base (capability-aware routing, the fail-closed
mask/fallback gates from PR #5 review rounds, and the consolidation
refactors)? Re-ran the formal phase-1 speed gate and the full parity matrix at
HEAD `610148e`. Artifacts:
`local-scratch/benchmarks/flashdeberta/headcheck_20260708/` (same
methodology: 10 warmup / 50 profiled steps, GA=8, packed doc-block arms,
summary-only). Environment unchanged: RTX 5090, driver `580.159.04`, torch
`2.9.1+cu128`, triton `3.5.1`, flashdeberta `0.0.7`.

| S | eager tok/s | flash tok/s | ratio | peak mem eager -> flash |
|---|---|---|---|---|
| 1024 | 33913 | 43232 | 1.27x | 7.71 -> 6.78 GiB |
| 2048 | 18147 | 24545 | 1.35x | 11.36 -> 7.90 GiB |
| 4096 | 11381 | 15109 | 1.33x | 18.44 -> 10.32 GiB |

- Every arm reproduces the 2026-07-06 campaign gate within 3% (eager and
  flash alike); ratios match to the second decimal. The review-round
  fail-closed gates cost nothing on the training hot path, as designed: the
  packed doc-block branch in `prepare_flash_attention_batch_metadata` returns
  before the new prefix-mask check, and the per-layer check is skipped
  whenever metadata `seq_lengths` are published.
- Step-metric means are identical between arms at all three lengths
  (`gen_loss` 9.4640/9.4378/9.4279, `disc_acc` 0.8503/0.8506/0.8506 to
  displayed precision) - same data, same learning signal.
- Full parity matrix OK at HEAD: dense, fixed_padded, varlen, local_bias,
  docblock and docbias at `1024`/`2048`/`4096` (plus the `b4` batch case),
  outputs and selected gradients all inside limits.
- Training equivalence itself was not re-run; the 2026-07-06 1000-step
  battery remains the evidence of record (flash-vs-eager <=0.94% on every
  metric, inside flash's own ~1.7% atomics-noise floor), and no commit since
  touched numerics on the non-fallback paths.

### PR #5 P2: Dense Doc-Block Route Bounded To Measured Lengths

Reviewer finding (confirmed live at `610148e`): the `4096_plus` bucket has no
upper bound, and the `sm_120` docblock policy row mapped it to dense
`docblock_bias` - so a packed doc-block batch at any `S > 4096` would route
dense and save the quadratic `(B,H,S,S)` bias (about 1.5 GiB per layer at
`8192`, B=1, H=12), an OOM instead of the ragged fallback the docs promise
for unlisted shapes. Fix keeps policy table-owned: route-policy rows now
accept optional `min_seq_len`/`max_seq_len` bounds (checked in
`flash_route_choice` when the caller supplies `seq_len`; a bounded-out row
resolves as if the namespace had no entry, so the consumer's conservative
default applies), and the shipped `sm_120` `4096_plus` dense row carries
`max_seq_len: 4096`. The `docblock_bias_seq_len` knob remains the explicit
exact-length opt-in for unmeasured lengths. Regression assertions added to
the existing route test (verified failing pre-fix); gpu-support override
example updated to model the bounded row.

## 2026-07-09 - Pre-Merge Adversarial Review Campaign

Five-perspective adversarial review of the whole branch before merging PR #5
(model lifecycle, attention/mask semantics, repo hygiene, route-table/config,
production long-run behavior), constrained to code reading plus small CPU
repros. Every confirmed finding was fixed in its own commit with a regression
test verified failing pre-fix (except the consistency-pinning test, which
guards currently-correct behavior); the rest went into the flash doc's Known
follow-ups with exact names.

### Fixed (one commit each)

- Three sm_120 tuning-table tests crashed (not skipped) on CPU-only machines:
  they monkeypatched `torch.cuda.get_device_capability` behind an unpatched
  `current_device()` call and a per-index capability cache. Now patch the
  repo seam `device_compute_capability` at the consuming op module.
- Override-table `seq_buckets` rows were unreachable: bucket resolution is
  first-match-forward and the shipped buckets cover every length, while the
  merge appended override rows. Override buckets now prepend. Reversing the
  iteration instead was rejected - it would flip shipped resolution at 1024
  (`under_2048` before `1024_exact`) and silently disable the local-bias
  policy row and 1024_exact kernel rows.
- Reconfiguring the same override path after rewriting the file served the
  stale table forever (path-string no-op guard). The guard now also compares
  an `(mtime_ns, size)` signature - one stat call on the per-batch hot path.
- `flash_padding_route` ignored policy-row `min/max_seq_len` bounds (never
  threaded `seq_len` into `flash_route_choice`); inert with the shipped
  table but the same fail-open asymmetry the docblock bound fix closed.
- Fail-open relative-position divergence: eager clamps relative positions to
  `max_relative_positions` before log-bucketing, the flash kernels bucket
  unclamped (87% of buckets differ at seq 1024 with span 128). Unreachable
  with repo configs (`-1`), live for external checkpoints pinning a short
  span. Build-time validation now rejects `0 < span < max_position_embeddings`
  for flash.
- Silent fallback invisibility under compiled training: per-call fallback
  warnings are (correctly) dead inside compiled graphs and the batch-prep
  non-prefix-mask eager downgrade logged nothing. Batch prep (host-side,
  outside graphs) now warns once per process; docs corrected -
  `FLASHDEBERTA_WARN_FALLBACKS` defaults on, not off.
- Dense doc-block routing had no batch-size guard: the saved `(B,H,S,S)`
  bias costs ~4.5 GiB per batch element at 4096, so bumping batch size on a
  shipped packed config OOM'd at step 1 with no route linkage. Policy rows
  now accept `max_batch_size`; shipped sm_120 dense rows carry 8/2/2 at
  1024/2048-family/4096, sized to each bucket's worst case (the 2048 buckets
  span to 4095 where bias cost matches 4096). Out-of-bounds goes ragged.
- `export_meta.json` leaked the exporting machine's absolute checkpoint/run
  paths inside every exported directory; now records directory names only.
- New consistency test pins the three independent active-token definitions
  (GA weighting, tokens/sec logging, RTD loss / `doc_ids.ne(0)`) to each
  other for packed doc-block batches with padded tails.
- Removed the two zero-caller pack import-error getters
  (`flashdeberta_{prefix,segment}_pack_import_error`).

### Reviewed and intentionally not changed

- The test-only public dense-bias custom op duplicated the production
  bias_op registration but was load-bearing for GPU parity tests and the
  no-Triton import contract during this round. Later consolidation moved
  those tests to production seams and removed the wrapper.
- `reduce_keep_mask_to_2d` still slices longer masks by design (embeddings /
  RTD activity path where masks legitimately outrun sliced hidden states).
- Eval-bypass contract (callers that skip
  `prepare_flash_attention_batch_metadata` and manually drop `doc_ids` get
  cross-document attention with no in-model guard) is documented in the
  data-pipeline guide; model forwards already reject a raw `doc_ids` kwarg
  loudly, and the repo ships no eval CLI today.

### Verified clean by the review (no action)

Export config stripping complete and stock-`transformers` loadable exports;
loud failures for missing/mispinned flashdeberta at resume; dropout+flash
double-gated; no recompile storms possible (route dispatch is an uncompiled
dict lookup over prebuilt graphs); degenerate doc-block batches guarded;
resume tooling asserts no bit-exactness on flash paths; no committed junk,
stale docs, or dead config knobs; `doc_ids==0` means inactive consistently
across attention, loss, GA, and logging.

## 2026-07-09 - Second Adversarial Round: The Hardening Round Itself

Five fresh adversarial reviewers re-attacked the previous round's 12 commits
(`edfcd6a..876f896`): fix correctness, test integrity, docs-vs-code, blast
radius, and a maintainer-in-3-months operational lens. The mechanisms all
survived (override precedence, reload signature, bounds threading, guard
boundary, export rename - several proven with CPU checks against the real
loader), but the round had made routing *safer while quieter*: it added
silent behavior changes without the signal to see them.

### Fixed (one commit each, regression tests verified failing pre-fix)

- Dense-vs-ragged doc-block routing was invisible in compiled training: the
  per-layer `FLASHDEBERTA_DEBUG_STATS` counters are hard no-ops under
  `torch.compile` and batch prep recorded nothing, so a batch-size bump
  crossing a new `max_batch_size` bound silently cost the ~1.3x speedup.
  Batch prep now logs the chosen route once per `(route, S, B)` and warns
  when table bounds - not missing measurements - keep a shape ragged.
  Validated on the 5090: all three notices fire with correct routes.
- All host-side once-per-process flash notices are now rank-0 gated via the
  launcher `RANK` env var; previously every rank logged its own copy,
  against the `is_main_process` convention used elsewhere.
- `docblock_bias_seq_len` bypassed the table's batch/seq bounds silently (it
  is checked before the table and never read `batch_size`); a stale tuning
  knob reused in a larger-batch config would OOM on the `(B,H,S,S)` backward
  bias with no trail. Now warns once per shape when the knob (not the table)
  is why dense engaged, and the bypass is documented on the config class and
  in the generated config reference (guidance sourced from
  `tools/generate_config_reference.py` - the checked-in doc is
  generator-owned).
- Same-name `seq_buckets` overrides could only widen a shipped bucket:
  narrowing silently fell through to the still-present shipped row. Same
  names now replace the shipped bucket; new names still prepend.
- `flash_padding_route` had `batch_size` in hand (it feeds bucket density)
  but never forwarded it to the bound check - the same fail-open asymmetry
  the docblock path fixed, one namespace over.
- The flash span guard silently no-oped when `max_position_embeddings` was
  `None` (zero is rejected upstream, `None` skips that check); a pinned
  `max_relative_positions` with no known position range now fails fast.
- The active-token consistency test claimed four pinned definitions but
  asserted three: `flash_active_tokens` (segment-metadata cumsum, a
  structurally different algorithm) is now pinned through
  `prepare_flash_attention_batch_metadata`, plus a genuinely-packed
  single-row case (`attention_mask=None`) the padded fixture never reached.
- New invariant test: a bounded-out sm_120 batch must resolve exactly like
  hardware with no measured rows (the consumer's hardcoded ragged default
  only coincidentally equals the table wildcard row; now pinned).
- Docs: the `4.8`/`1.6 GiB` bias figures were bytes/1e9 mislabeled as GiB
  (actual: `4.5`/`1.5 GiB` at the repo's `1024^3` convention); gpu-support
  now states the sm_120 dense-row bounds encode the 32 GiB benchmark card's
  VRAM (not a capability fact) with a copy-pasteable override row for
  bigger same-capability cards; route stability's dependence on the
  hardcoded `drop_last=True` is recorded at the DataLoader site and in the
  flash caveats.

### Verified clean this round (no action)

Every doc claim about knob names, env defaults, forward signatures, and the
`doc_ids` misuse contract; the 87% bucket-divergence figure (reproduced at
87.54%); raising a bound via override tables (proven end-to-end against the
real loader); all prior-round tests genuinely regression-sensitive; TOCTOU
ordering in the override reload; export rename and deleted getters have zero
stale references.

Gates: full suite 581 passed / 12 skipped, `audit_contracts` 14/14 PASS,
GPU notice validation plus a 2-step compiled flash RTD profile probe
(6.73 GiB peak, step metrics consistent with the recorded baseline).
