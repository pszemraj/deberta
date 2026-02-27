# FSDP2, Runtime, and Export

This document is the primary reference for distributed training setup and runtime knobs.

For model/backbone options and load/source-resolution policy, see [`docs/model.md`](model.md) and [`docs/model.md#source-resolution-contract`](model.md#source-resolution-contract). For input pipeline behavior, see [`docs/data.md`](data.md). For RTD objective/loss details, see [`docs/objective.md`](objective.md).

## Accelerate + FSDP2

This project targets PyTorch 2.9.1+ and uses Hugging Face Accelerate with:

- `distributed_type: FSDP`
- `fsdp_version: 2`

Provided configs:

- RoPE backbone: [`configs/fsdp2_1node.yaml`](../configs/fsdp2_1node.yaml)
  - wrap class: `DebertaRoPELayer`
- HF DeBERTa compatibility path: [`configs/fsdp2_hf_deberta_1node.yaml`](../configs/fsdp2_hf_deberta_1node.yaml)
  - wrap class: `DebertaV2Layer`

## Precision Defaults

Training defaults to mixed precision `bf16` autocast (`train.mixed_precision=bf16`).

This is autocast-based mixed precision, not full-parameter bf16 casting.

Runtime behavior:

- if CUDA bf16 is unavailable or a tiny bf16 preflight matmul fails, training falls back to full precision (`no`)
- `train.mixed_precision` accepts `bf16` or `no`

## Run Directory and Tracker Naming

Runtime defaults are config-driven:

- if `train.output_dir` is unset, training auto-creates `runs/<project_name>/<timestamp>_<config_stem_or_run>`
- if `train.report_to=wandb` and `train.run_name` is unset, run naming defaults to the resolved output directory basename

## W&B Gradients and Weights

When `train.report_to=wandb`, model watch is config-driven and enabled by default for gradients:

- `train.wandb_watch=gradients` (default)
- `train.wandb_watch_log_freq=100` (default)

Supported watch modes:

- `none`
- `gradients`
- `parameters`
- `all`

Example YAML override for gradients + weights:

```yaml
train:
  report_to: wandb
  wandb_watch: all
  wandb_watch_log_freq: 200
```

## TF32 Policy

`train.tf32=true` by default.

The training loop prefers modern PyTorch fp32 precision controls when available and falls back to legacy `allow_tf32` flags on older builds.

For `torch.compile` max-autotune modes, legacy TF32 flags may be forced for compatibility.

## `torch.compile`

Enable with `train.torch_compile=true` and `train.torch_compile_mode` in `default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`.

If `train.torch_compile=true`, compile failures are treated as hard errors (no silent eager fallback).

RTD wrapper glue (sampling/corruption/label construction) remains eager by design.

Default compile scope is resolved from `train.torch_compile_scope=auto`:

- baseline: compile both `generator` and `discriminator` backbones
- temporary default-mode mitigation: for `model.backbone_type=hf_deberta_v2` with `train.torch_compile_mode=default` and backend `inductor`, auto scope compiles FFN blocks only (`generator.encoder.layer[*].intermediate/output`, `discriminator.encoder.layer[*].intermediate/output`)

This preserves compile on dominant MLP FLOPs while avoiding known unstable default-mode inductor paths in full HF-DeBERTa attention compile.

Recommended stable HFv2 compile path:

- `model.hf_attention_kernel=stable`
- `train.torch_compile_mode=default`
- `train.torch_compile_backend=inductor`
- `train.torch_compile_scope=ffn` (or `auto` with the default-mode fallback)

Full-backbone HFv2 compile with inductor is currently unstable and not recommended for production pretraining runs. The runtime now emits a warning if this path is requested explicitly.

Compile behavior is configured directly via train/model config:

- `train.torch_compile_scope=auto|backbones|encoder|gen_encoder|disc_encoder|ffn|gen_ffn|disc_ffn`
- `train.torch_compile_backend=inductor|aot_eager`

Native HF attention-kernel variants (`model.hf_attention_kernel`) are defined in [`docs/model.md#hf-compatibility-mode-notes`](model.md#hf-compatibility-mode-notes).
For native HF runs, prefer `model.hf_attention_kernel=stable`.

Compile is applied to module `forward` callables (module identity is preserved), so
new checkpoints keep canonical state-dict keys. Resume still normalizes legacy
compile-wrapper key segments (`._orig_mod`) when needed for older checkpoints.
Main-process runtime now shows tqdm progress for both training steps and resume
data-replay micro-batch alignment.

### Compile Parity Protocol

Use `scratch/compile_parity_check.py` for eager-vs-compiled parity checks on the same weights, batch, and RNG state.

- default gate: `--eval` mode (deterministic parity threshold is enforced)
- optional diagnostic: `--train` mode (informational deltas only)
- compile bisect scopes: `--scope gen|disc|both|matrix`
- backend isolation: `--backend inductor|aot_eager`

Use `tools/compile_drift_probe.py` when parity is inconclusive and train-mode drift appears near/after warmup:

- compares eager vs compiled train steps from identical initial weights
- reports per-step loss delta + compact parameter drift
- prints best-effort Dynamo counters (`stats` / `recompiles`) for graph-churn diagnosis

Use `scratch/hf_attention_inductor_repro.py` for a minimal one-layer train-step repro focused on native HF attention internals.
Use `scratch/compile_log_summary.py` to convert `wandb/*/files/output.log` into final/min/max/checkpoint/runtime summary tables.

### Non-Finite Diagnostics

Training keeps strict scalar checks and non-finite recovery behavior for gradient overflows.

- checks run before backward (`gen_loss_raw`, `disc_loss_raw`, forward/backward scalar objectives)
- checks run before optimizer step (global gradient norm, and post-clip gradient norm when clipping is enabled)
- when non-finite gradients are detected, training first sanitizes non-finite grad elements to zero and retries the norm check
- if gradients are still non-finite after sanitize, the accumulation window is skipped, recovery is applied (LR backoff and periodic optimizer-state reset), and training continues
- skipped non-finite windows are logged as `nonfinite_window_skipped=1` with `nonfinite_skip_total`, `nonfinite_skip_streak`, and recovery metrics
- when triggered, a compact debug artifact is written to:
  - `<output_dir>/debug/nonfinite_step_<STEP>_<TAG>.json`
  - includes step/lr, compile mode, embedding sharing mode, scalar snapshots, and compact RNG state heads

## Training Metrics Logging

Step logs and tracker metrics prioritize signal over mostly-constant counters.

- primary scalar metrics: `loss`, `gen_loss`, `disc_loss`, `disc_acc`, `lr`
- throughput/scale metrics: `input_tokens_per_sec`, `input_tokens_seen`
- noisy low-information counters (for example per-step `gen_tok`/`disc_tok`) are intentionally excluded from periodic logs

## Interruption and Crash Handling

`deberta train` now uses a crash-safe shutdown path:

- `KeyboardInterrupt` (CTRL+C) and other uncaught exceptions are logged with crash type/reason and step.
- Main process appends a crash marker row to `<output_dir>/metrics.jsonl.gz`.
- If trackers are enabled, crash fields are logged to the tracker step; for W&B runs, crash summary fields are also set before finish.
- A best-effort final checkpoint save is attempted if training progressed and that step was not already saved.
- Tracker shutdown and logger flush happen in a `finally` block so logs/artifacts are flushed even on failure.

For distributed failures, crash-time final-save can be skipped to avoid collective deadlocks after backend failure.

## SDPA Kernel Policy

Use `train.sdpa_kernel` to set SDPA backend preference:

- `auto` (default)
- `flash`
- `mem_efficient`
- `math`

`flash` is strict flash behavior. If flash attention is not supported for the runtime/device/shape, execution should fail instead of silently falling back.

`train.sdpa_kernel` is only behaviorally relevant when `model.backbone_type='rope'` and `model.attention_implementation='sdpa'`. For rope eager attention, validation requires `train.sdpa_kernel=auto` to avoid inert config differences.

When `data.pack_sequences=true` and `data.block_cross_document_attention=true`, packed batches may emit the pairwise `(B, S, S)` keep-mask defined in [`docs/data.md#pairwise-mask-contract-block_cross_document_attentiontrue`](data.md#pairwise-mask-contract-block_cross_document_attentiontrue). That mask path is incompatible with strict flash SDPA kernels, so `train.sdpa_kernel=flash` is rejected by config validation for that workflow.

Use `auto`, `mem_efficient`, or `math` for packed+doc-block training runs.

## Embedding Sharing and FSDP Safety

FSDP-safety details and model-level sharing semantics (including post-init module replacement behavior) are defined in [`docs/model.md#embedding-sharing`](model.md#embedding-sharing).

## Checkpointing and Export

Recommended during training:

- keep `fsdp_state_dict_type: SHARDED_STATE_DICT`

For reliable artifact export, run `deberta export` after training:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python deberta export \
  <RUN_DIR>/checkpoint-<STEP> \
  --output-dir <RUN_DIR>/exported_hf \
  --what discriminator
```

The exporter consolidates to full state on rank 0 and writes standalone HF artifacts.
Legacy compiled checkpoints that include `._orig_mod` key segments are remapped
automatically during export, so older wrapper artifacts do not block consolidation.

Run directories now include `run_metadata.json` with a `config_schema_version`. Resume/export validate that schema and fail fast on unknown versions instead of silently proceeding with ambiguous config metadata.
Run directories also persist:

- `config_original.yaml` (source config snapshot when provided; otherwise current resolved config baseline)
- `config_resolved.yaml` (fully resolved model/data/train payload used at runtime)

When `train.report_to=wandb`, `config_original.yaml` is uploaded once at startup using
the filename pattern `config_deberta_<run_name>`.
