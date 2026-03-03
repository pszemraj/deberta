# FSDP2, Runtime, and Export

See also: [model/backbone](model.md), [source resolution](model.md#source-resolution-contract), [data pipeline](data.md), [RTD objective](objective.md).

## Accelerate + FSDP2

This project targets PyTorch 2.9.1+ and uses Hugging Face Accelerate with:

- `distributed_type: FSDP`
- `fsdp_version: 2`

Provided configs:

- RoPE backbone: [`configs/accelerate/fsdp2_1node.yaml`](../configs/accelerate/fsdp2_1node.yaml)
  - wrap class: `DebertaRoPELayer`
- HF DeBERTa compatibility path: [`configs/accelerate/fsdp2_hf_deberta_1node.yaml`](../configs/accelerate/fsdp2_hf_deberta_1node.yaml)
  - wrap class: `DebertaV2Layer`

## Precision Defaults

Training defaults to mixed precision `bf16` autocast (`train.mixed_precision=bf16`).

This is autocast-based mixed precision, not full-parameter bf16 casting.

Runtime behavior:

- if CUDA bf16 is unavailable or a tiny bf16 preflight matmul fails, training fails fast with an explicit runtime error
- `train.mixed_precision` accepts `bf16` or `no`
- to run in full precision, set `train.mixed_precision=no` explicitly

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

The training loop configures TF32 using legacy backend flags:

- `torch.backends.cuda.matmul.allow_tf32`
- `torch.backends.cudnn.allow_tf32`

## Resume Data Alignment

When resuming from checkpoints, data-iterator alignment is controlled by:

- `train.resume_data_strategy=auto|replay|restart_epoch`
- `train.resume_replay_max_micro_batches` (used by `auto`)

Behavior:

- `replay`: replays consumed microbatches exactly (deterministic, resume latency scales with replay length)
- `restart_epoch`: skips replay and starts iterator at `start_epoch=global_step` (O(1) resume, data order may differ)
- `auto`: replays only when `consumed_micro_batches <= resume_replay_max_micro_batches`; otherwise switches to `restart_epoch`

Checkpoint metadata in `data_state.json` stores:

- `consumed_micro_batches`
- `lr_mult` (persistent non-finite recovery multiplier)
- `optimizer_param_digest` (single digest for coupled mode; generator/discriminator digests for decoupled mode)
- `global_step`
- `gradient_accumulation_steps`

Resume fails fast when this file is missing; approximate replay offsets are not used.

## `torch.compile`

Enable with `train.torch_compile=true` and `train.torch_compile_mode` in `default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`.

If `train.torch_compile=true`, compile failures are treated as hard errors (no silent eager fallback).

RTD wrapper glue (sampling/corruption/label construction) remains eager by design.

Default compile scope is resolved from `train.torch_compile_scope=auto`:

- baseline: compile both `generator` and `discriminator` backbones
- exception: for `model.backbone_type=rope` with `data.block_cross_document_attention=true`, auto scope downgrades to `ffn` to avoid mask-shape churn recompiles

Recommended HFv2 compile path:

- `train.torch_compile_mode=default`
- `train.torch_compile_backend=inductor`
- `train.torch_compile_scope=backbones` (or `auto`)

Compile behavior is configured directly via train/model config:

- `train.torch_compile_scope=auto|backbones|encoder|gen_encoder|disc_encoder|ffn|gen_ffn|disc_ffn`
- `train.torch_compile_backend=inductor|aot_eager`

Native HF attention-kernel variants (`model.hf_attention_kernel`) are defined in [HF DeBERTa-v2 backbone](model.md#hf-deberta-v2-backbone).
`dynamic` and `stable` are both supported for compiled native HF runs.

Compile is applied to module `forward` callables (module identity is preserved),
so checkpoints keep canonical state-dict keys. Resume still normalizes
compile-wrapper key segments (`._orig_mod`) when needed for wrapper-produced
checkpoints.

### Compile Parity Protocol

Use `local-scratch/compile_parity_check.py` for eager-vs-compiled parity checks on the same weights, batch, and RNG state.

- default gate: `--eval` mode (deterministic parity threshold is enforced)
- optional diagnostic: `--train` mode (informational deltas only)
- compile bisect scopes: `--scope gen|disc|both|matrix`
- backend isolation: `--backend inductor|aot_eager`

Use `tools/compile_drift_probe.py` when parity is inconclusive and train-mode drift appears near/after warmup:

- compares eager vs compiled train steps from identical initial weights
- reports per-step loss delta + compact parameter drift
- prints best-effort Dynamo counters (`stats` / `recompiles`) for graph-churn diagnosis

Use `local-scratch/hf_attention_inductor_repro.py` for a minimal one-layer train-step repro focused on native HF attention internals.
Use `local-scratch/compile_log_summary.py` to convert `wandb/*/files/output.log` into final/min/max/checkpoint/runtime summary tables.

### Non-Finite Diagnostics

Training keeps strict scalar checks and non-finite recovery behavior for gradient overflows.

- checks run before backward (`gen_loss_raw`, `disc_loss_raw`, forward/backward scalar objectives)
- checks run before optimizer step (global gradient norm, and post-clip gradient norm when clipping is enabled)
- when non-finite gradients are detected, the entire accumulation window is skipped (no partial/sanitized optimizer step)
- recovery is then applied (LR backoff and periodic optimizer-state reset) before continuing
- skipped non-finite windows are logged as `nonfinite_window_skipped=1` with `nonfinite_skip_total`, `nonfinite_skip_streak`, and recovery metrics
- when triggered, a compact debug artifact is written to:
  - `<output_dir>/debug/nonfinite_step_<STEP>_<TAG>.json`
  - includes step/lr, compile mode, embedding sharing mode, scalar snapshots, and compact RNG state heads

## Training Metrics Logging

Step logs and tracker metrics prioritize signal over mostly-constant counters.

- primary scalar metrics: `loss`, `gen_loss`, `disc_loss`, `disc_acc`, `lr`
- throughput/scale metrics: `input_tokens_per_sec`, `input_tokens_seen`
- noisy low-information counters (for example per-step `gen_tok`/`disc_tok`) are intentionally excluded from periodic logs
- token-weighted GA zero-window counters (`zero_*`) are never sent to trackers/W&B; they are written only to
  `<output_dir>/metrics.jsonl.gz` when either `train.debug_metrics=true` or `DEBERTA_DEBUG=1`

## Interruption and Crash Handling

`deberta train` uses a crash-safe shutdown path:

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

When `data.pack_sequences=true` and `data.block_cross_document_attention=true`, packed batches may emit the pairwise `(B, S, S)` keep-mask defined in the [doc-blocking contract](data.md#doc-blocking-contract-block_cross_document_attentiontrue). That mask path is incompatible with strict flash SDPA kernels, so `train.sdpa_kernel=flash` is rejected by config validation for that workflow.

Use `auto`, `mem_efficient`, or `math` for packed+doc-block training runs.

## Checkpointing and Export

Recommended during training:

- keep `fsdp_state_dict_type: SHARDED_STATE_DICT`

For reliable artifact export, run `deberta export` after training:

```bash
accelerate launch --config_file configs/accelerate/fsdp2_1node.yaml --no_python deberta export \
  <RUN_DIR>/checkpoint-<STEP> \
  --output-dir <RUN_DIR>/exported_hf \
  --what discriminator
```

The exporter consolidates to full state on rank 0 and writes standalone HF artifacts.
Legacy compiled checkpoints that include `._orig_mod` key segments are remapped
automatically during export, so older wrapper artifacts do not block consolidation.

RoPE export loading behavior is documented in [model export interoperability](model.md#export-interoperability-rope).

### In-Training HF Export

`train.export_hf_final=true` (default) exports the discriminator into `<output_dir>/final_hf` at the end of training by invoking `deberta export` in a separate subprocess from the last saved checkpoint.
This path uses `--allow-partial-export` so final export stays best-effort instead of failing on non-critical key drift.

This keeps the training process isolated from export-time teardown issues while still producing a final artifact automatically.

For full control and reproducibility, you can still run `deberta export` manually after training.

The `deberta train` CLI also uses a fast process exit on successful completion by default (`DEBERTA_FAST_EXIT_AFTER_TRAIN=1`) to avoid interpreter-finalization crashes seen in some CUDA extension stacks. Set `DEBERTA_FAST_EXIT_AFTER_TRAIN=0` to use normal interpreter shutdown behavior.

Run directories include `run_metadata.json` with a `config_schema_version`.
Resume/export validate that schema and fail fast on unknown versions instead of
silently proceeding with ambiguous config metadata.
Run directories also persist:

- `config_original.yaml` (source config snapshot when provided; otherwise current resolved config baseline)
- `config_resolved.yaml` (fully resolved model/data/train payload used at runtime)

When `train.report_to=wandb`, config snapshots are uploaded once at startup:

- `config_original_deberta_<run_name>.yaml`
- `config_resolved_deberta_<run_name>.yaml` (when present)
- `config_source_deberta_<run_name>.<ext>` (when a source config path is available)
