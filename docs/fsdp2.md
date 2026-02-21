# FSDP2, Runtime, and Export

This document is the primary reference for distributed training setup and runtime knobs.

For model/backbone options, see [`docs/model.md`](model.md). For input pipeline behavior, see [`docs/data.md`](data.md). For RTD objective/loss details, see [`docs/objective.md`](objective.md).

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

## TF32 Policy

`train.tf32=true` by default.

The training loop prefers modern PyTorch fp32 precision controls when available and falls back to legacy `allow_tf32` flags on older builds.

For `torch.compile` max-autotune modes, legacy TF32 flags may be forced for compatibility.

## `torch.compile`

Enable with:

- `train.torch_compile=true`
- `train.torch_compile_mode` in:
  - `default`
  - `reduce-overhead`
  - `max-autotune`
  - `max-autotune-no-cudagraphs`

If compile fails at runtime, training logs a warning and continues without compile.

## SDPA Kernel Policy

Use `train.sdpa_kernel` to set SDPA backend preference:

- `auto` (default)
- `flash`
- `mem_efficient`
- `math`
- `flash_only`

This is best-effort backend configuration. `flash_only` can fail if hardware or tensor shapes are not flash-compatible.

When `data.pack_sequences=true`, packed batches may emit 3D document-blocking attention masks. Those masks are not compatible with flash-only SDPA kernels, so `train.sdpa_kernel=flash_only` is rejected by config validation for that workflow.

Use `auto`, `flash`, or `mem_efficient` for packed training runs.

## Embedding Sharing and FSDP Safety

`embedding_sharing` supports `none|es|gdes`.

To stay FSDP-safe, sharing is implemented without reusing module instances across generator/discriminator; it ties weights via lightweight adapters to avoid problematic shared-module wrapping.

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
