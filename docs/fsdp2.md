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

## TF32 Policy

`train.tf32=true` by default.

The training loop prefers modern PyTorch fp32 precision controls when available and falls back to legacy `allow_tf32` flags on older builds.

For `torch.compile` max-autotune modes, legacy TF32 flags may be forced for compatibility.

## `torch.compile`

Enable with `train.torch_compile=true` and `train.torch_compile_mode` in `default|reduce-overhead|max-autotune|max-autotune-no-cudagraphs`.

If compile fails at runtime, training logs a warning and continues without compile.

Current limitation: RTD generator sampling/corruption uses dynamic operations (`multinomial` + indexed token replacement) that can introduce graph breaks. The project keeps compile support as best-effort and tracks compile-boundary refinement as a follow-up in [`docs/roadmap.md`](roadmap.md).

## SDPA Kernel Policy

Use `train.sdpa_kernel` to set SDPA backend preference:

- `auto` (default)
- `flash`
- `mem_efficient`
- `math`
- `flash_only`

This is best-effort backend configuration. `flash_only` can fail if hardware or tensor shapes are not flash-compatible.

`train.sdpa_kernel` is only behaviorally relevant when `model.backbone_type='rope'` and `model.attention_implementation='sdpa'`. For rope eager attention, validation requires `train.sdpa_kernel=auto` to avoid inert config differences.

When `data.pack_sequences=true` and `data.block_cross_document_attention=true`, packed batches may emit the pairwise `(B, S, S)` keep-mask defined in [`docs/data.md#pairwise-mask-contract-block_cross_document_attentiontrue`](data.md#pairwise-mask-contract-block_cross_document_attentiontrue). That mask path is incompatible with flash-only SDPA kernels, so `train.sdpa_kernel=flash_only` is rejected by config validation for that workflow.

Use `auto`, `flash`, or `mem_efficient` for packed training runs.

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

Run directories now include `run_metadata.json` with a `config_schema_version`. Resume/export validate that schema and fail fast on unknown versions instead of silently proceeding with ambiguous config metadata.
