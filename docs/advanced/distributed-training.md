# Distributed training

## Launch with Accelerate

Single-node templates:

- `configs/accelerate/fsdp2_hf_deberta_1node.yaml`
- `configs/accelerate/fsdp2_1node.yaml`

Example:

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_base.yaml
```

## FSDP version handling

The provided configs target FSDP2 (`fsdp_version: 2`).

FSDP1 is not maintained as a first-class path in this repo; use FSDP2 configs for supported behavior.

## Token-weighted gradient accumulation

`train.token_weighted_gradient_accumulation=true` scales per-microbatch objectives by active token counts across accumulation windows. This avoids unequal weighting when token counts vary between microbatches/ranks.

## Resume data alignment

`train.checkpoint.resume_data_strategy` controls iterator alignment after checkpoint resume:

- `replay`
- `restart_epoch`
- `auto`

`train.checkpoint.resume_replay_max_micro_batches` gates when `auto` switches from replay to restart.

## Runtime snapshots and tracking

Run snapshot files are described in [Guides / Configuration](../guides/configuration.md#snapshot-files-and-reproducibility).

`metrics.jsonl.gz` is written when `train.debug_metrics=true` (plus crash rows when training exits via an exception).

When `logging.wandb.enabled=true`, tracker config and snapshot files are uploaded at startup from the main process.

## Training runtime module layout

Training internals are split across:

- [`src/deberta/training/entrypoint.py`](../../src/deberta/training/entrypoint.py): orchestration and top-level train/dry-run entrypoints
- [`src/deberta/training/runtime.py`](../../src/deberta/training/runtime.py): optimizer/scheduler/dataloader/runtime builders
- [`src/deberta/training/steps.py`](../../src/deberta/training/steps.py): per-window step helpers, token-weighted window accounting, non-finite recovery utilities
- [`src/deberta/training/checkpointing.py`](../../src/deberta/training/checkpointing.py): periodic checkpoint/save-progress and resume-progress normalization helpers
- [`src/deberta/training/metrics.py`](../../src/deberta/training/metrics.py): tracker payload coercion, JSONL appends, non-finite debug artifacts
- [`src/deberta/training/compile.py`](../../src/deberta/training/compile.py): compile-scope resolution, TF32/SDPA setup, compile-target helpers
