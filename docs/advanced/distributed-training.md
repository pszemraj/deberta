# Distributed training

## Launch with Accelerate

Single-node FSDP2 templates:

- native `hf_deberta_v2`: [`fsdp2_hf_deberta_1node.yaml`](../../configs/accelerate/fsdp2_hf_deberta_1node.yaml)
- experimental `rope`: [`fsdp2_1node.yaml`](../../configs/accelerate/fsdp2_1node.yaml)

Both templates set `num_processes: 8`; change it to the number of GPUs on the node. See the
[Quickstart](../getting-started/quickstart.md#2-small-parity-style-run-on-fineweb-edu) for a launch
command.

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

Run snapshot files are described in the
[config reference](../../configs/config-reference.yaml).

`metrics.jsonl.gz` is written when `logging.debug.metrics=true`. Crash rows are appended to the same file on abnormal exit regardless of that setting.

When `logging.wandb.enabled=true`, tracker config and snapshot files are uploaded at startup from the main process.

## Training runtime module layout

Runtime internals live under [`src/deberta/training/`](../../src/deberta/training/).
Public entrypoints and signatures are listed in [API / Training](../api/training.md).
