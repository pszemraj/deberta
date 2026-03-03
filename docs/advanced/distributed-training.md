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

`train.resume_data_strategy` controls iterator alignment after checkpoint resume:

- `replay`
- `restart_epoch`
- `auto`

`train.resume_replay_max_micro_batches` gates when `auto` switches from replay to restart.

## Runtime snapshots and tracking

Each run persists config snapshots in output dir and writes metrics to `metrics.jsonl.gz`.

When `train.report_to=wandb`, tracker config and snapshot files are uploaded at startup from the main process.
