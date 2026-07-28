# Distributed training

## Current support boundary

Multi-GPU training is not supported. The repository retains FSDP2 launch templates and partial unit coverage for future work, but there is no end-to-end test demonstrating that decoupled RTD phases, GDES synchronization, checkpoint save/resume, `torch.compile`, and FlashDeBERTa work together across ranks. Do not use these templates for a long-running job until that validation lands in a dedicated follow-up.

Automatic `train.checkpoint.export_hf_final` launches a plain single-process export subprocess. It does not recreate the training job's distributed/FSDP2 topology, so loading a sharded checkpoint through that path is unverified and may fail after training completes. Any experimental multi-GPU run must set `train.checkpoint.export_hf_final=false` and treat [manual checkpoint consolidation](../guides/exporting-models.md#experimental-fsdp-checkpoint-consolidation) as part of validating the target topology.

Crash-time checkpoint saving is also intentionally skipped after a distributed rank failure because the remaining ranks may be unable to complete checkpoint collectives. The last successfully completed collective checkpoint is the recovery boundary.

## Launch with Accelerate

Experimental single-node FSDP2 templates:

- native `hf_deberta_v2`: [`fsdp2_hf_deberta_1node.yaml`](../../configs/accelerate/fsdp2_hf_deberta_1node.yaml)
- experimental `rope`: [`fsdp2_1node.yaml`](../../configs/accelerate/fsdp2_1node.yaml)

Both templates set `num_processes: 8`; change it to the number of GPUs on the node. They are scaffolding for the follow-up validation work, not a supported recipe.

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_deberta_v3_bespoke_100k.yaml
```

## FSDP version handling

The provided experimental configs target FSDP2 (`fsdp_version: 2`).

FSDP1 is not maintained. Future multi-GPU validation will target FSDP2 only.

## FlashDeBERTa with compile

The code has intended wrapping boundaries for FSDP2, `torch.compile`, and FlashDeBERTa, but compatibility has not been established by an end-to-end multi-GPU run. Required follow-up coverage includes both RTD phase backward passes, GDES synchronization after generator steps and resume, route stability on every rank, collective checkpoint creation and restore, and standalone export of the resulting sharded checkpoint. Route and kernel behavior are described in [FlashDeBERTa attention](flash-attention.md).
