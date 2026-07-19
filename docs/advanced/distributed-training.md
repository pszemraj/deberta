# Distributed training

## Launch with Accelerate

Single-node FSDP2 templates:

- native `hf_deberta_v2`: [`fsdp2_hf_deberta_1node.yaml`](../../configs/accelerate/fsdp2_hf_deberta_1node.yaml)
- experimental `rope`: [`fsdp2_1node.yaml`](../../configs/accelerate/fsdp2_1node.yaml)

Both templates set `num_processes: 8`; change it to the number of GPUs on the node. See the
[Quickstart](../getting-started/quickstart.md#2-choose-a-pretraining-recipe) for a launch
command.

## FSDP version handling

The provided configs target FSDP2 (`fsdp_version: 2`).

FSDP1 is not maintained as a first-class path in this repo; use FSDP2 configs for supported behavior.

## FlashDeBERTa with compile

FSDP2, `torch.compile`, and FlashDeBERTa use compatible wrapping boundaries, but the combined multi-GPU path does not yet have an end-to-end regression test. Run the target distributed configuration before relying on this combination for a long job. Route and kernel behavior are described in [FlashDeBERTa attention](flash-attention.md).
