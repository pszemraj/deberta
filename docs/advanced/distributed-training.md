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
