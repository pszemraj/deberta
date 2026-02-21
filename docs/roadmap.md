# Roadmap

This file is the single source of truth for deferred documentation-backed follow-ups.

For implemented behavior, use the concept docs:

- model/backbone: [`docs/model.md`](model.md)
- data/packing: [`docs/data.md`](data.md)
- objective/loss: [`docs/objective.md`](objective.md)
- runtime/FSDP/export: [`docs/fsdp2.md`](fsdp2.md)

## CLI and UX

- make constrained CLI choice parsing case-insensitive so CLI behavior matches config-file normalization
- improve help output for mutually exclusive boolean flag pairs so default rendering is less ambiguous

## Model

- add an explicit `swiglu_adjust_intermediate` option for scratch RoPE builds so SwiGLU-vs-MLP ablations can preserve FFN parameter budget (for example 2/3 intermediate scaling)

## Data/Packing

- evaluate checkpointing packer worker/buffer state (not just consumed micro-batch count) for stricter resume determinism across worker/process layouts

## Objective/Loss

- add optional token-count-weighted gradient accumulation across microbatches for workloads with highly variable active-token counts
