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

## Data/Packing

- evaluate checkpointing packer worker/buffer state (not just consumed micro-batch count) for stricter resume determinism across worker/process layouts
- replace collator-time dense document attention mask materialization (`B x S x S`) with a compact doc-boundary representation and on-device block masking to reduce CPU bottlenecks at long context lengths
