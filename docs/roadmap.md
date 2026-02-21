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
- vectorize whole-word n-gram masking (`mlm_max_ngram > 1`) to remove per-sample Python retry loops that can bottleneck long-context CPU dataloading
- clarify and normalize `data.shuffle_buffer_size` semantics for non-streaming/map-style datasets (`>0` currently behaves as a boolean enable, so buffer magnitude is inert)

## Runtime/Compile

- refine torch.compile boundaries so generator/discriminator backbones can stay compiled while RTD sampling/corruption logic runs eagerly (to avoid dynamic-op graph-break overhead)
- reject or explicitly no-op `train.sdpa_kernel` when `model.backbone_type='rope'` and `model.attention_implementation='eager'` (kernel policy currently accepted even though eager attention bypasses SDPA backend selection)
- add run-level schema/version metadata for persisted config snapshots so export/resume can detect config-schema drift explicitly instead of relying on dataclass defaults

## Model Perf

- investigate active-token-only projection paths for heavily padded 2D batches (skip attention/FFN projection FLOPs on known-dead pad positions rather than zeroing outputs post projection)
