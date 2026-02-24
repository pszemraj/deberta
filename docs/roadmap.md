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
- refactor token-weighted GA to avoid whole-window microbatch buffering so large per-batch metadata does not scale host memory linearly with `gradient_accumulation_steps`
- replace whole-word n-gram retry-loop masking (`mlm_max_ngram > 1`) with a deterministic linear-time candidate walk (and/or vectorized path) to avoid worst-case per-sample retry spin on long sequences

## Runtime/Compile

- track and document mode/backend-specific full-backbone compile stability for `hf_deberta_v2`, including `inductor` vs `aot_eager` parity deltas and any required temporary boundary fallbacks

## Model Perf

- investigate active-token-only projection paths for heavily padded 2D batches (skip attention/FFN projection FLOPs on known-dead pad positions rather than zeroing outputs post projection)
