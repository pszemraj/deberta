# Roadmap

Deferred follow-ups. See [model](model.md), [data](data.md), [objective](objective.md), [runtime](fsdp2.md) for current behavior.

## CLI and UX

- make constrained CLI choice parsing case-insensitive so CLI behavior matches config-file normalization
- improve help output for mutually exclusive boolean flag pairs so default rendering is less ambiguous

## Data/Packing

- implement bit-identical resume mode for packed streaming by checkpointing iterator cursor, packer buffer, and worker RNG state (not just consumed micro-batch count)
- ~~replace collator-time dense document attention mask materialization with compact doc-boundary representation and on-device block masking~~ — **resolved**: collator now emits `doc_ids (B,S)` and the training loop builds the 3D mask on-device; future: evaluate `flex_attention` block-sparse path to avoid dense `(B,S,S)` materialization entirely
- refactor token-weighted GA to avoid whole-window microbatch buffering so large per-batch metadata does not scale host memory linearly with `gradient_accumulation_steps`
- replace whole-word n-gram retry-loop masking (`mlm_max_ngram > 1`) with a deterministic linear-time candidate walk (and/or vectorized path) to avoid worst-case per-sample retry spin on long sequences
- add optional strict whole-word n-gram masking mode that disallows token-level tail fill when n-gram span sampling under-fills budget

## Runtime/Compile

- run the native HF compile decision matrix (`backbones` control vs `ffn` candidate across gdes/none/es, plus 512 NaN gate) and promote/remove fallback based on quality+throughput gates
- upstream a minimal reproducible native-HF attention inductor drift case using `local-scratch/hf_attention_inductor_repro.py` with `backend=aot_eager` control and `model.hf_attention_kernel` variants
- for rope backbone compile hardening, evaluate replacing custom `RMSNorm` with `torch.nn.RMSNorm` and compare convergence/runtime
- for rope backbone compile hardening, evaluate static/sliced RoPE cache buffers to remove compile-time cache-build branching
- ~~auto compile scope + doc-blocking mask shape churn~~ — **resolved**: `_resolve_compile_scope` auto-downgrades to FFN when `block_cross_document_attention=True` to avoid alternating None/3D mask shapes under compile

## HF DeBERTa-v2 Modernization

Parity gaps vs the RoPE path that matter for fair head-to-head comparison:

- optional RMSNorm support (currently LayerNorm only) — ~10-15% faster, different convergence profile
- optional SwiGLU FFN support (currently MLP-only; SwiGLU is ~2/3 the parameters for equivalent capacity)
- configurable `use_bias` (currently hardcoded per layer; RoPE path defaults to bias-free)

These do not affect disentangled attention or RTD correctness; they are efficiency/parity items.

## Model Perf

- investigate active-token-only projection paths for heavily padded 2D batches (skip attention/FFN projection FLOPs on known-dead pad positions rather than zeroing outputs post projection)
- ~~for HF DeBERTa-v2 backbone: evaluate lazy or on-demand relative-position table construction to avoid O(max_len²) init-time allocation~~ — **resolved**: removed upfront `_rel_pos_table` buffer; all kernels now call `build_relative_position()` on-device per forward (cheap, compile-safe)
- ~~HF DeBERTa-v2 2D→(B,1,S,S) padding mask expansion~~ — **resolved**: `get_attention_mask` now returns `(B,1,1,S)` broadcast mask for 2D padding inputs, avoiding O(S²) outer product

## Export Interop

- add a RoPE export codegen path that packages generated modeling/config files plus `auto_map` metadata so `transformers.AutoModel.from_pretrained(...)` works without manual custom-class imports
