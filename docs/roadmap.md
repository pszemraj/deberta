# Roadmap

Deferred follow-ups. See [model](model.md), [data](data.md), [objective](objective.md), [runtime](fsdp2.md) for current behavior.

## CLI and UX

- make constrained CLI choice parsing case-insensitive so CLI behavior matches config-file normalization
- improve help output for mutually exclusive boolean flag pairs so default rendering is less ambiguous

## Data/Packing

- evaluate checkpointing packer worker/buffer state (not just consumed micro-batch count) for stricter resume determinism across worker/process layouts
- ~~replace collator-time dense document attention mask materialization with compact doc-boundary representation and on-device block masking~~ — **resolved**: collator now emits `doc_ids (B,S)` and the training loop builds the 3D mask on-device; future: evaluate `flex_attention` block-sparse path to avoid dense `(B,S,S)` materialization entirely
- refactor token-weighted GA to avoid whole-window microbatch buffering so large per-batch metadata does not scale host memory linearly with `gradient_accumulation_steps`
- replace whole-word n-gram retry-loop masking (`mlm_max_ngram > 1`) with a deterministic linear-time candidate walk (and/or vectorized path) to avoid worst-case per-sample retry spin on long sequences

## Runtime/Compile

- run the native HF compile decision matrix (`backbones` control vs `ffn` candidate across gdes/none/es, plus 512 NaN gate) and promote/remove fallback based on quality+throughput gates
- upstream a minimal reproducible native-HF attention inductor drift case using `local-scratch/hf_attention_inductor_repro.py` with `backend=aot_eager` control and `model.hf_attention_kernel` variants
- for rope backbone compile hardening, evaluate replacing custom `RMSNorm` with `torch.nn.RMSNorm` and compare convergence/runtime
- for rope backbone compile hardening, evaluate static/sliced RoPE cache buffers to remove compile-time cache-build branching

## Model Perf

- investigate active-token-only projection paths for heavily padded 2D batches (skip attention/FFN projection FLOPs on known-dead pad positions rather than zeroing outputs post projection)
- ~~for HF DeBERTa-v2 backbone: evaluate lazy or on-demand relative-position table construction to avoid O(max_len²) init-time allocation~~ — **resolved**: removed upfront `_rel_pos_table` buffer; all kernels now call `build_relative_position()` on-device per forward (cheap, compile-safe)
