# Developer Notes

> THIS DOCUMENT IS STANDALONE FOR TRACKING DEV RELATED WORK.
> DO NOT DELETE THIS FILE OR MERGE ITS CONTENT INTO OTHER DOCS DURING DOCUMENTATION CONSOLIDATION.
> Keep this file focused on developer/runtime implementation notes and compatibility-break tracking.

## Pre-release Compatibility Policy

Until this repo has a stable release, backward compatibility is not required. Prefer correctness and simplicity over resume/checkpoint compatibility.

## Intentional Compatibility Breaks

- Run metadata schema is version-gated, and schema mismatches fail fast during resume/export: [`src/deberta/config.py`](../src/deberta/config.py)
  - `RUN_CONFIG_SCHEMA_VERSION`
  - `validate_run_metadata_schema(...)`
- Persisted `model_config.json` / `data_config.json` snapshots are parsed with strict key validation. Unknown legacy keys now fail with a clear error instead of being silently coerced: [`src/deberta/config.py`](../src/deberta/config.py)
  - `load_model_config_snapshot(...)`
  - `load_data_config_snapshot(...)`
- Resume path uses strict snapshot loaders before comparing runtime configs: [`src/deberta/training/pretrain.py`](../src/deberta/training/pretrain.py)
  - `_persist_or_validate_run_configs(...)`
- Export path uses the same strict snapshot loaders: [`src/deberta/export_cli.py`](../src/deberta/export_cli.py)
  - `run_export(...)`

Legacy keys removed from current dataclasses include `data.eval_split`, `data.preprocessing_num_workers`, `train.per_device_eval_batch_size`, and `train.eval_steps`.

## Training Progress Bars

- Runtime uses `tqdm.auto.tqdm` on main process for:
  - training-step progress (`train`)
  - resume data-replay progress (`resume-replay`)
- Implemented in [`src/deberta/training/pretrain.py`](../src/deberta/training/pretrain.py).

## Attention Mask Contracts

### RoPE backbone

Standard multi-head attention (no disentangled position bias). Position encoded via RoPE rotation on Q/K.

- No mask (`None`): unpadded/packed batch, no doc-blocking. SDPA receives no mask.
- 2D mask (`B,S`): key-padding mask. SDPA receives `(B,1,1,S)` broadcast via unsqueeze.
- 3D mask (`B,S,S`): doc-blocking pairwise keep-mask. Built on-device from `doc_ids` by `_build_doc_block_mask()`.
  - Diagonal contract: diagonal encodes query activity (`True` for active, `False` for pad/inactive).
  - SDPA safety: inactive queries get one keep edge to CLS key so rows are never all-`False`.

### HF DeBERTa-v2 backbone

Disentangled attention adds C2P + P2C relative-position bias via `pos_key_proj` and `pos_query_proj`. Position encoded via learned relative-position embeddings.

- No mask (`None`): unpadded batch. Encoder passes `None` to layers.
- 2D mask (`B,S`): padding mask. `get_attention_mask()` returns `(B,1,1,S)` (not dense `(B,1,S,S)`).
- 3D mask (`B,S,S`): pairwise mask. `get_attention_mask()` returns `(B,1,S,S)`.

### Doc-blocking flow (packed sequences)

1. Collator `_compute_document_ids()` returns compact `doc_ids (B,S)` long tensor (1-based active ids, 0 for pad).
2. Training loop moves `doc_ids` to device with batch tensors.
3. `_build_doc_block_mask(doc_ids)` constructs `(B,S,S)` keep-mask on-device.
4. Model receives standard `attention_mask=(B,S,S)`.

Future: replace dense `(B,S,S)` path with `flex_attention` block-sparse path (see [roadmap](roadmap.md)).

## Compile Scope

- `_resolve_compile_scope()` auto-selects scope based on backbone and data config:
  - HF DeBERTa-v2 + inductor: `auto -> ffn`
  - RoPE + doc-blocking: `auto -> ffn`
  - RoPE without doc-blocking: `auto -> backbones`
- FFN-only compile targets:
  - HF DeBERTa-v2: `encoder.layer[i].intermediate` and `.output`
  - RoPE: `encoder.layers[i].mlp`

## Nonfinite Recovery

- Persistent `lr_mult` multiplier (default `1.0`) applied after each `lr_scheduler.step()`.
- On nonfinite window skip: `lr_mult *= 0.5` with floor at `0.01`.
- On successful optimizer step: `lr_mult *= 1.1` until it reaches `1.0`.
- Every 4 consecutive nonfinite windows: optimizer momentum/variance state is cleared.
- `lr_mult` is saved/restored in `data_state.json`.

## Collator Masking

- Random token replacement samples from `_non_special_token_ids` (excludes pad/cls/sep/mask), using `len(tokenizer)` as effective vocab size.
- `special_tokens_mask` is removed from the batch before return and never transferred to GPU.
- Packed streaming strips leading `[SEP]` tokens from chunks to avoid degenerate `[CLS, SEP, ...]` starts.
- With `embedding_sharing='es'`, divergent `generator_learning_rate` is rejected during config validation.

## Metrics Logging

- `_append_metrics_jsonl_row()` opens/closes gzip per write for crash safety.

## TODOs

- Add an export-time codegen path for RoPE checkpoints that starts from official HF DeBERTa modeling/configuration sources, applies this repo's RoPE/RMSNorm/SwiGLU diffs, and writes generated modeling files into the export directory with `auto_map` metadata so `AutoModel.from_pretrained(...)` can work without manual custom-code setup.
