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
- Removed external `*_config_name_or_path` indirection for backbone configs. Architecture/config sourcing now uses `*_model_name_or_path` plus explicit `model.hf_max_position_embeddings` for HF context overrides: [`src/deberta/config.py`](../src/deberta/config.py), [`src/deberta/modeling/builder.py`](../src/deberta/modeling/builder.py)
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

## HF DeBERTa-v2 Modernization Plan (PyTorch 2.9.1+)

The branch history already records failed attempts to run full-backbone inductor compile on native HFv2 (`75b4812`, `3878736`, `9d46654`, `829b6ad`) and the current stable fallback (`auto -> ffn`).

Goal: improve HFv2 throughput/memory while keeping DeBERTa-v2 disentangled attention semantics and the v3 RTD+GDES training objective unchanged.

### Phase 0: Baseline + Gates

- [ ] Keep `tools/compile_drift_probe.py` as the mandatory stability gate for any HFv2 compile/kernel change and add fixed presets for `ffn`, `encoder`, and `backbones`.
- [ ] Add a repeatable HFv2 vs RoPE train-step throughput harness (single-GPU synthetic + one real config slice) and record baseline tokens/sec + peak memory in CI artifacts.
- [ ] Define hard gates before merge:
  - no train-mode catastrophic drift vs eager baseline in probe runs
  - no non-finite spike regression in short RTD training runs
  - measurable speedup and memory reduction on HFv2 path

### Phase 1: Safe Compile Expansion (No Attention Compile)

- [ ] Keep default `train.torch_compile_scope=auto -> ffn` for HFv2 until all gates pass.
- [ ] Add an explicit HFv2-safe scope that compiles only non-attention blocks (FFN + selected head/projection modules), leaving `DisentangledSelfAttention` eager.
- [ ] Add scope-level tests that prove routing is deterministic and rejects unsupported combinations.

### Phase 2: Eager Attention Modernization (Architecture-Preserving)

- [ ] Add a fused QKV projection fast path for the common `query_states is hidden_states` case to reduce projection kernel launches.
- [ ] Cache relative-position gather indices by `(query_len, key_len, device)` for cached kernels to reduce per-step index construction overhead.
- [ ] Introduce a numerics policy for HFv2 attention:
  - default: stable mixed path (matmul in runtime dtype, bias/softmax accumulation in fp32)
  - fallback: full fp32 score path for pathological runs
- [ ] Benchmark `dynamic`, `cached_bmm`, and `stable` under the same harness and keep only variants that pass drift + performance gates.

### Phase 3: SDPA-Class Backend Exploration (Opt-In)

- [ ] Prototype a PyTorch 2.9.1+ `flex_attention` backend for HFv2 that preserves C2P/P2C bias semantics exactly.
- [ ] Keep the backend experimental and opt-in (`model.hf_attention_kernel=flex_experimental`) until parity and drift gates are green.
- [ ] Validate parity against current stable kernel with fixed-seed forward/backward checks before any training run.

### Phase 4: Rollout Criteria

- [ ] Promote new backend/scope defaults only after:
  - compile drift probe remains stable across warmup + post-warmup windows
  - end-to-end RTD loss curves match baseline behavior (no late divergence regime shift)
  - HFv2 recovers a meaningful portion of the current gap to RoPE on 512/1024 contexts
- [ ] Keep fallback controls documented and tested (`hf_attention_kernel=stable`, `torch_compile_scope=ffn`) for immediate rollback.

## TODOs

- Add an export-time codegen path for RoPE checkpoints that starts from official HF DeBERTa modeling/configuration sources, applies this repo's RoPE/RMSNorm/SwiGLU diffs, and writes generated modeling files into the export directory with `auto_map` metadata so `AutoModel.from_pretrained(...)` can work without manual custom-code setup.
