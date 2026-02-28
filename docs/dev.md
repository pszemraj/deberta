# Developer Notes

Internal implementation notes that are not part of user-facing runtime docs.

## Training Progress Bars

- Runtime uses `tqdm.auto.tqdm` on main process for:
  - training-step progress (`train`)
  - resume data-replay progress (`resume-replay`)
- Implemented in `src/deberta/training/pretrain.py`.

## Attention Mask Contracts

### RoPE backbone

Standard multi-head attention (no disentangled position bias). Position encoded via RoPE rotation on Q/K.

- **No mask (None)**: unpadded/packed batch, no doc-blocking. SDPA receives no mask — fastest path.
- **2D mask (B,S)**: key-padding mask. SDPA receives `(B,1,1,S)` broadcast via unsqueeze.
- **3D mask (B,S,S)**: doc-blocking pairwise keep-mask. Built on-device from `doc_ids` by `_build_doc_block_mask()`.
  - **Diagonal contract**: diagonal encodes query activity (`True` for active, `False` for pad/inactive).
  - **SDPA safety**: inactive queries get a single keep edge to CLS key so rows are never all-False.
  - Query activity is read from the diagonal in O(B×S) via `torch.diagonal()`, not from row reduction.

### HF DeBERTa-v2 backbone

Disentangled attention: adds C2P + P2C relative-position bias to scores via separate `pos_key_proj`/`pos_query_proj` projections. Position encoded via learned relative-position embeddings.

- **No mask (None)**: unpadded batch. Encoder passes `None` to layers — no mask materialized.
- **2D mask (B,S)**: padding mask. `get_attention_mask()` returns `(B,1,1,S)` broadcast shape (not `(B,1,S,S)` outer product). Avoids O(S²) allocation.
- **3D mask (B,S,S)**: pairwise mask. `get_attention_mask()` returns `(B,1,S,S)` via unsqueeze.

### Doc-blocking flow (packed sequences)

1. Collator `_compute_document_ids()` returns compact `doc_ids (B,S)` long tensor (1-based for active, 0 for pad). No dense mask on CPU.
2. Training loop moves `doc_ids` to device with the rest of the batch.
3. `_build_doc_block_mask(doc_ids)` constructs `(B,S,S)` keep-mask on-device with diagonal-as-activity + CLS safety edge for inactive rows.
4. Model receives standard `attention_mask=(B,S,S)` — interface unchanged.

Future: replace dense `(B,S,S)` with `flex_attention` block-sparse path (see [roadmap](roadmap.md)).

## Compile Scope

- `_resolve_compile_scope()` auto-selects scope based on backbone and data config:
  - **HF DeBERTa-v2 + inductor**: `auto` → `ffn` (attention inductor drift risk).
  - **RoPE + doc-blocking**: `auto` → `ffn` (mask shape churn between None and 3D causes recompilation).
  - **RoPE, no doc-blocking**: `auto` → `backbones` (stable mask contract).
- FFN-only compile supports both encoder layouts:
  - HF DeBERTa-v2: `encoder.layer[i].intermediate` + `.output`
  - RoPE: `encoder.layers[i].mlp`

## Nonfinite Recovery

- Persistent `lr_mult` multiplier (default 1.0) applied after every `lr_scheduler.step()`.
- On nonfinite window skip: `lr_mult *= 0.5` (floor at 1% of scheduled LR).
- On successful optimizer step: `lr_mult *= 1.1` (gradual recovery toward 1.0).
- Every 4 consecutive nonfinite windows: optimizer state (momentum/variance) is cleared.
- `lr_mult` is saved/restored in `data_state.json` alongside `consumed_micro_batches`.

## Collator Masking

- Random token replacement samples from `_non_special_token_ids` (excludes pad/cls/sep/mask), using `len(tokenizer)` as effective vocab size (includes added tokens).
- `special_tokens_mask` is popped from the batch before return — never transferred to GPU.
- Packed streaming strips leading `[SEP]` tokens from chunks to prevent degenerate `[CLS, SEP, ...]` starts.
- With `embedding_sharing='es'`, divergent `generator_learning_rate` is rejected at config validation time.

## Metrics Logging

- `_append_metrics_jsonl_row()` opens/closes gzip per write. This is intentional for crash safety — no dangling file handles on interrupt. Overhead is acceptable vs. losing metrics.

## TODOs

- TODO: add an export-time codegen path for RoPE checkpoints that starts from official HF DeBERTa modeling/configuration sources, applies this repo's RoPE/RMSNorm/SwiGLU diffs, and writes generated modeling files into the export directory with `auto_map` metadata so `AutoModel.from_pretrained(...)` can work without manual custom-code setup.
