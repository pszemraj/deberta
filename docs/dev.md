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
- Floating-point attention masks are rejected at runtime. Callers must pass bool/integer keep masks to avoid ambiguous semantics (`0/1` keep masks vs additive `0/-inf` masks).

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
- When compile scope includes RoPE attention (`backbones`/`encoder`), rotary cos/sin caches are prefilled before `torch.compile` and compiled forwards use slice-only access. Missing/undersized compile-time caches now fail fast.

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

## Correctness Audit Notes

Full pipeline audit covering forward pass faithfulness, loss computation, attention masks, numerical stability, data pipeline, config fidelity, and checkpoint roundtrip. No critical or high-severity issues found.

### RTD Objective Correctness (verified)

- Generator loss: restricted to masked positions only; CLS/SEP/PAD excluded via `special_tokens_mask` + `maskable` filter in collator.
- Discriminator loss: computed over ALL active non-padding tokens (not just replaced ones) — the classic ELECTRA-bug does not apply here.
- Edge case: when the generator samples the original token, discriminator label is correctly 0 (not replaced).
- Sampling: Gumbel-max stochastic sampling, not argmax — corruption is non-trivial.
- Gradient isolation: `torch.no_grad()` wraps sampling/corruption/label construction; discriminator cannot backprop through generator sampling.
- GDES: `base_weight` is `requires_grad=False`, sync uses `.detach()` + `.copy_()` under `@torch.no_grad()`, runs after `optimizer.step()`.
- Loss precision: both `gen_logits.float()` and `disc_logits.float()` enforce fp32 for CE/BCE.

### Numerical Stability (verified)

- RMSNorm: eps=1e-6, computation upcast to fp32 internally ([`modeling/norm.py`](../src/deberta/modeling/norm.py)).
- HFv2 LayerNorm: eps=1e-7 (HF default), safe because `nn.LayerNorm` CUDA kernel upcasts internally.
- Both backbones upcast Q/K to fp32 before attention score computation (HFv2 disentangled and RoPE eager paths).
- RoPE SDPA path: delegates precision to PyTorch fused kernels.
- Gradient checkpointing: at full-layer boundaries only, no fused-op bisection.
- No embedding scaling applied in either backbone — no double-scaling risk.

### Known Limitations / Deferred Improvements

1. **Resume data replay is approximate, not exact** — `PackedStreamingDataset.buffer` (partial document tokens) is not checkpointed. After resume + replay, the internal packing buffer starts empty, causing slight data order divergence at the resume boundary. The `restart_epoch` strategy diverges completely by design.
2. **Nonfinite recovery streak not checkpointed** — `nonfinite_skip_streak` and `nonfinite_skip_total` reset to 0 on resume. If training was mid-recovery (e.g., streak=3 of 4 before optimizer state reset), the streak restarts. `lr_mult` IS checkpointed (the more important state).
3. **`train_config.json` not validated on resume** — `_persist_or_validate_run_configs()` validates model and data configs but intentionally skips train config, allowing hyperparameter changes between runs. Architecture-adjacent training params (`gradient_accumulation_steps`, `mixed_precision`) can change silently.
4. **N-gram masking fallback fill is subword-level** — When the n-gram budget loop does not reach `num_to_mask`, the tail fill samples individual subword positions (not whole-word groups). Empirically verified at 0% activation rate across 500 packed multi-doc trials per length at 1024/2048/4096 context with the wordpiece-32k tokenizer and SmolLM2 stage-4 data (12.2% continuation-token density). The retry budget `max(50, 10 * len(word_groups))` scales with sequence length and comfortably exceeds the mask budget at all target context lengths. TODO: add optional strict WWM mode behind a config flag if needed for ablation studies (see [`roadmap.md`](roadmap.md)).
5. **Partial export is now explicit opt-in** — `deberta export` now rejects partial backbone state loads by default. `--allow-partial-export` re-enables best-effort intersection loads and should only be used for intentionally partial recovery workflows.
6. **Nonfinite recovery constants not configurable** — `_NONFINITE_LR_BACKOFF`, `_NONFINITE_LR_MULT_FLOOR`, `_NONFINITE_LR_MULT_RECOVERY`, `_NONFINITE_OPT_STATE_RESET_EVERY` are module-level constants in [`pretrain.py`](../src/deberta/training/pretrain.py), not exposed via YAML.
7. **`_EXPORT_CONFIG_STRIP_KEYS` duplicated** — Identical lists in [`export_cli.py`](../src/deberta/export_cli.py) and [`pretrain.py`](../src/deberta/training/pretrain.py). Should be consolidated to a single source of truth.
8. **HFv2 attention mask fill value is `-1e4`** — Deliberate stability choice (avoids `inf * 0 = NaN` in gradients). Safe for typical sequence lengths; RoPE's `torch.finfo(dtype).min` approach is more robust but both are correct.
9. **Flash SDPA + doc-blocking incompatibility** — Enforced by config validation (`validate_training_workflow_options`), not runtime assertion. If validation is bypassed, PyTorch's SDPA dispatch would silently fall back to a non-flash kernel.
10. **KEEL topology has no separate final output norm** — The last layer's `outer_norm2` serves this role. If KEEL is meant to match pure pre-norm + final-norm behavior exactly, a dedicated final norm would be needed. Current behavior is internally consistent.
11. **Adam epsilon silently raised to 1e-6 under bf16** — `_build_optimizer()` overrides `adam_epsilon < 1e-6` to `1e-6` for bf16 stability. Logged but could surprise hyperparameter sweeps targeting very small epsilon values.
12. **Dense doc-blocking is O(S^2)** — `_build_doc_block_mask()` materializes `(B,S,S)` keep masks. Config validation now warns when `data.pack_sequences=true`, `data.block_cross_document_attention=true`, and `data.max_seq_length > 2048`, but this remains expensive for long contexts.
13. **N-gram masking implementation is Python-heavy** — `_mask_tokens_ngram()` still relies on per-sample token-string conversion and loop-based span assembly. Correctness is acceptable, but throughput can become dataloader-bound at scale.
14. **Grad-norm telemetry computes full-model norms every sync step** — this is intentional for now to maximize non-finite detection, but it adds per-step overhead and should become configurable once default stability policy is finalized.
15. **`suppress(Exception)` usage is intentionally broad in non-critical paths** — useful for crash-safe logging/cleanup, but a follow-up pass should narrow exception scopes and add explicit one-shot warnings where silent fallback is still too opaque.
16. **HF DeBERTa-v2 `pos_att_type=p2p` branch is still deferred** — current disentangled attention parity work closed high/medium-impact RTD and training gaps first. Add full `p2p` scoring branch plus scale-factor parity before calling backbone parity complete.
17. **Relative-position log-bucketing clamp parity is still deferred** — `_make_log_bucket_position` should clamp raw relative offsets to `[-max_position+1, max_position-1]` before bucketing to match Microsoft reference edge behavior at long context.

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
- Implement segment-aware/sparse doc-blocking (or FlexAttention varlen equivalent) so packed long-context runs do not require dense `(B,S,S)` masks.
- Vectorize n-gram masking setup by replacing per-batch `convert_ids_to_tokens()` calls with precomputed continuation-token metadata.
- Add configurable grad-norm/check cadence (`always`, `on_clip`, `every_n_steps`) with default preserving current safety-first behavior.
- Audit `suppress(Exception)` callsites and narrow to expected exception types in resume/export correctness paths; keep best-effort behavior only for optional telemetry/cleanup.
- Implement `pos_att_type=p2p` in `DisentangledSelfAttention` and add a fixed-seed parity unit test against the reference branch math.
- Clamp relative offsets before log-bucket conversion in HFv2 native attention utilities and add long-context edge-case tests.
