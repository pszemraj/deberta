# Architectures

## Backbone types

`model.backbone_type` selects one of two encoder families.

| Setting | `hf_deberta_v2` | `rope` |
|---|---|---|
| Attention | DeBERTa disentangled attention (C2C/C2P/P2C/P2P) | standard QKV + RoPE |
| Position | relative-position buckets/embeddings | rotary embeddings |
| Norm | LayerNorm | RMSNorm |
| FFN | MLP | MLP or SwiGLU |
| Primary use | DeBERTa-v2/v3 parity path | experimental modernized path |

For `hf_deberta_v2`, scratch runs synthesize backbone configs in-repo from `model.hf.model_size` + explicit overrides. Pretrained runs load discriminator config from `model.pretrained.discriminator_path`, and load generator config from `model.pretrained.generator_path` when provided (otherwise generator config is derived from discriminator config).

## HF-size presets (`model.hf.model_size`)

`model.hf.model_size` supports `xsmall`, `small`, `base`, `large`.

Generator defaults are derived from discriminator width/heads/ffn and half depth on `hf_deberta_v2`.

## RTD architecture notes

- discriminator and generator backbones are separate modules
- `model.embedding_sharing` supports `none`, `es`, `gdes`
- decoupled two-phase RTD updates are enabled by default (`train.decoupled_training=true`)

## Default deltas vs released DeBERTa-v3 configs

Intentional repo defaults:

- dropout defaults are `0.0` (`model.dropout.hidden_prob`, `model.dropout.attention_probs_prob`)
- RTD parity++ default `train.objective.disc_loss_weight` is `10.0`
- default `optim.adam.beta2` is `0.999`

Use explicit config values when you want different behavior.

## Parity divergences

- Intentional divergence: generator sampling excludes special/control token ids via a forbidden vocabulary mask (for example PAD/CLS/SEP/MASK). This differs from original DeBERTa sampling and is kept intentionally.
- TODO (strict parity follow-up): evaluate narrowing discriminator embedding sharing to only word+position embeddings; current sharing also includes `token_type_embeddings`.
- TODO (architecture follow-up): evaluate defaulting `generator_intermediate_size` from `generator_hidden_size` when only width is overridden; current behavior inherits discriminator FFN width unless explicitly set.

## FlashDeBERTa follow-ups

- Dense packed `1024` now uses a repo-local local-bias flash path for the common small-batch training regime. Instead of retuning the original fixed disentangled backward kernel directly, the adapter materializes dense DeBERTa relative bias and dispatches through FlashDeBERTa's flash-with-bias kernels when that route is faster on current GPUs.
- Padded varlen now uses dedicated repo-local prefix-pack Triton kernels, including shared pair/triple pack and unpack paths, and the padded backward path now builds packed `grad_out` plus `delta` in one fused step instead of a standalone prefix-pack followed by a separate preprocess kernel.
- Unpacked `1024` now defaults masked batches to the fixed flash path with per-example `seq_lengths`. Sequential microbench and RTD benchmarking showed that route beats the varlen backward kernels at `1024`, while the varlen path regains the advantage at `2048+`.
- Longer padded runs now use measured split backward heuristics instead of one shared upstream config. On the current `sm_120` machine bucket, `2048` uses `KV=(64,32,2,4)` with `Q=(32,64,2,4)`, and `4096+` uses `KV=(32,64,2,4)` with the upstream-style `Q=(64,64,3,8)`.
- With those tuned buckets, unpacked FlashDeBERTa now beats eager end to end on the repo's `2048` and `4096` HF DeBERTa RTD configs, not just in the synthetic microbench.
- Current profiling result for the tuned longer padded runs: the dominant remaining CUDA cost is still the varlen backward kernels themselves, especially `_bwd_kv_dise_kernel_varlen`, with the repo-local prefix pack/unpack kernels reduced to secondary overhead.
- Packed doc-block batches are now supported for `hf_deberta_v2` with a regime split. The measured packed-docblock `1024` case routes through dense flash-with-bias by default, reusing the pairwise keep mask directly. That short dense-bias route now builds its `(B,H,S,S)` additive bias through a repo-local opaque custom op, so the old expand/gather/scaling chain no longer dominates steady-state packed-docblock flash time. Longer packed doc-block runs keep the segment-aware flash route: the collator stores compact `doc_ids`, compile metadata expands them into fixed-shape segment descriptors outside the graph, and an opaque doc-block custom op repacks contiguous document spans into a ragged batch for the tuned varlen kernels instead of materializing a dense pairwise mask.
- The dense flash-with-bias wrapper now has its own repo-local tuning seam. `src/deberta/modeling/flashdeberta_bias_op.py` resolves `FLASHDEBERTA_BIAS_FWD_*`, the generic `FLASHDEBERTA_BIAS_BWD_*` fallback, and the split `FLASHDEBERTA_BIAS_BWD_KV_*` / `FLASHDEBERTA_BIAS_BWD_Q_*` overrides before falling back to upstream FlashDeBERTa config selection. The wrapper now launches the raw bias backward `KV` and `Q` Triton kernels directly, so those two backward surfaces can be tuned independently without forking the whole flash-with-bias wrapper. `tools/flashdeberta_bias_tune.py` samples real packed doc-block batches from the repo dataloader so those overrides can be promoted from measured RTD evidence rather than synthetic shapes.
- The fused dense-bias builder itself now has a separate tuning seam too. `src/deberta/modeling/flashdeberta_dense_bias_op.py` resolves `FLASHDEBERTA_DENSE_BIAS_*` overrides for the repo-local `(B,H,S,S)` bias assembly kernel, so tuning the builder no longer needs to perturb the downstream flash-with-bias attention kernels. Its backward rule now saves per-row bucket ranges and reduces dense bias gradients with contiguous segment reductions, which removed the old scatter-heavy backward hotspot from the packed-docblock `1024` profile. On the current measured `sm_120` packed-docblock `1024` hot path, the promoted builder tile is `BLOCK_M=64`, `BLOCK_N=128`, `stages=2`, `warps=4`.
- `tools/flashdeberta_varlen_tune.py` is the supported path for refreshing these heuristics on another GPU or after larger kernel changes. It samples the real unpacked loader and persists route/kernel summaries under `local-scratch/benchmarks/flashdeberta/...`.
- TODO (flash optimization follow-up): benchmark and, only if warranted, add the upstream small-batch local-bias path for `512 < seq_len < 1024` with very small training batches.

## RoPE-specific controls

`rope` adds controls not used by `hf_deberta_v2`:

- `norm_arch`: `post` or `keel`
- `ffn_type`: `mlp` or `swiglu`
- `rope_theta`, `rotary_pct`
- `use_bias`, `keel_alpha_init`, `keel_alpha_learnable`
