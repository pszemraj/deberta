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
- Padded varlen now uses dedicated repo-local prefix-pack Triton kernels, including shared pair/triple pack and unpack paths, so q/k/v and positional tensors are no longer copied through independent generic gather/scatter launches.
- TODO (flash optimization follow-up): benchmark and, only if warranted, add the upstream small-batch local-bias path for `512 < seq_len < 1024` with very small training batches.
- TODO (flash feature follow-up): add packed doc-block-aware flash attention for `hf_deberta_v2` only after the simpler padding-style flash path is proven worthwhile in longer-context runs.

## RoPE-specific controls

`rope` adds controls not used by `hf_deberta_v2`:

- `norm_arch`: `post` or `keel`
- `ffn_type`: `mlp` or `swiglu`
- `rope_theta`, `rotary_pct`
- `use_bias`, `keel_alpha_init`, `keel_alpha_learnable`
