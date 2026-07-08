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

Generator defaults are derived from discriminator width/heads/ffn and half depth on `hf_deberta_v2` (`rope` derives one-third depth).

## RTD architecture notes

- discriminator and generator backbones are separate modules
- `model.embedding_sharing` supports `none`, `es`, `gdes`
- decoupled two-phase RTD updates are enabled by default (`train.decoupled_training=true`)

## Default deltas vs released DeBERTa-v3 configs

Intentional repo defaults:

- dropout defaults are `0.0` (`model.dropout.hidden_prob`, `model.dropout.attention_probs_prob`)
- `train.objective.disc_loss_weight` has an effective default of `10.0` for `hf_deberta_v2` when
  left unset (the raw dataclass default is `50.0`)
- default `optim.adam.beta2` is `0.999`

Use explicit config values when you want different behavior.

## Parity divergences

- Intentional divergence: generator sampling excludes special/control token ids via a forbidden vocabulary mask (for example PAD/CLS/SEP/MASK). This differs from original DeBERTa sampling and is kept intentionally.
- TODO (strict parity follow-up): evaluate narrowing discriminator embedding sharing to only word+position embeddings; current sharing also includes `token_type_embeddings`.
- TODO (architecture follow-up): evaluate defaulting `generator_intermediate_size` from `generator_hidden_size` when only width is overridden; current behavior inherits discriminator FFN width unless explicitly set.

## FlashDeBERTa

The native backbone can run disentangled attention through Triton FlashDeBERTa kernels
(`model.hf.attention_impl=flash`), including packed doc-block routing. Routes, kernel tuning,
tooling, caveats, and open follow-ups: [Advanced / FlashDeBERTa attention](flash-attention.md).

## RoPE-specific controls

`rope` adds controls not used by `hf_deberta_v2`:

- `norm_arch`: `post` or `keel`
- `ffn_type`: `mlp` or `swiglu`
- `rope_theta`, `rotary_pct`
- `use_bias`, `keel_alpha_init`, `keel_alpha_learnable`
