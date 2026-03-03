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

For `hf_deberta_v2`, backbone configs are synthesized in-repo from `model.hf_model_size` + explicit overrides. The trainer does not load architecture config JSON from HF model hubs.

## HF-size presets (`hf_model_size`)

`hf_model_size` supports `xsmall`, `small`, `base`, `large`.

Generator defaults are derived from discriminator width/heads/ffn and half depth on `hf_deberta_v2`.

## RTD architecture notes

- discriminator and generator backbones are separate modules
- `embedding_sharing` supports `none`, `es`, `gdes`
- decoupled two-phase RTD updates are enabled by default (`train.decoupled_training=true`)

## Default deltas vs released DeBERTa-v3 configs

Intentional repo defaults:

- dropout defaults are `0.0` (`model.hidden_dropout_prob`, `model.attention_probs_dropout_prob`)
- RTD parity++ default `train.disc_loss_weight` is `10.0`
- default `train.adam_beta2` is `0.999`

Use explicit config values when you want different behavior.

## RoPE-specific controls

`rope` adds controls not used by `hf_deberta_v2`:

- `norm_arch`: `post` or `keel`
- `ffn_type`: `mlp` or `swiglu`
- `rope_theta`, `rotary_pct`
- `use_bias`, `keel_alpha_init`, `keel_alpha_learnable`
