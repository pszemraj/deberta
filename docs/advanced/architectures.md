# Architectures

## Backbone types

`model.backbone_type` selects one of two encoder families.

| Setting | `hf_deberta_v2` | `rope` |
|---|---|---|
| Attention | DeBERTa disentangled attention (C2C/C2P/P2C/P2P) | standard QKV + RoPE |
| Position | relative-position buckets/embeddings | rotary; optional learned absolute input embeddings |
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
- newly attached MLM and RTD heads use the owning backbone's `initializer_range`; backbone
  parameters are not reinitialized
- native generators with `position_biased_input=false` use Enhanced Mask Decoding with raw learned
  position states, penultimate-layer KV, and two shared applications of the last layer
- generic backbone `z_steps` is not an EMD substitute; RTD generators require `z_steps=0`
- Document-blocked packing preserves standalone objective semantics as described in the [data pipeline](../guides/data-pipeline.md#cross-document-attention-blocking).

## Parity divergences

- Generator sampling excludes special/control token ids through a forbidden vocabulary mask (for
  example PAD/CLS/SEP/MASK), unlike the original DeBERTa sampler.
- Discriminator embedding sharing includes word, position, and token-type embeddings.
- When only generator hidden size is overridden, generator FFN width still inherits the
  discriminator width unless explicitly set.

## FlashDeBERTa

The native backbone can run disentangled attention through Triton FlashDeBERTa kernels
(`model.hf.attention_impl=flash`), including packed doc-block routing. Routes, kernel tuning,
tooling, and caveats: [Advanced / FlashDeBERTa attention](flash-attention.md).

## RoPE-specific controls

`rope` adds controls for rotary geometry, optional learned absolute input embeddings, MLP/SwiGLU selection, RMSNorm placement, attention implementation, and KEEL residual scaling. Exact fields and interactions are listed under `model.rope.*` in the [config reference](../../configs/config_reference.yaml).
