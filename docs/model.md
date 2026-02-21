# Model Architecture and Config

This document is the primary reference for model/backbone choices in this repo.

For data-path details, see [`docs/data.md`](data.md). For RTD objective/loss behavior, see [`docs/objective.md`](objective.md). For distributed/runtime tuning, see [`docs/fsdp2.md`](fsdp2.md).

## Backbone Modes

`model.backbone_type` supports:

- `rope` (default): modernized encoder stack in this repo
- `hf_deberta_v2`: Hugging Face DeBERTa v2/v3 compatibility path

## `rope` Backbone Knobs

Key options in `ModelConfig`:

- positional encoding:
  - `rope_theta`
  - `rotary_pct`
  - `use_absolute_position_embeddings` (default `false`)
- residual/norm topology:
  - `norm_arch`: `post` or `keel`
  - `norm_eps`
  - `keel_alpha_init`
  - `keel_alpha_learnable`
- attention:
  - `attention_implementation`: `sdpa` (recommended) or `eager`
- FFN block:
  - `ffn_type`: `swiglu` (default) or `mlp`
  - note: `ffn_type` is applied for `model.from_scratch=true`; pretrained RoPE loads preserve the checkpoint's FFN type unless you provide matching configs.
- optional activation checkpointing:
  - `gradient_checkpointing`

## Generator/Discriminator Configuration

RTD uses separate generator and discriminator backbones.

- discriminator config is the primary source
- generator config can be provided explicitly or derived
- derived generator can be adjusted with:
  - `generator_num_hidden_layers`
  - `generator_hidden_size`
  - `generator_intermediate_size`
  - `generator_num_attention_heads`

## Embedding Sharing

`model.embedding_sharing`:

- `none`
- `es` (vanilla embedding sharing)
- `gdes` (gradient-disentangled sharing; default)

`gdes` is implemented to remain compatible with FSDP2 wrapping.

## HF Compatibility Mode Notes

With `backbone_type=hf_deberta_v2`, the run uses the HF DeBERTa implementation.

RoPE-specific options (`rope_theta`, `rotary_pct`, `norm_arch`, `ffn_type`, etc.) do not apply in that mode.
