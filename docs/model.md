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
- dropout:
  - `hidden_dropout_prob` and `attention_probs_dropout_prob` default to `0.0`
  - these values are applied to both discriminator and generator configs unless explicitly set to `null` in config to preserve checkpoint-native dropout values
- FFN block:
  - `ffn_type`: `swiglu` (default) or `mlp`
  - `use_bias`: whether attention/FFN projections use bias (`false` by default for scratch RoPE builds)
  - `swiglu_adjust_intermediate` (default `true`) scales `intermediate_size` by `2/3` for scratch RoPE + SwiGLU builds so FFN parameter budget stays comparable to MLP settings
  - derived generator configs inherit discriminator scaling; explicit `generator_intermediate_size` remains explicit (not auto-rescaled)
  - note: `ffn_type` is applied for `model.from_scratch=true`; pretrained RoPE loads preserve the checkpoint's FFN type unless you provide matching configs.
- optional activation checkpointing:
  - `gradient_checkpointing`

## Norm Architecture Rationale (`norm_arch`)

`model.norm_arch: post` is the default by design, not a legacy leftover.

Why:

- This repo's primary encoder targets are shallow-to-mid depth (roughly 12-28 layers in shipped configs), where standard Post-LN is typically stable at the learning-rate ranges used in this project.
- Pre-LN became the dominant default because it is easier to optimize at very large depth in decoder-only LLMs. That does not automatically make it the best default for moderate-depth encoders.
- Empirical and recent literature trends point to stronger layer coupling / depth utilization under Post-LN style blocks, while Pre-LN can show higher layer redundancy as depth increases.

How to choose:

- Use `post` for the standard training recipes in this repo (recommended default).
- Use `keel` when pushing depth and/or optimization aggressiveness and you want additional stability margin while keeping Post-LN style behavior.
- Treat KEEL as the depth-scaling path, not as a correctness fix required for normal 12-28 layer runs.

For details on KEEL's residual form and paper context, see [`docs/keel-paper-technical-overview.md`](keel-paper-technical-overview.md).

Input embedding RMSNorm is intentionally kept in front of the encoder stack. With RMSNorm-based Post-LN/KEEL blocks this means first-layer input is already normalized, which is a deliberate stability choice in this codebase (not an accidental duplicate LayerNorm artifact).

## Generator/Discriminator Configuration

RTD uses separate generator and discriminator backbones.

- discriminator config is the primary source
- generator config can be provided explicitly or derived
- derived generator can be adjusted with:
  - `generator_num_hidden_layers`
  - `generator_hidden_size`
  - `generator_intermediate_size`
  - `generator_num_attention_heads`

If `generator_config_name_or_path` or `generator_model_name_or_path` is set, the derived-generator sizing knobs above must be unset.

## Embedding Sharing

`model.embedding_sharing`:

- `none`
- `es` (vanilla embedding sharing)
- `gdes` (gradient-disentangled sharing; default)

`gdes` is implemented to remain compatible with FSDP2 wrapping.

## HF Compatibility Mode Notes

With `backbone_type=hf_deberta_v2`, the run uses the HF DeBERTa implementation.

RoPE-specific options (`rope_theta`, `rotary_pct`, `norm_arch`, `ffn_type`, etc.) do not apply in that mode.

When `backbone_type=rope` and `from_scratch=false`, checkpoint sources must be RoPE checkpoints produced from this repo's architecture. Official HF DeBERTa v2/v3 checkpoints are architecturally incompatible with the RoPE backbone.
