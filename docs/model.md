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
- pretrained RoPE loading (`from_scratch=false`):
  - checkpoint-native architecture fields are preserved by default (`rope_theta`, `rotary_pct`, `norm_arch`, KEEL knobs, etc.)
  - non-default `ModelConfig` values are treated as explicit overrides
- FFN block:
  - `ffn_type`: `swiglu` (default) or `mlp`
  - `use_bias`: whether attention/FFN projections use bias (`false` by default for scratch RoPE builds)
  - `swiglu_adjust_intermediate` (default `true`) scales `intermediate_size` by `2/3` for scratch RoPE + SwiGLU builds so FFN parameter budget stays comparable to MLP settings
  - derived generator configs inherit discriminator scaling; explicit `generator_intermediate_size` remains explicit (not auto-rescaled)
  - note: `ffn_type` is applied for `model.from_scratch=true`; pretrained RoPE loads preserve the checkpoint's FFN type unless you provide matching configs.
- optional activation checkpointing:
  - `gradient_checkpointing`

## Norm Architecture Rationale (`norm_arch`)

Normalization-policy rationale, equations, and selection guidance are defined in [`docs/norm-strategy.md`](norm-strategy.md).

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

Operational constraint: embedding sharing adapters bind discriminator embeddings to the
generator embedding modules present at pretrainer construction time. If you replace
generator/discriminator embedding modules later (for example during manual model surgery),
recreate `DebertaV3RTDPretrainer` so sharing adapters are rebound consistently.

## HF Compatibility Mode Notes

With `backbone_type=hf_deberta_v2`, the run uses the HF DeBERTa implementation.

RoPE-specific options (`rope_theta`, `rotary_pct`, `norm_arch`, `ffn_type`, etc.) do not apply in that mode.

Packed 3D document-blocking masks are rope-only. The canonical pairwise-mask contract lives in [`docs/data.md#pairwise-mask-contract-block_cross_document_attentiontrue`](data.md#pairwise-mask-contract-block_cross_document_attentiontrue); HF backbone runs must keep `data.block_cross_document_attention=false`.

Pretraining heads follow backbone norm style by config:

- rope mode: RMSNorm heads
- hf_deberta_v2 mode: LayerNorm heads

When `backbone_type=rope` and `from_scratch=false`, checkpoint sources must be RoPE checkpoints produced from this repo's architecture. Official HF DeBERTa v2/v3 checkpoints are architecturally incompatible with the RoPE backbone.
