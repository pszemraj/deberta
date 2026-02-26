# Model Architecture and Config

This document is the primary reference for model/backbone choices in this repo.

For data-path details, see [`docs/data.md`](data.md). For RTD objective/loss behavior, see [`docs/objective.md`](objective.md). For distributed/runtime tuning (including compile scope resolution), see [`docs/fsdp2.md`](fsdp2.md).

## Backbone Modes

`model.backbone_type` supports:

- `rope` (default): modernized encoder stack in this repo
- `hf_deberta_v2`: native DeBERTa v2/v3-style backbone (HF-compatible config/weight path)

## Source Resolution Contract

Builder behavior is intentionally deterministic and split into two phases:

1. resolve config sources
2. resolve weight sources

### Config Sources

| Mode | Discriminator config | Generator config |
|---|---|---|
| `backbone_type=rope`, `from_scratch=true` | synthetic config built from `model.*` rope scratch fields | explicit `model.generator_config_name_or_path` / `model.generator_model_name_or_path` if set; otherwise derived from discriminator config |
| `backbone_type=rope`, `from_scratch=false` | `model.discriminator_config_name_or_path` else `model.discriminator_model_name_or_path` | explicit generator config/model source if set; otherwise derived from discriminator config |
| `backbone_type=hf_deberta_v2`, `from_scratch=true|false` | `model.discriminator_config_name_or_path` else `model.discriminator_model_name_or_path` | explicit generator config/model source if set; otherwise derived from discriminator config |

### Weight Sources

| Mode | Discriminator weights | Generator weights |
|---|---|---|
| `from_scratch=true` | random init (`from_config`) | random init (`from_config`) |
| `from_scratch=false` + explicit generator model source | `model.discriminator_model_name_or_path` | `model.generator_model_name_or_path` |
| `from_scratch=false` + no generator source (derived generator exception) | `model.discriminator_model_name_or_path` | discriminator fallback (`model.discriminator_model_name_or_path`) |

Strict pairing rule in pretrained mode:

- if `model.generator_config_name_or_path` is set, `model.generator_model_name_or_path` must also be set
- cross-component fallback is only allowed for the derived-generator mode (both generator source fields unset)

### Tokenizer Compatibility Policy

- scratch mode (`from_scratch=true`): config `vocab_size` and special token ids are aligned to the tokenizer
- pretrained mode (`from_scratch=false`): config/tokenizer vocabulary and special ids are validated; mismatches fail fast

## Pretrained RoPE Overrides (`from_scratch=false`)

Pretrained RoPE loads use explicit override fields only:

- `pretrained_max_position_embeddings`
- `pretrained_rope_theta`
- `pretrained_rotary_pct`
- `pretrained_use_absolute_position_embeddings`
- `pretrained_type_vocab_size`
- `pretrained_norm_arch`
- `pretrained_norm_eps`
- `pretrained_keel_alpha_init`
- `pretrained_keel_alpha_learnable`
- `pretrained_ffn_type`
- `pretrained_use_bias`
- `pretrained_initializer_range`

Legacy implicit behavior is removed:

- non-default scratch fields (for example `rope_theta`, `ffn_type`, `norm_arch`) are no longer interpreted as pretrained overrides
- to override pretrained RoPE configs, use the `pretrained_*` fields above

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
  - `hidden_dropout_prob` and `attention_probs_dropout_prob` default to `null` (no override)
  - set a numeric value (including `0.0`) to explicitly override discriminator/generator dropout
  - leaving them `null` preserves checkpoint-native dropout values for pretrained loads
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
In pretrained mode, explicit generator config also requires explicit generator model weights (strict pairing).

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

With `backbone_type=hf_deberta_v2`, training uses the repo-native DeBERTa-v2 implementation in
[`src/deberta/modeling/deberta_v2_native.py`](../src/deberta/modeling/deberta_v2_native.py).
Training does not instantiate `transformers.DebertaV2Model` directly.

HF checkpoints/configs are still used as sources (`AutoConfig`/`from_pretrained`) for compatibility.

RoPE-specific options (`rope_theta`, `rotary_pct`, `norm_arch`, `ffn_type`, etc.) do not apply in that mode.

Packed 3D document-blocking masks are rope-only. The canonical pairwise-mask contract lives in [`docs/data.md#pairwise-mask-contract-block_cross_document_attentiontrue`](data.md#pairwise-mask-contract-block_cross_document_attentiontrue); HF backbone runs must keep `data.block_cross_document_attention=false`.

Native HF attention kernel is configured via `model.hf_attention_kernel`:

- `dynamic` (default)
- `cached_bmm` (cached relative-position ids + bmm bias path)
- `stable` (compile-focused cached-bmm path with fp32 score/probability accumulation and static rel-pos table slicing)

Compile runtime policy for this mode (for example, auto FFN-only fallback in default+inductor) is documented in [`docs/fsdp2.md#torchcompile`](fsdp2.md#torchcompile).

Pretraining heads follow backbone norm style by config:

- rope mode: RMSNorm heads
- hf_deberta_v2 mode: LayerNorm heads

When `backbone_type=rope` and `from_scratch=false`, checkpoint sources must be RoPE checkpoints produced from this repo's architecture. Official HF DeBERTa v2/v3 checkpoints are architecturally incompatible with the RoPE backbone.
