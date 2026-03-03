# Model Architecture and Config

See also: [replication guide](replication.md), [data pipeline](data.md), [RTD objective](objective.md), [runtime/compile](fsdp2.md).

## Backbone Families

`model.backbone_type` supports two architecturally distinct encoder families that share the same RTD pretraining objective and GDES embedding sharing:

| | `hf_deberta_v2` (default) | `rope` (experimental opt-in) |
|---|---|---|
| **Attention** | Disentangled (C2C + C2P + P2C + optional P2P relative-position bias) | Standard multi-head (QKV → RoPE → SDPA) |
| **Position encoding** | Learned relative-position embeddings + bucket indices | Rotary embeddings (geometric, no learned table) |
| **Normalization** | LayerNorm | RMSNorm |
| **FFN** | MLP only (GELU) | SwiGLU (default) or MLP |
| **Bias** | Hardcoded per layer | Configurable (`use_bias`) |
| **Pretraining objective** | RTD (ELECTRA-style) + GDES | RTD (ELECTRA-style) + GDES |

The `hf_deberta_v2` backbone faithfully implements the DeBERTa-v2/v3 architecture including disentangled attention with content-to-position (C2P), position-to-content (P2C), and optional position-to-position (P2P) bias terms via separate `pos_key_proj`/`pos_query_proj` projections.

The `rope` backbone is an experimental modern encoder path that uses the DeBERTa-v3 RTD pretraining recipe and GDES embedding sharing but does **not** implement disentangled attention. Position information is encoded geometrically via rotary embeddings in Q/K vectors.

## Source Resolution Contract

Builder behavior is intentionally deterministic and split into two phases:

1. resolve config sources
2. resolve weight sources

### Config Sources

| Mode | Discriminator config | Generator config |
|---|---|---|
| `backbone_type=rope`, `from_scratch=true` | synthetic config built from `model.*` rope scratch fields | explicit `model.generator_model_name_or_path` if set; otherwise derived from discriminator config |
| `backbone_type=rope`, `from_scratch=false` | `model.discriminator_model_name_or_path` | explicit `model.generator_model_name_or_path` if set; otherwise derived from discriminator config |
| `backbone_type=hf_deberta_v2`, `from_scratch=true|false` | `model.discriminator_model_name_or_path` | explicit `model.generator_model_name_or_path` if set; otherwise derived from discriminator config |

### Weight Sources

| Mode | Discriminator weights | Generator weights |
|---|---|---|
| `from_scratch=true` | random init (`from_config`) | random init (`from_config`) |
| `from_scratch=false` + explicit generator model source | `model.discriminator_model_name_or_path` | `model.generator_model_name_or_path` |
| `from_scratch=false` + no generator source (derived generator exception) | `model.discriminator_model_name_or_path` | discriminator fallback (`model.discriminator_model_name_or_path`) |

Generator-source rule:

- set `model.generator_model_name_or_path` to use explicit generator config+weights from that source
- leave it unset to use derived-generator fallback from discriminator source

### Tokenizer Compatibility Policy

- scratch mode (`from_scratch=true`): config `vocab_size` and special token ids are aligned to the tokenizer
- pretrained mode (`from_scratch=false`): config/tokenizer vocabulary and special ids are validated; mismatches fail fast
- optional vocab controls:
  - `model.tokenizer_allow_vocab_resize=true` enables tokenizer growth via `add_tokens(...)`
  - `model.tokenizer_vocab_target=<N>` requests a minimum tokenizer/model vocab size (`N`)
  - `model.tokenizer_vocab_multiple=<M>` rounds resolved vocab up to multiple `M` (`1` disables)
- placeholder growth tokens use the inert pattern `<|deberta_extra_token_{idx}|>`

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

Pretrained override behavior is explicit:

- non-default scratch fields (for example `rope_theta`, `ffn_type`, `norm_arch`) are not interpreted as pretrained overrides
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

### Export Interoperability (RoPE)

RoPE exports are saved with `model_type="deberta-rope"` and currently require this repo's model class:

```python
from transformers import AutoTokenizer
from deberta.modeling.rope_encoder import DebertaRoPEModel

model = DebertaRoPEModel.from_pretrained("<export_dir>")
tokenizer = AutoTokenizer.from_pretrained("<export_dir>")
```

`transformers.AutoModel.from_pretrained("<export_dir>")` is not currently supported for RoPE exports without custom auto-class registration and packaged modeling code.

## Norm Architecture Rationale (`norm_arch`)

Normalization-policy rationale, equations, and selection guidance are in [norm-strategy.md](norm-strategy.md).

## Generator/Discriminator Configuration

RTD uses separate generator and discriminator backbones.

- discriminator config is the primary source
- generator config can be provided explicitly or derived
- derived generator can be adjusted with:
  - `generator_num_hidden_layers`
  - `generator_hidden_size`
  - `generator_intermediate_size`
  - `generator_num_attention_heads`

If `generator_model_name_or_path` is set, the derived-generator sizing knobs above must be unset.

## Embedding Sharing

`model.embedding_sharing`:

- `none`
- `es` (vanilla embedding sharing)
- `gdes` (gradient-disentangled sharing; default)

`gdes` remains compatible with FSDP2 wrapping.

Operational constraint: embedding sharing adapters bind discriminator embeddings to the
generator embedding modules present at pretrainer construction time. If you replace
generator/discriminator embedding modules later (for example during manual model surgery),
recreate `DebertaV3RTDPretrainer` so sharing adapters are rebound consistently.

## HF DeBERTa-v2 Backbone

With `backbone_type=hf_deberta_v2`, training uses the repo-native DeBERTa-v2 implementation in
[`src/deberta/modeling/deberta_v2_native.py`](../src/deberta/modeling/deberta_v2_native.py).
Training does not instantiate `transformers.DebertaV2Model` directly.

HF checkpoints/configs are still used as sources (`AutoConfig`/`from_pretrained`) for compatibility.

Context length for this mode should be configured directly in YAML with
`model.hf_max_position_embeddings` (supported when `model.from_scratch=true`).

### Architectural properties

- **Disentangled attention**: decomposes attention scores into content-to-content (C2C), content-to-position (C2P), and position-to-content (P2C) terms via separate `pos_key_proj`/`pos_query_proj` projections
- **Relative position**: learned embedding table with bucket indices, computed on-device per forward (no upfront O(max_len²) allocation)
- **Padding masks**: 2D `(B,S)` padding masks stay as `(B,1,1,S)` broadcast — no S² outer product expansion
- **No-mask fast path**: when `attention_mask=None` (unpadded batches), no mask is materialized

### Current limitations vs RoPE path

- LayerNorm only (no RMSNorm option) — tracked in [roadmap](roadmap.md)
- MLP FFN only (no SwiGLU option) — tracked in [roadmap](roadmap.md)
- Bias not configurable — tracked in [roadmap](roadmap.md)
- `data.block_cross_document_attention` must be `false` (packed 3D doc-blocking masks are rope-only; see [data.md](data.md))

### Attention kernel selection

RoPE-specific options (`rope_theta`, `rotary_pct`, `norm_arch`, `ffn_type`, etc.) do not apply in this mode.

`model.hf_attention_kernel` selects the disentangled attention implementation:

- `dynamic` (default): computes disentangled bias with the dynamic einsum+gather path
- `cached_bmm`: computes disentangled bias with a batched-matmul+gather path
- `stable`: semantic alias for the `cached_bmm` kernel, used by compile-stability presets

Compile scope and stability policy for this mode is in [runtime: torch.compile](fsdp2.md#torchcompile).

Pretraining heads follow backbone norm style by config:

- rope mode: RMSNorm heads
- hf_deberta_v2 mode: LayerNorm heads

When `backbone_type=rope` and `from_scratch=false`, checkpoint sources must be RoPE checkpoints produced from this repo's architecture. Official HF DeBERTa v2/v3 checkpoints are architecturally incompatible with the RoPE backbone.
