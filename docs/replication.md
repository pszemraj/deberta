# Replication Guide (DeBERTa-v2/v3)

Use this page when you want strict architecture parity rather than the repo's default modernized DeBERTa-v2 workflow.

See also: [model](model.md), [objective](objective.md), [data](data.md), [norm strategy](norm-strategy.md).

## Strict Architecture Settings

For DeBERTa-v2/v3 parity in this repo, use the `deberta_v3_parity` profile on the native
HF-compatible backbone:

```yaml
model:
  profile: deberta_v3_parity
  backbone_type: hf_deberta_v2
  tokenizer_name_or_path: microsoft/deberta-v3-base
  discriminator_model_name_or_path: microsoft/deberta-v3-base
  generator_model_name_or_path: null
  from_scratch: true
  embedding_sharing: gdes
  hf_attention_kernel: dynamic

data:
  dataset_name: HuggingFaceFW/fineweb-edu
  dataset_config_name: default
  streaming: true
  pack_sequences: true
  block_cross_document_attention: false
  max_seq_length: 512

train:
  max_steps: 500000
  mlm_probability: 0.15
  mlm_max_ngram: 1
  mask_token_prob: 1.0
  random_token_prob: 0.0
  disc_loss_weight: 10.0
  adam_epsilon: 1.0e-6
  token_weighted_gradient_accumulation: false
  decoupled_training: true
  report_to: none
```

Notes:

- `model.profile=deberta_v3_parity` applies parity-oriented defaults where values are still unset.
- `backbone_type=hf_deberta_v2` is the key switch for disentangled attention + LayerNorm parity.
- `train.decoupled_training=true` is the default and keeps two-phase RTD updates enabled.
- `train.mlm_max_ngram=1` is intentional for RTD parity and means token-level masking (not whole-word masking).
- Decoupled mode is intentionally incompatible with `model.embedding_sharing=es`; use `gdes` (recommended) or `none`.
- Keep RoPE-only knobs at defaults (they do not apply on the HF-compatible backbone).
- This repo keeps RTD objective semantics and GDES behavior explicit and tested; it does not guarantee bit-identical reproduction of original Microsoft training pipelines.

## Canonical Configs

Use the default long-run parity configs in `configs/`:

- `configs/pretrain_hf_deberta_v2_parity_base.yaml`
- `configs/pretrain_hf_deberta_v2_parity_small.yaml`

## CLI Preset

`deberta train` supports a built-in preset:

- `--preset deberta-v3-base`

Behavior:

- with config file: preset applies **model-only** defaults (data/train stay from config unless you override via explicit CLI flags)
- without config file: preset applies model + data + train starter defaults (including `max_steps=500000` and `report_to=none`)
- explicit CLI flags always override preset-applied values

Examples:

```bash
# Full starter run (no config file)
deberta train --preset deberta-v3-base

# Config-driven run with strict parity defaults
deberta train configs/pretrain_hf_deberta_v2_parity_base.yaml

# Preset + explicit override
deberta train --preset deberta-v3-base --max_steps 20000
```
