# Replication Guide (DeBERTa-v2/v3)

Use this page when you want strict architecture parity rather than the repo's modern RoPE defaults.

See also: [model](model.md), [objective](objective.md), [data](data.md), [norm strategy](norm-strategy.md).

## Strict Architecture Settings

For DeBERTa-v2/v3 architecture parity in this repo, use the native HF-compatible backbone:

```yaml
model:
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
  mlm_max_ngram: 3
  mask_token_prob: 0.8
  random_token_prob: 0.1
  disc_loss_weight: 50.0
  report_to: none
```

Notes:

- `backbone_type=hf_deberta_v2` is the key switch for disentangled attention + LayerNorm parity.
- Keep RoPE-only knobs at defaults (they do not apply on the HF-compatible backbone).
- `mlm_max_ngram=3` enables whole-word n-gram masking mode used in DeBERTa-style recipes.
- This repo keeps RTD objective semantics and GDES behavior explicit and tested; it does not guarantee bit-identical reproduction of original Microsoft training pipelines.

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

# Config-driven run, but force model parity settings from preset
deberta train configs/pretrain_rope_fineweb_edu.yaml --preset deberta-v3-base

# Preset + explicit override
deberta train --preset deberta-v3-base --max_steps 20000
```
