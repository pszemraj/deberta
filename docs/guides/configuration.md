# Configuration

## Config shape

The loader accepts nested YAML/JSON with these top-level sections:

- `model`
- `data`
- `train`
- `optim`
- `checkpoint`
- `logging`
- `debug`

Runtime training is driven by `ModelConfig`, `DataConfig`, and `TrainConfig`. The additional sections are parsed for strict top-level schema support and projected onto runtime train/logging knobs where applicable.

## Load pipeline

`deberta train` uses this order:

1. parse YAML/JSON
2. resolve variables (`$variables.*`, `{$variables.*}`, `${variables.*}`)
3. build dataclasses
4. apply preset defaults (`--preset`, if provided)
5. apply explicit CLI flags (`--max_steps`, `--dataset_name`, ...)
6. apply typed dotted overrides (`--override section.field=value`)
7. apply profile/backbone effective defaults (`apply_profile_defaults`)
8. run validation (`validate_model_config`, `validate_data_config`, `validate_train_config`, workflow checks)

Invalid keys, bad types, and incompatible combinations raise immediately.

## Precedence

Highest priority wins:

1. `--override section.field=value`
2. explicit CLI field flags
3. config file values
4. preset-injected values
5. dataclass defaults

When a value from the source config file is changed later by preset/CLI/default normalization, the CLI emits explicit mutation warnings to stderr.

## Variable interpolation

Supported forms:

- type-preserving full replacement: `$variables.foo`
- inline string substitution: `run_{$variables.foo}` or `run_${variables.foo}`

Circular references and unknown variable paths fail fast.

## Dotted overrides

Examples:

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_base.yaml \
  --override train.max_steps=2000 \
  --override model.hf_model_size=small \
  --override data.max_seq_length=1024
```

Overrides are type-cast against dataclass field annotations (`bool`, `int`, `float`, `str`, optional types).

## Parity++ effective defaults (`backbone_type=hf_deberta_v2`)

When these fields are not explicitly set by user config or CLI, the loader applies:

- `train.mask_token_prob = 1.0`
- `train.random_token_prob = 0.0`
- `train.disc_loss_weight = 10.0`
- `train.adam_epsilon = 1e-6`
- `train.warmup_steps = 10000`
- `train.token_weighted_gradient_accumulation = true`

`train.learning_rate` is not auto-changed by parity++ defaults.

## Snapshot files and reproducibility

At run start, the trainer writes:

- `config_original.yaml`: exact source file when available, otherwise resolved payload fallback
- `config_resolved.yaml`: runtime-resolved payload used for tracking/logging

For W&B runs, the trainer uploads:

- `config_resolved.yaml`
- source config file path payload when available
- `config_original.yaml` fallback when no source file is available
