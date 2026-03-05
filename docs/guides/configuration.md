# Configuration

## Config shape

The loader accepts nested YAML/JSON with these top-level sections:

- `model`
- `data`
- `train`
- `optim`
- `logging`

There is no flat/legacy mode and no extra top-level sections.

## Load pipeline

`deberta train` uses this order:

1. parse YAML/JSON
2. resolve variables (`$variables.*`, `{$variables.*}`, `${variables.*}`)
3. build nested frozen dataclasses
4. apply preset defaults (`--preset`, if provided)
5. apply direct dotted CLI flags (`--train.max_steps 2000`, `--optim.scheduler.warmup_steps 500`, ...)
6. apply profile/backbone effective defaults (`apply_profile_defaults`)
7. run validation (`validate_model_config`, `validate_data_config`, `validate_train_config`, `validate_optim_config`, `validate_logging_config`, workflow checks)

Invalid keys, bad types, and incompatible combinations raise immediately.

## Precedence

Highest priority wins:

1. direct dotted CLI flags (`--section.path value`)
2. preset-injected values (only model fields when a config file is provided; full preset payload when no config file is provided)
3. config file values
4. dataclass defaults

When a value from the source config file is changed later by preset/CLI/default normalization, the CLI emits explicit mutation warnings to stderr.

## Variable interpolation

Supported forms:

- type-preserving full replacement: `$variables.foo`
- inline string substitution: `run_{$variables.foo}` or `run_${variables.foo}`

Circular references and unknown variable paths fail fast.

## Dotflag examples

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_base.yaml \
  --train.max_steps 2000 \
  --optim.scheduler.warmup_steps 200 \
  --data.packing.max_seq_length 1024 \
  --logging.wandb.enabled true \
  --logging.backend none
```

All values are type-cast against dataclass field annotations (`bool`, `int`, `float`, `str`, optional types).

For YAML/JSON config files, boolean fields must be true booleans (`true`/`false`), not quoted strings (for example, `"false"` is rejected). Numeric fields accept numeric literals and numeric strings (for example, `1e-6`).

## Dry-run behavior

`deberta train --dry-run` runs validation and runtime preflight without starting optimization/training loops and without writing checkpoints.

Dry-run may still access networked dataset/tokenizer sources and populate Hugging Face caches.

## Parity++ effective defaults (`model.backbone_type=hf_deberta_v2`)

When these fields are not explicitly set by user config or CLI, the loader applies:

- `train.objective.mask_token_prob = 1.0`
- `train.objective.random_token_prob = 0.0`
- `train.objective.disc_loss_weight = 10.0`
- `optim.adam.epsilon = 1e-6`
- `optim.scheduler.warmup_steps = 10000`
- `train.token_weighted_gradient_accumulation = true`

`optim.lr.base` is not auto-changed by parity++ defaults.

## Snapshot files and reproducibility

At run start, the trainer writes:

- checkpoint output dir:
  - `model_config.json`
  - `data_config.json`
  - `train_config.json`
  - `optim_config.json`
  - `logging_config.json`
  - `run_metadata.json`
- logging output dir:
  - `config_original.yaml`
  - `config_resolved.yaml`

If `logging.output_dir` is unset, it defaults to `train.checkpoint.output_dir`.

By default `train.checkpoint.export_hf_final=true`, so successful training also performs a final export pass into `<train.checkpoint.output_dir>/final_hf`.

For W&B runs, the trainer uploads:

- `config_resolved.yaml`
- source config file payload when available
- `config_original.yaml` fallback when no source file is available
