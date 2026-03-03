# Exporting models

## Why export is separate

Training checkpoints store the RTD pretrainer (generator + discriminator). Downstream tasks usually need only the discriminator as a standalone HF model.

`deberta export` consolidates checkpoint state and writes standalone Hugging Face artifacts.

## Basic export

```bash
deberta export <run_dir>/checkpoint-<step> \
  --what discriminator \
  --output-dir <run_dir>/exported_hf
```

`--what` supports:

- `discriminator`
- `generator`
- `both`

## FSDP checkpoint consolidation

For distributed runs (`distributed_type=FSDP`), export uses Accelerate/Torch distributed checkpoint loading and gathers full state for final artifact writing.

Key knobs:

- `--offload-to-cpu` / `--no-offload-to-cpu`
- `--rank0-only` / `--no-rank0-only`

Default output path is `<run_dir>/exported_hf` and must be empty if it already exists.

## GDES merge behavior

When training used `embedding_sharing=gdes`, discriminator embedding weights are represented as base + bias components.

During export, embedding tensors are merged back into standard HF embedding weights:

- `merged_weight = generator_weight + discriminator_bias`

This produces standard export weights compatible with normal HF loading for the target backbone.

## Partial export mode

By default export is strict on state-dict compatibility. Use `--allow-partial-export` only for recovery/debug cases.

## Config/tokenizer artifacts

Export writes tokenizer files and cleaned `config.json` artifacts. Training-internal keys are stripped from exported model configs.
