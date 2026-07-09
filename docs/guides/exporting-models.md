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

## Output layout

- `--what discriminator` or `--what generator`: writes a single HF model in a flat directory at `--output-dir`.
- `--what both`: writes `--output-dir/discriminator/` and `--output-dir/generator/` (each a standalone HF model dir).

Native `hf_deberta_v2` exports load through stock Hugging Face `AutoModel` APIs. RoPE exports are
standalone artifacts but require this package's `DebertaRoPEModel` implementation.

## Shared embedding export

Embedding-sharing behavior is described in [Architectures](../advanced/architectures.md#rtd-architecture-notes).
Export converts both shared modes back to ordinary embedding weights for word, position, and
token-type embeddings:

- `es`: use the generator embedding weight
- `gdes`: use `generator_weight + discriminator_bias`

## Partial export mode

By default export is strict on state-dict compatibility. Use `--allow-partial-export` only for recovery/debug cases.

## Config/tokenizer artifacts

Export writes tokenizer files, cleaned `config.json`, `README.md`, `LICENSE`, and
`export_meta.json`. Training-internal keys are removed from model configs, and export metadata uses
run/checkpoint directory names rather than machine-local absolute paths.

Safetensors output is enabled by default. Use `--no-safe-serialization` only when a consumer
requires PyTorch serialization.
