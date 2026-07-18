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

Export infers the run directory from the checkpoint parent and requires that directory's `model_config.json`, `data_config.json`, and `run_metadata.json`. If a checkpoint was moved elsewhere, pass `--run-dir <original-run-dir>` explicitly.

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

- `--what discriminator` or `--what generator`: writes the model, tokenizer, `export_meta.json`, and supporting files in a flat directory at `--output-dir`.
- `--what both`: writes model weights, config, README, and license under `--output-dir/discriminator/` and `--output-dir/generator/`; tokenizer files and `export_meta.json` remain at the shared `--output-dir` root. Load the selected model from its component directory and the tokenizer from the root.

Native `hf_deberta_v2` exports load through stock Hugging Face `AutoModel` APIs. RoPE exports are
standalone artifacts but require this package's `DebertaRoPEModel` implementation.

Flash-trained checkpoints are reconstructed with eager attention during consolidation, so export does not require the optional FlashDeBERTa runtime.

Training-only keys are removed from exported model configs, and export metadata uses run/checkpoint directory names rather than machine-local absolute paths. Safetensors output is enabled by default; use `--no-safe-serialization` only when a consumer requires PyTorch serialization.

## Shared embedding export

Embedding-sharing behavior is described in [Architectures](../advanced/architectures.md#rtd-architecture-notes).
Export converts both shared modes back to ordinary embedding weights for word, position, and
token-type embeddings:

- `es`: use the generator embedding weight
- `gdes`: use `generator_weight + discriminator_bias`

## Partial export mode

Manual `deberta export` is strict on state-dict compatibility by default. Use `--allow-partial-export` only for recovery or debugging. The automatic `train.checkpoint.export_hf_final` subprocess deliberately allows partial loading so a completed training run is not failed by an export-only mismatch; run the manual command afterward when strict verification is required.
