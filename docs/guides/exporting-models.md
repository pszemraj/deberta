# Exporting models

## Why export is separate

Training checkpoints store the RTD pretrainer (generator + discriminator). Downstream tasks usually need only the discriminator as a standalone HF model.

`deberta export` consolidates checkpoint state and writes standalone Hugging Face artifacts. Training materializes the exact discriminator config, generator config, and tokenizer into the run directory; resume and export use those owned artifacts rather than resolving the original mutable local path or Hub ID again.

## Basic export

```bash
deberta export <run_dir>/checkpoint-<step> \
  --what discriminator \
  --output-dir <run_dir>/exported_hf
```

Export infers the run directory from the checkpoint parent and requires that directory's high-level config snapshots, materialized component configs, and `tokenizer/` directory. It validates `run_metadata.json` when that optional snapshot is present. If a checkpoint was moved elsewhere, pass `--run-dir <original-run-dir>` explicitly.

`--what` supports:

- `discriminator`
- `generator`
- `both`

## Experimental FSDP checkpoint consolidation

The manual exporter contains an experimental path that uses Accelerate/Torch distributed checkpoint loading and gathers full state for final artifact writing. See [Distributed training](../advanced/distributed-training.md) for the support boundary and required validation.

FSDP export offloads the consolidated state to CPU and gathers it on rank 0 by default. Override those defaults only when the alternative fits the available memory:

- `--no-offload-to-cpu`
- `--no-rank0-only`

Default output path is `<run_dir>/exported_hf` and must be empty if it already exists.

## Output layout

- `--what discriminator` or `--what generator`: writes the model, tokenizer, `export_meta.json`, and supporting files in a flat directory at `--output-dir`.
- `--what both`: writes model weights, config, README, and license under `--output-dir/discriminator/` and `--output-dir/generator/`; tokenizer files and `export_meta.json` remain at the shared `--output-dir` root. Load the selected model from its component directory and the tokenizer from the root.

`export_meta.json` records `artifact_type` (`rtd_pretrained_encoder` for one component, `rtd_pretrained_encoder_bundle` for both), the requested target, `strict_state_load`, `includes_rtd_head`, and the per-component `embedding_materialization` described under [shared embedding export](#shared-embedding-export).

Native `hf_deberta_v2` exports load through stock Hugging Face `AutoModel` APIs. RoPE exports are
standalone artifacts but require this package's `DebertaRoPEModel` implementation.
Native config materialization canonicalizes positional-attention terms and rejects bucketed relative spans that stock Hugging Face would interpret differently, so a strict state load cannot hide an attention-function mismatch.

Flash-trained checkpoints are reconstructed with eager attention during consolidation, so export does not require the optional FlashDeBERTa runtime.

Training-only keys are removed from exported model configs, and export metadata uses run/checkpoint directory names rather than machine-local absolute paths. Safetensors output is enabled by default; use `--no-safe-serialization` only when a consumer requires PyTorch serialization.

## Shared embedding export

Embedding-sharing behavior is described in [Architectures](../advanced/architectures.md#rtd-architecture-notes).
Export converts both shared modes back to ordinary embedding weights for word, position, and
token-type embeddings, and records the choice as `embedding_materialization` in `export_meta.json`:

- `none`: the discriminator keeps its own checkpoint weights (`discriminator_checkpoint`)
- `es`: use the generator embedding weight (`generator_checkpoint_shared`)
- `gdes`: use `generator_weight + discriminator_bias` (`generator_checkpoint_plus_discriminator_bias`)

Generator materialization is always `generator_checkpoint`.

## Strict verification and partial export

Both manual `deberta export` and automatic `train.checkpoint.export_hf_final` are strict by
default: state-dict loading must be exact, and each staged encoder is reloaded and its outputs
verified against the in-memory materialized encoder before the artifact is published.

For automatic export on same-directory continuation, the refreshed artifact is verified before it
replaces the previous `final_hf`, so a failed refresh leaves the previous artifact intact. On the
supported single-process path, a failed final checkpoint save or automatic export fails the
training command rather than finishing silently.

`deberta export --allow-partial-export` relaxes both checks (state-dict strictness and staged
encoder parity). Use it only for recovery or debugging, never for publishing artifacts.
