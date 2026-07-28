# Quickstart

## 1) Tiny CPU smoke run

Create enough local text for packed batches, then run the genuinely small CPU recipe:

```bash
mkdir -p data
seq 1 64 | sed 's/^/tiny training sentence /' > data/tiny.txt
deberta train configs/tiny_cpu_smoke.yaml
```

The config runs a 64-wide, two-layer experimental RoPE discriminator for ten updates. Its checkpoint directory is generated under `runs/deberta-smoke/` so repeated smoke runs do not overwrite one another.

## 2) Choose a pretraining recipe

The tracked examples cover the supported training modes:

- `configs/pretrain_deberta_v3_bespoke_100k.yaml`: native DeBERTa-v3-style RTD with the BEE-spoke tokenizer, 512-token packing, and 100,000 updates.
- `configs/pretrain_rope_fineweb_edu.yaml`: experimental RoPE/RMSNorm/SwiGLU RTD.
- `configs/flashdeberta/pretrain_flashdeberta_1024.yaml`: native FlashDeBERTa at 1,024 tokens; requires the `flash` extra and a supported CUDA GPU.

Copy the closest recipe before changing its dataset, batch size, schedule, or output policy. Every accepted key and interaction is documented in the [config reference](../../configs/config_reference.yaml).

For a single-process run:

```bash
deberta train configs/pretrain_deberta_v3_bespoke_100k.yaml
```

To validate a config before committing GPU time, add `--dry-run`: it checks config contracts and
probes the tokenizer, dataset, collator, and model construction without training or writing
checkpoints (it may still download and cache tokenizer/dataset assets).

Training prints one metrics line per logging interval:

```text
step=200 | lr=9.800e-05 | loss=8.1035 | gen=6.4413 | disc=0.0332 | gain=0.0114 | acc=0.9862 | pos=0.0139 | tok/s=44712.3 | tok_seen=104857600
```

`gen` and `disc` are the per-objective losses and `loss` their weighted sum over enabled
objectives. `gain` is the number to watch for discriminator health: it is the BCE improvement over
a constant class-prior predictor, and stays at or below zero when the discriminator collapses. Raw
`acc` is dominated by the unreplaced-token majority class, and `pos` is the fraction of scored
tokens the generator actually replaced. The same reasoning drives offline checkpoint scoring; see
[Evaluating checkpoints](../guides/evaluating-checkpoints.md).

Multi-GPU training is intentionally unsupported. [Distributed training](../advanced/distributed-training.md) lists the unvalidated contracts and the experimental FSDP2 launch scaffolding. Validate the uncompiled Flash example before combining FlashDeBERTa with `torch.compile` in a supported single-process run.

## 3) Resume exact training state

Exact continuation restores model, optimizer, scheduler, RNG, global-step, and data-progress state. It is different from starting a new step-zero run with pretrained weights.
The run directory owns the materialized generator/discriminator configs and tokenizer used for construction, so continuation does not depend on the original pretrained path or Hub repository remaining unchanged.

Start from the source run's `config_resolved.yaml`, not a generic resume template:

```bash
cp runs/<project>/<run>/config_resolved.yaml resume.yaml
```

For exact same-directory continuation, edit `resume.yaml` so that:

- `train.max_steps` stays unchanged. The scheduler is defined by the original step horizon, so a
  changed value is rejected rather than silently reshaping the resumed optimization schedule.
- `train.checkpoint.resume_from_checkpoint` is `auto`.
- `train.checkpoint.resume_data_strategy` is `replay`, which reconstructs loader/worker data
  position without consuming the checkpoint-restored process RNG.
- both output directories stay unchanged.

Then run:

```bash
deberta train resume.yaml
```

To continue into a new run, point `train.checkpoint.resume_from_checkpoint` at an explicit committed `checkpoint-<step>` path and change both `train.checkpoint.output_dir` and `logging.output_dir`. Model, data, optimizer, and logging settings other than the logging output directory must remain compatible with the source snapshots; the exact current checks are documented on `train.checkpoint.resume_from_checkpoint` in the config reference.

Resume provenance is checked during preflight and again immediately before absent snapshots are copied into a new run directory. A source snapshot that changes between those checks fails the run.

## 4) Evaluate and export

Checkpoint scoring on a deterministic shuffle of the training source is covered in
[Evaluating checkpoints](../guides/evaluating-checkpoints.md). Automatic final export and manual
component export are covered in [Exporting models](../guides/exporting-models.md).
