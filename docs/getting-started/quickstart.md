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

For single-node FSDP2, set `num_processes` in the architecture-specific Accelerate file to the available GPU count, then launch:

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_deberta_v3_bespoke_100k.yaml
```

See [Distributed training](../advanced/distributed-training.md) for the native and RoPE launcher distinction. Validate the uncompiled Flash example before combining FlashDeBERTa with `torch.compile`.

## 3) Resume exact training state

Exact continuation restores model, optimizer, scheduler, RNG, global-step, and data-progress state. It is different from starting a new step-zero run with pretrained weights.

Start from the source run's `config_resolved.yaml`, not a generic resume template:

```bash
cp runs/<project>/<run>/config_resolved.yaml resume.yaml
```

For same-directory continuation, increase `train.max_steps`, set `train.checkpoint.resume_from_checkpoint` to `auto`, and leave both output directories unchanged. Then run:

```bash
deberta train resume.yaml
```

To continue into a new run, use an explicit committed `checkpoint-<step>` path and change both `train.checkpoint.output_dir` and `logging.output_dir`. Model, data, optimizer, and logging settings other than the logging output directory must remain compatible with the source snapshots; the exact current checks are documented on `train.checkpoint.resume_from_checkpoint` in the config reference.

## 4) Find or export the discriminator

When `train.checkpoint.export_hf_final` is enabled, training requires the checkpoint at the final global step and strictly exports its discriminator under `<output_dir>/final_hf`; a missing checkpoint or export failure fails the training command. For another checkpoint, the generator, or both components, see [Exporting models](../guides/exporting-models.md).
