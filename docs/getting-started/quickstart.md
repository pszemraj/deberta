# Quickstart

## 1) Tiny CPU smoke run

Create a tiny local text file and run the smoke config:

```bash
mkdir -p data
printf 'hello world\nthis is a tiny test\n' > data/tiny.txt
deberta train configs/tiny_cpu_smoke.yaml
```

## 2) Small parity-style run on FineWeb-Edu

Use the provided small parity config and shorten the run with dotted overrides. Every accepted
field and matching CLI flag is listed in the
[config reference](../../configs/config_reference.yaml):

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --train.max_steps 500 \
  --optim.scheduler.warmup_steps 50 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small \
  --logging.wandb.enabled false
```

Distributed launch (single node FSDP2 config). The template defaults to eight processes; set `num_processes` to the available GPU count as described in [Distributed training](../advanced/distributed-training.md#launch-with-accelerate).

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --train.max_steps 500 \
  --optim.scheduler.warmup_steps 50 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small_fsdp \
  --logging.wandb.enabled false
```

Optional FlashDeBERTa trial run:

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --model.hf.attention_impl flash \
  --train.max_steps 50 \
  --optim.scheduler.warmup_steps 5 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small_flash \
  --logging.wandb.enabled false
```

This command requires the `flash` extra. Route selection and hardware constraints are covered in
[FlashDeBERTa attention](../advanced/flash-attention.md).

## 3) Find the exported discriminator

The supplied config enables `train.checkpoint.export_hf_final`. Each parity command therefore attempts to write the discriminator under its distinct output directory, such as `runs/quickstart_hfv2_small/final_hf`. For manual exports, other checkpoints, or both RTD components, see [Exporting models](../guides/exporting-models.md), including the distinction between automatic partial export and strict manual export.
