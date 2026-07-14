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
  --train.checkpoint.output_dir runs/quickstart_hfv2_small \
  --logging.wandb.enabled false
```

Distributed launch (single node FSDP2 config):

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml
```

Optional FlashDeBERTa trial run:

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --model.hf.attention_impl flash
```

This command requires the `flash` extra. Route selection and hardware constraints are covered in
[FlashDeBERTa attention](../advanced/flash-attention.md).

## 3) Find the exported discriminator

The supplied config enables `train.checkpoint.export_hf_final`. The first command therefore
attempts to write the discriminator to `runs/quickstart_hfv2_small/final_hf`; the distributed
examples use the config's `runs/hf_deberta_v2_parity_small/final_hf` unless given the same output
override. For manual exports, other checkpoints, or both RTD components, see
[Exporting models](../guides/exporting-models.md).
