# Quickstart

## 1) Tiny CPU smoke run

Create a tiny local text file and run the smoke config:

```bash
mkdir -p data
printf 'hello world\nthis is a tiny test\n' > data/tiny.txt
deberta train configs/tiny_cpu_smoke.yaml
```

## 2) Small parity-style run on FineWeb-Edu

Use the provided small parity config and shorten the run with dotted overrides:

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --train.max_steps 500 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small \
  --logging.backend none \
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

`--model.hf.attention_impl flash` requires the flash extra (see
[Installation](installation.md#install-package)) and is valid only for the `hf_deberta_v2`
backbone. Route selection, kernel tuning, benchmarking tools, and accepted caveats are covered in
[Advanced / FlashDeBERTa attention](../advanced/flash-attention.md).

## 3) Export discriminator for downstream use

```bash
deberta export runs/quickstart_hfv2_small/checkpoint-500 \
  --what discriminator \
  --output-dir runs/quickstart_hfv2_small/exported_hf
```

## 4) Check run snapshots

Snapshot files and metadata are documented in
[Guides / Configuration](../guides/configuration.md#snapshot-files-and-reproducibility).
