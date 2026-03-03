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
  --override train.max_steps=500 \
  --override train.output_dir=runs/quickstart_hfv2_small \
  --override train.report_to=none
```

Distributed launch (single node FSDP2 config):

```bash
accelerate launch --config_file configs/accelerate/fsdp2_hf_deberta_1node.yaml --no_python \
  deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml
```

## 3) Export discriminator for downstream use

```bash
deberta export runs/quickstart_hfv2_small/checkpoint-500 \
  --what discriminator \
  --output-dir runs/quickstart_hfv2_small/exported_hf
```

## 4) Check run snapshots

Each run directory includes:

- `config_original.yaml`
- `config_resolved.yaml`
- `model_config.json`
- `data_config.json`
