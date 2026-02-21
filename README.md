# deberta: a modern refresh

A PyTorch-first modern refresh of [DeBERTa pretraining](https://github.com/microsoft/DeBERTa), focused on [DeBERTaV3](https://arxiv.org/abs/2111.09543)-style replaced token detection with RoPE, Accelerate, and FSDP2 workflows.

---

- [Install](#install)
- [CLI Entrypoints](#cli-entrypoints)
- [Quickstart](#quickstart)
- [Documentation](#documentation)
- [Configs](#configs)
- [Repo Layout](#repo-layout)
- [Citations](#citations)

---

## Install

Clone + editable install:

```bash
git clone https://github.com/pszemraj/deberta.git
cd deberta
# activate your Python environment first, then:
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[wandb]'
```

Direct install from GitHub:

```bash
pip install "git+https://github.com/pszemraj/deberta.git"
```

## CLI Entrypoints

`pyproject.toml` defines installable commands:

- `deberta train` (training)
- `deberta export` (checkpoint consolidation/export)

Use `--help` on each command for full argument docs.

## Quickstart

Train from YAML (recommended):

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python \
  deberta train configs/pretrain_rope_c4_en.yaml
```

Long-context presets:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python \
  deberta train configs/pretrain_rope_c4_en_2048.yaml
```

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python \
  deberta train configs/pretrain_rope_c4_en_4096.yaml
```

Export from an FSDP2 checkpoint:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python deberta export \
  runs/deberta_rope_rtd/checkpoint-10000 \
  --what discriminator \
  --output-dir runs/deberta_rope_rtd/exported_hf
```

## Documentation

- Model/backbone options (`rope` vs `hf_deberta_v2`, `swiglu`, KEEL toggles, embedding sharing): [`docs/model.md`](docs/model.md)
- Data loading, packing, masking, and SDPA/attention-mask behavior: [`docs/data.md`](docs/data.md)
- RTD objective, loss terms, and generator masked-only logits path: [`docs/objective.md`](docs/objective.md)
- FSDP2 setup, precision/runtime knobs (`bf16` autocast, TF32, `torch.compile`, SDPA policy), and export flow: [`docs/fsdp2.md`](docs/fsdp2.md)
- KEEL architecture paper notes and rationale: [`docs/keel-paper-technical-overview.md`](docs/keel-paper-technical-overview.md)

## Configs

- Base pretraining config: [`configs/pretrain_rope_c4_en.yaml`](configs/pretrain_rope_c4_en.yaml)
- Long context: [`configs/pretrain_rope_c4_en_2048.yaml`](configs/pretrain_rope_c4_en_2048.yaml), [`configs/pretrain_rope_c4_en_4096.yaml`](configs/pretrain_rope_c4_en_4096.yaml)
- CPU smoke: [`configs/tiny_cpu_smoke.yaml`](configs/tiny_cpu_smoke.yaml)
- Accelerate/FSDP2: [`configs/fsdp2_1node.yaml`](configs/fsdp2_1node.yaml), [`configs/fsdp2_hf_deberta_1node.yaml`](configs/fsdp2_hf_deberta_1node.yaml)

## Repo Layout

- `src/deberta/` - package source
- `src/deberta/training/pretrain.py` - training loop
- `src/deberta/export_cli.py` - FSDP-aware exporter
- `configs/` - training and accelerate configs
- `docs/` - canonical docs by concept

## Citations

```bibtex
@misc{he2021debertav3,
      title={DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
      author={Pengcheng He and Jianfeng Gao and Weizhu Chen},
      year={2021},
      eprint={2111.09543},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
