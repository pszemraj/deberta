# deberta: a modern refresh

A PyTorch-first modern refresh of [DeBERTa pretraining](https://github.com/microsoft/DeBERTa), focused on [DeBERTaV3](https://arxiv.org/abs/2111.09543)-style replaced token detection with RoPE, Accelerate, and FSDP2 workflows.

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

## CLI Entrypoints

`pyproject.toml` defines installable commands:

- `deberta train` (training)
- `deberta export` (checkpoint consolidation/export)
- `python -m deberta` (module entrypoint)

Use `--help` on each command for full argument docs.

## Quickstart

Train from YAML (recommended):

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python \
  deberta train configs/pretrain_rope_fineweb_edu.yaml
```

Long-context and custom debug presets live under [`configs/`](configs/).

Export from an FSDP2 checkpoint:

```bash
accelerate launch --config_file configs/fsdp2_1node.yaml --no_python deberta export \
  runs/deberta_rope_rtd/checkpoint-10000 \
  --what discriminator \
  --output-dir runs/deberta_rope_rtd/exported_hf
```

## Documentation

- Model/backbone config and architecture behavior (including load/source resolution contract): [`docs/model.md`](docs/model.md)
- Data pipeline, packing, and masking behavior: [`docs/data.md`](docs/data.md)
- RTD objective and loss semantics: [`docs/objective.md`](docs/objective.md)
- FSDP2/runtime/export behavior: [`docs/fsdp2.md`](docs/fsdp2.md)
- Normalization strategy (`post` default, `keel` upgrade path): [`docs/norm-strategy.md`](docs/norm-strategy.md)
- Deferred follow-ups: [`docs/roadmap.md`](docs/roadmap.md)

## Configs

- Base pretraining config: [`configs/pretrain_rope_fineweb_edu.yaml`](configs/pretrain_rope_fineweb_edu.yaml)
- Long context: [`configs/pretrain_rope_fineweb_edu_2048.yaml`](configs/pretrain_rope_fineweb_edu_2048.yaml), [`configs/pretrain_rope_fineweb_edu_4096.yaml`](configs/pretrain_rope_fineweb_edu_4096.yaml)
- CPU smoke: [`configs/tiny_cpu_smoke.yaml`](configs/tiny_cpu_smoke.yaml)
- Accelerate/FSDP2: [`configs/fsdp2_1node.yaml`](configs/fsdp2_1node.yaml), [`configs/fsdp2_hf_deberta_1node.yaml`](configs/fsdp2_hf_deberta_1node.yaml)

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
