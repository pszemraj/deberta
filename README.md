# deberta: a modern refresh

PyTorch-first DeBERTa pretraining focused on DeBERTa-v3 RTD workflows. The default path is `backbone_type=hf_deberta_v2` (native DeBERTa-v2/v3 architecture in this repo) with optional `rope` experiments.

## Install

```bash
git clone https://github.com/pszemraj/deberta.git
cd deberta
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[wandb]'
```

## CLI

- `deberta train`
- `deberta export`
- `python -m deberta`

Use `--help` for full flags.

## Quickstart

```bash
# config-driven parity run
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml

# preset starter (no config file)
deberta train --preset deberta-v3-base

# export one checkpoint
deberta export runs/<run_name>/checkpoint-<step> \
  --what discriminator \
  --output-dir runs/<run_name>/exported_hf
```

## Docs

Start here: [`docs/index.md`](docs/index.md)

- install: [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
- quickstart: [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md)
- configuration guide: [`docs/guides/configuration.md`](docs/guides/configuration.md)
- data pipeline: [`docs/guides/data-pipeline.md`](docs/guides/data-pipeline.md)
- exporting models: [`docs/guides/exporting-models.md`](docs/guides/exporting-models.md)
- architecture and runtime internals: [`docs/advanced/architectures.md`](docs/advanced/architectures.md), [`docs/advanced/distributed-training.md`](docs/advanced/distributed-training.md), [`docs/advanced/torch-compile.md`](docs/advanced/torch-compile.md)

Serve docs locally with MkDocs:

```bash
pip install mkdocs-material 'mkdocstrings[python]'
mkdocs serve
```

## Configs

- parity: [`configs/pretrain_hf_deberta_v2_parity_base.yaml`](configs/pretrain_hf_deberta_v2_parity_base.yaml), [`configs/pretrain_hf_deberta_v2_parity_small.yaml`](configs/pretrain_hf_deberta_v2_parity_small.yaml)
- rope: [`configs/pretrain_rope_fineweb_edu.yaml`](configs/pretrain_rope_fineweb_edu.yaml), [`configs/pretrain_rope_fineweb_edu_2048.yaml`](configs/pretrain_rope_fineweb_edu_2048.yaml), [`configs/pretrain_rope_fineweb_edu_4096.yaml`](configs/pretrain_rope_fineweb_edu_4096.yaml)
- accelerate: [`configs/accelerate/fsdp2_1node.yaml`](configs/accelerate/fsdp2_1node.yaml), [`configs/accelerate/fsdp2_hf_deberta_1node.yaml`](configs/accelerate/fsdp2_hf_deberta_1node.yaml)

RoPE exports require `deberta.modeling.rope_encoder.DebertaRoPEModel.from_pretrained(...)`.

## Citation

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
