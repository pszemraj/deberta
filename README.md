# deberta: a modern refresh

PyTorch-first DeBERTa pretraining focused on DeBERTa-v3 RTD workflows. The default path is `backbone_type=hf_deberta_v2` (native DeBERTa-v2/v3 architecture in this repo) with optional `rope` experiments.

## Install

```bash
git clone <repo-url> deberta
cd deberta
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[wandb]'
```

## Train and export

```bash
deberta train configs/pretrain_hf_deberta_v2_parity_small.yaml \
  --train.max_steps 500 \
  --train.checkpoint.output_dir runs/quickstart_hfv2_small

deberta export runs/quickstart_hfv2_small/checkpoint-500 \
  --what discriminator \
  --output-dir runs/quickstart_hfv2_small/exported_hf
```

By default successful training also performs a final export pass into
`<train.checkpoint.output_dir>/final_hf` (`train.checkpoint.export_hf_final=true`).

## Docs

Start with [`docs/index.md`](docs/index.md), then use:
- [Getting Started / Installation](docs/getting-started/installation.md)
- [Getting Started / Quickstart](docs/getting-started/quickstart.md)
- [Guides / Configuration](docs/guides/configuration.md)

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
