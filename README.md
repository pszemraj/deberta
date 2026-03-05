# deberta: a modern refresh

PyTorch-first DeBERTa pretraining focused on DeBERTa-v3 RTD workflows. The default path is `backbone_type=hf_deberta_v2` (native DeBERTa-v2/v3 architecture in this repo) with optional `rope` experiments.

Optional FlashDeBERTa acceleration is available as a runtime patch for the native `hf_deberta_v2` backbone. Install the extra with `pip install -e '.[flash]'` and launch training through `tools/train_flashdeberta.py`.

## Install

Use [Getting Started / Installation](docs/getting-started/installation.md).

## Train and export

Use [Getting Started / Quickstart](docs/getting-started/quickstart.md) for first runs and
[Guides / Exporting Models](docs/guides/exporting-models.md) for checkpoint consolidation and HF artifacts.

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
