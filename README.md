# deberta: a modern refresh

PyTorch-first DeBERTa pretraining focused on DeBERTa-v3 RTD workflows. The default path is
`model.backbone_type=hf_deberta_v2` (the native DeBERTa-v2/v3 architecture in this repo), with an
optional experimental `rope` backbone.

Optional FlashDeBERTa acceleration is available for the native backbone, including packed
doc-block routing. See [FlashDeBERTa attention](docs/advanced/flash-attention.md).

## Get started

Use [Installation](docs/getting-started/installation.md), then follow the
[Quickstart](docs/getting-started/quickstart.md). For checkpoint consolidation and standalone model
artifacts, see [Exporting models](docs/guides/exporting-models.md).

## Docs

The complete documentation index is at [`docs/index.md`](docs/index.md).

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
