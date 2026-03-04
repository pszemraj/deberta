# deberta documentation

## Start here

- install and environment setup: [Getting Started / Installation](getting-started/installation.md)
- first run on a small config: [Getting Started / Quickstart](getting-started/quickstart.md)

## Guides

- config loading, precedence, defaults, and overrides: [Guides / Configuration](guides/configuration.md)
- streaming packing, doc-block masking, and collator behavior: [Guides / Data Pipeline](guides/data-pipeline.md)
- checkpoint consolidation and Hugging Face export: [Guides / Exporting Models](guides/exporting-models.md)

## Advanced

- `hf_deberta_v2` vs `rope` architecture behavior: [Advanced / Architectures](advanced/architectures.md)
- accelerate/FSDP usage, resume behavior, and token-weighted GA: [Advanced / Distributed Training](advanced/distributed-training.md)
- compile scopes and graph-stability guidance: [Advanced / torch.compile](advanced/torch-compile.md)

## API reference

- [API / Modeling](api/modeling.md)
- [API / Data](api/data.md)
- [API / Training](api/training.md)

Regenerate API pages from package docstrings with:

```bash
python tools/generate_api_docs.py
```
