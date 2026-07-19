# deberta documentation

## Start here

- install and environment setup: [Getting Started / Installation](getting-started/installation.md)
- first run on a small config: [Getting Started / Quickstart](getting-started/quickstart.md)

## Guides

- complete per-key training config contract: [Config reference](../configs/config_reference.yaml)
- dataset loading and packing, doc-block masking, and collator behavior: [Guides / Data Pipeline](guides/data-pipeline.md)
- checkpoint consolidation and Hugging Face export: [Guides / Exporting Models](guides/exporting-models.md)

## Advanced

- `hf_deberta_v2` vs `rope` architecture behavior: [Advanced / Architectures](advanced/architectures.md)
- Accelerate and FSDP2 launch behavior: [Advanced / Distributed Training](advanced/distributed-training.md)
- compile scopes and graph-stability guidance: [Advanced / torch.compile](advanced/torch-compile.md)
- flash routes, kernel tuning, and caveats: [Advanced / FlashDeBERTa attention](advanced/flash-attention.md)
- per-GPU expectations and tuning for non-`sm_120` hardware: [Advanced / GPU support](advanced/gpu-support.md)

## API reference

- [API / Modeling](api/modeling.md)
- [API / Data](api/data.md)
- [API / Training](api/training.md)
