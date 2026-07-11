# Installation

## Requirements

Python `>=3.10` is required. Package installation resolves the runtime dependencies declared in
[`pyproject.toml`](../../pyproject.toml).

For GPU training, use a CUDA-capable PyTorch build compatible with your driver/toolkit.

## Install package

```bash
git clone https://github.com/pszemraj/deberta.git deberta
cd deberta
pip install -e .
```

Optional extras:

```bash
pip install -e '.[dev]'
pip install -e '.[wandb]'
pip install -e '.[flash]'
```

The `flash` extra installs the optional FlashDeBERTa and Triton dependencies used by the native
`hf_deberta_v2` backbone. See [FlashDeBERTa attention](../advanced/flash-attention.md) for runtime
requirements and configuration.

## Verify CLI

```bash
deberta --help
deberta train --help
deberta export --help
```

Continue with the [Quickstart](quickstart.md).
