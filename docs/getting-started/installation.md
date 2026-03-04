# Installation

## Requirements

Core runtime requirements from `pyproject.toml`:

- Python `>=3.10`
- `torch>=2.9.1`
- `transformers>=4.45.0`
- `accelerate>=1.6.0`
- `datasets>=3.0.0`
- `sentencepiece>=0.2.0`

For GPU training, use a CUDA-capable PyTorch build compatible with your driver/toolkit.

## Install package

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

## Generate API markdown docs

```bash
python tools/generate_api_docs.py
```

This writes static Markdown files to `docs/api/` directly from public package docstrings.

## Verify CLI

```bash
deberta --help
deberta train --help
deberta export --help
```
