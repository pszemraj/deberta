"""Shared export helpers for embedding merges and artifact generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from deberta.io_utils import dump_json, load_json_mapping

EXPORT_CONFIG_STRIP_KEYS = frozenset(
    {
        "hf_attention_kernel",
        "use_rmsnorm_heads",
        "cls_token_id",
        "mask_token_id",
        "sep_token_id",
        "bos_token_id",
        "eos_token_id",
    }
)
_REPO_URL = "https://github.com/pszemraj/deberta"


def clean_exported_config(config_path: Path, *, strict: bool) -> None:
    """Remove training-internal keys from exported HF ``config.json`` files.

    :param Path config_path: Path to exported ``config.json``.
    :param bool strict: Whether malformed JSON should raise ``ValueError``.
    :raises ValueError: If ``strict=True`` and ``config_path`` is malformed.
    """
    if not config_path.exists():
        return
    try:
        raw = load_json_mapping(config_path)
    except Exception as exc:
        if strict:
            raise ValueError(f"Failed to parse exported config JSON at {config_path}.") from exc
        return

    cleaned = {k: v for k, v in raw.items() if k not in EXPORT_CONFIG_STRIP_KEYS}
    dump_json(cleaned, config_path)


def write_export_readme_and_license(
    output_dir: Path,
    *,
    model_cfg: Any,
    export_config: Any | None = None,
    data_cfg: Any | None = None,
    train_cfg: Any | None = None,
    embedding_sharing: str,
) -> None:
    """Write README/Licensing artifacts for a standalone exported model directory.

    :param Path output_dir: Export destination directory.
    :param Any model_cfg: Model configuration dataclass.
    :param Any | None export_config: Optional exported backbone config (preferred source for architecture stats).
    :param Any | None data_cfg: Optional data configuration dataclass.
    :param Any | None train_cfg: Optional training configuration dataclass.
    :param str embedding_sharing: Embedding sharing mode.
    """

    def _first_int_attr(*objs: Any, attr: str) -> int:
        """Return the first int-coercible attribute value across candidate objects.

        :param Any objs: Candidate objects to inspect in order.
        :param str attr: Attribute name to read from each object.
        :return int: First coercible integer value, or 0 when none is present.
        """
        for obj in objs:
            if obj is None:
                continue
            value = getattr(obj, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return 0

    backbone = str(getattr(model_cfg, "backbone_type", "unknown"))
    runtime_cfg = export_config if export_config is not None else model_cfg
    hidden = _first_int_attr(runtime_cfg, model_cfg, attr="hidden_size")
    layers = _first_int_attr(runtime_cfg, model_cfg, attr="num_hidden_layers")
    heads = _first_int_attr(runtime_cfg, model_cfg, attr="num_attention_heads")
    seq_len = int(getattr(data_cfg, "max_seq_length", 0) or 0)
    if seq_len == 0:
        seq_len = _first_int_attr(runtime_cfg, model_cfg, attr="max_position_embeddings")
    steps = int(getattr(train_cfg, "max_steps", 0) or 0)

    if backbone == "rope":
        arch_desc = "RoPE encoder (RMSNorm, SwiGLU, rotary embeddings)"
        usage_snippet = """from transformers import AutoTokenizer
from deberta.modeling.rope_encoder import DebertaRoPEModel

model = DebertaRoPEModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = (
            "Note: RoPE exports use a custom `model_type` (`deberta-rope`) and are not currently "
            "loadable via `transformers.AutoModel.from_pretrained(...)` without custom auto-class registration."
        )
    else:
        arch_desc = "DeBERTa-v2 (disentangled attention, LayerNorm)"
        usage_snippet = """from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("path/to/this/dir")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/dir")
"""
        compatibility_note = ""

    readme = f"""---
library_name: transformers
tags:
- deberta
- encoder
- rtd
- fill-mask
license: mit
---

# {backbone}-{hidden}h-{layers}L-{heads}H

RTD-pretrained encoder ({arch_desc}).

| Parameter | Value |
|---|---|
| Backbone | `{backbone}` |
| Hidden size | {hidden} |
| Layers | {layers} |
| Attention heads | {heads} |
| Max sequence length | {seq_len} |
| Embedding sharing | `{embedding_sharing}` |
| Training steps | {steps} |

## Training

Pretrained with replaced-token detection (RTD / ELECTRA-style) using
[pszemraj/deberta]({_REPO_URL}).

## Usage

```python
{usage_snippet}
```
{compatibility_note}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    mit_license = (
        "MIT License\n\n"
        "Copyright (c) 2025-2026 Peter Szemraj\n\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        'of this software and associated documentation files (the "Software"), to deal\n'
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n\n"
        'THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n'
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE.\n"
    )
    (output_dir / "LICENSE").write_text(mit_license, encoding="utf-8")


def split_pretrainer_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split RTD pretrainer state dict into discriminator and generator prefixes.

    :param dict[str, torch.Tensor] state_dict: Full pretrainer state dict.
    :return tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: ``(disc_sd, gen_sd)``.
    """
    disc: dict[str, torch.Tensor] = {}
    gen: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("discriminator."):
            disc[k[len("discriminator.") :]] = v
        elif k.startswith("generator."):
            gen[k[len("generator.") :]] = v
    return disc, gen


def load_intersection_state_dict(
    model: Any,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = False,
    context: str = "state_dict",
) -> Any:
    """Load only keys present in both model and source state dict.

    :param Any model: Target model/module exposing ``state_dict`` and ``load_state_dict``.
    :param dict[str, torch.Tensor] state_dict: Source state dict.
    :param bool strict: When ``True``, fail on any missing/unexpected keys after overlap filtering.
    :param str context: Human-readable context included in strict-mode failures.
    :raises RuntimeError: If ``strict=True`` and the intersection load is partial.
    :return Any: ``load_state_dict`` return value.
    """
    source_keys = set(state_dict.keys())
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    incompatible = model.load_state_dict(filtered, strict=False)

    missing_model_keys = sorted(model_keys - source_keys)
    unexpected_source_keys = sorted(source_keys - model_keys)
    missing_loaded = list(getattr(incompatible, "missing_keys", []))
    unexpected_loaded = list(getattr(incompatible, "unexpected_keys", []))

    if strict and (missing_model_keys or unexpected_source_keys or missing_loaded or unexpected_loaded):
        missing_preview = ", ".join(missing_model_keys[:8])
        unexpected_preview = ", ".join(unexpected_source_keys[:8])
        loaded_missing_preview = ", ".join(missing_loaded[:8])
        loaded_unexpected_preview = ", ".join(unexpected_loaded[:8])
        raise RuntimeError(
            f"{context}: partial state_dict load rejected: "
            f"missing_model_keys={len(missing_model_keys)} ({missing_preview}), "
            f"unexpected_source_keys={len(unexpected_source_keys)} ({unexpected_preview}), "
            f"missing_after_load={len(missing_loaded)} ({loaded_missing_preview}), "
            f"unexpected_after_load={len(unexpected_loaded)} ({loaded_unexpected_preview})."
        )
    return incompatible


def merge_embeddings_into_export_backbone(
    *,
    export_model: Any,
    disc_sd: dict[str, torch.Tensor],
    gen_sd: dict[str, torch.Tensor],
    mode: str,
    fp32_accumulate: bool,
) -> None:
    """Merge generator/discriminator embedding tensors into an export backbone.

    :param Any export_model: Export backbone model.
    :param dict[str, torch.Tensor] disc_sd: Discriminator state dict.
    :param dict[str, torch.Tensor] gen_sd: Generator state dict.
    :param str mode: Embedding sharing mode.
    :param bool fp32_accumulate: Whether to add tensors in fp32 before casting.
    """
    if mode not in {"es", "gdes"}:
        return

    if not hasattr(export_model, "embeddings"):
        return

    def merge_attr(attr: str) -> None:
        """Merge one embedding attribute into the export model.

        :param str attr: Embedding attribute name.
        """
        if not hasattr(export_model.embeddings, attr):
            return
        gen_w = gen_sd.get(f"embeddings.{attr}.weight")
        if gen_w is None:
            return

        if mode == "es":
            merged = gen_w
        else:
            bias = disc_sd.get(f"embeddings.{attr}.bias")
            if bias is None:
                raise RuntimeError(f"Missing discriminator bias for embeddings.{attr}.bias (gdes)")
            if fp32_accumulate:
                merged = gen_w.detach().float() + bias.detach().float()
            else:
                merged = gen_w.to(dtype=bias.dtype) + bias

        emb_mod = getattr(export_model.embeddings, attr)
        if hasattr(emb_mod, "weight") and emb_mod.weight is not None:
            emb_mod.weight.data.copy_(merged.to(emb_mod.weight.dtype))

    for attr in ("word_embeddings", "position_embeddings", "token_type_embeddings"):
        merge_attr(attr)
