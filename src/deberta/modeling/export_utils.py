"""Shared export helpers for embedding-merge behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

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


def clean_exported_config(config_path: Path, *, strict: bool) -> None:
    """Remove training-internal keys from exported HF ``config.json`` files.

    :param Path config_path: Path to exported ``config.json``.
    :param bool strict: Whether malformed JSON should raise ``ValueError``.
    :raises ValueError: If ``strict=True`` and ``config_path`` is malformed.
    """
    if not config_path.exists():
        return
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object at {config_path}, got {type(raw).__name__}.")
    except Exception as exc:
        if strict:
            raise ValueError(f"Failed to parse exported config JSON at {config_path}.") from exc
        return

    cleaned = {k: v for k, v in raw.items() if k not in EXPORT_CONFIG_STRIP_KEYS}
    config_path.write_text(json.dumps(cleaned, indent=2) + "\n", encoding="utf-8")


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

    merge_attr("word_embeddings")
    merge_attr("position_embeddings")
    merge_attr("token_type_embeddings")
