"""Shared export helpers for embedding-merge behavior."""

from __future__ import annotations

from typing import Any

import torch


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
