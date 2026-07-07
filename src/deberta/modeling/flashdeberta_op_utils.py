"""Shared torch/CUDA infrastructure helpers for FlashDeBERTa custom-op modules."""

from __future__ import annotations

from typing import Any

import torch


def lookup_registered_op(namespace: str, name: str) -> Any | None:
    """Return a previously registered custom op overload, if one exists.

    :param str namespace: Operator namespace.
    :param str name: Operator name.
    :return Any | None: Registered overload or ``None``.
    """

    ns = getattr(torch.ops, namespace, None)
    if ns is None or not hasattr(ns, name):
        return None
    op = getattr(ns, name)
    return getattr(op, "default", op)


def device_compute_capability(device: torch.device) -> tuple[int, int]:
    """Return CUDA compute capability for one device.

    :param torch.device device: Device to query.
    :return tuple[int, int]: ``(major, minor)`` capability, or ``(0, 0)`` off CUDA.
    """

    if device.type != "cuda":
        return (0, 0)
    index = device.index
    if index is None:
        return torch.cuda.get_device_capability()
    return torch.cuda.get_device_capability(index)
