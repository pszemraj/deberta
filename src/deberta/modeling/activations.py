"""Shared activation-function resolution."""

from __future__ import annotations

from typing import Any


def get_act_fn(name_or_fn: str | Any) -> Any:
    """Resolve an activation function from ``transformers.ACT2FN``.

    :param str | Any name_or_fn: Activation name or callable.
    :raises KeyError: If an unknown activation name is provided.
    :return Any: Resolved activation object or the original non-string value.
    """
    from transformers.activations import ACT2FN

    if isinstance(name_or_fn, str):
        return ACT2FN[name_or_fn]
    return name_or_fn
