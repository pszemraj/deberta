"""Shared mapping transformation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten nested mappings into a dotted leaf mapping.

    :param Mapping[str, Any] mapping: Input nested mapping.
    :param str prefix: Optional dotted prefix.
    :param str sep: Path separator between nested keys.
    :return dict[str, Any]: Flattened mapping of dotted leaf keys to values.
    """
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        path = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_mapping(value, prefix=path, sep=sep))
        else:
            out[str(path)] = value
    return out
