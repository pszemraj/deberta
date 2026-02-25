"""Shared JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dump_json(obj: Any, path: Path) -> None:
    """Write JSON to disk with stable formatting.

    :param Any obj: Serializable object.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object mapping from disk.

    :param Path path: Source path.
    :raises ValueError: If parsed payload is not a JSON object.
    :return dict[str, Any]: Parsed mapping.
    """
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(raw).__name__}.")
    return raw
