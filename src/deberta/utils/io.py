"""Shared JSON I/O helpers."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any


def dump_json(obj: Any, path: Path) -> None:
    """Write JSON to disk with stable formatting using atomic replacement.

    :param Any obj: Serializable object.
    :param Path path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object mapping from disk.

    :param Path path: Source path.
    :raises ValueError: If parsed payload is not a JSON object.
    :return dict[str, Any]: Parsed mapping.
    """

    def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Construct a JSON object while rejecting duplicate keys.

        :param list[tuple[str, Any]] pairs: Ordered object key/value pairs.
        :return dict[str, Any]: Unique-key object mapping.
        """
        mapping: dict[str, Any] = {}
        for key, value in pairs:
            if key in mapping:
                raise ValueError(f"Duplicate JSON key {key!r} at {path}.")
            mapping[key] = value
        return mapping

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(raw).__name__}.")
    return raw
