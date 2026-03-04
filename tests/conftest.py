from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def mock_checkpoint(tmp_path: Path):
    """Return a helper that creates checkpoint-like directories for tests."""

    def _make(
        *,
        root: Path | None = None,
        name: str = "checkpoint-1",
        with_weights: bool = True,
        with_data_state: bool = True,
        with_complete: bool = True,
        consumed_micro_batches: int = 0,
        data_state_extra: dict[str, Any] | None = None,
    ) -> Path:
        parent = Path(root) if root is not None else tmp_path
        ckpt = parent / str(name)
        ckpt.mkdir(parents=True, exist_ok=True)
        if with_weights:
            (ckpt / "model.safetensors").write_bytes(b"weights")
        if with_data_state:
            payload: dict[str, Any] = {"consumed_micro_batches": int(consumed_micro_batches)}
            if data_state_extra:
                payload.update(dict(data_state_extra))
            (ckpt / "data_state.json").write_text(json.dumps(payload), encoding="utf-8")
        if with_complete:
            (ckpt / ".complete").write_text("ok\n", encoding="utf-8")
        return ckpt

    return _make
