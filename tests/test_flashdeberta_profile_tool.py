from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


def test_timed_phase_synchronizes_cuda_at_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools_dir))
    profile_mod = importlib.import_module("flashdeberta_rtd_profile")

    events: list[str] = []

    class _Record:
        def __enter__(self) -> None:
            events.append("record_enter")

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            del exc_type, exc, tb
            events.append("record_exit")

    timestamps = iter((1.0, 1.125))

    def _clock() -> float:
        events.append("clock")
        return next(timestamps)

    monkeypatch.setattr(profile_mod.torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(profile_mod.torch.profiler, "record_function", lambda _name: _Record())
    monkeypatch.setattr(profile_mod.time, "perf_counter", _clock)

    phase_times_ms: dict[str, list[float]] = defaultdict(list)
    with profile_mod._TimedPhase("forward", phase_times_ms):
        events.append("body")

    assert events == ["sync", "record_enter", "clock", "body", "sync", "clock", "record_exit"]
    assert phase_times_ms == {"forward": [125.0]}
