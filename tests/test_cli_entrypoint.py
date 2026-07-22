from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def test_module_entrypoint_invokes_main(monkeypatch: pytest.MonkeyPatch):
    # No required args: if __main__ wiring is correct, argparse exits.
    monkeypatch.setattr(sys, "argv", ["deberta"])
    with pytest.raises(SystemExit):
        runpy.run_module("deberta", run_name="__main__")


def test_package_bounds_transformers_to_supported_major() -> None:
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    dependency = next(
        line.strip().removesuffix(",").strip('"')
        for line in pyproject.splitlines()
        if line.strip().startswith('"transformers')
    )

    assert dependency == "transformers>=4.45.0,<5"
