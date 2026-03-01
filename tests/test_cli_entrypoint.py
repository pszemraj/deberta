from __future__ import annotations

import runpy
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "argv", "require_transformers"),
    [
        ("deberta", ["deberta"], False),
        ("deberta.export_cli", ["deberta.export_cli"], True),
    ],
)
def test_module_entrypoints_invoke_main(
    monkeypatch: pytest.MonkeyPatch, module_name: str, argv: list[str], require_transformers: bool
):
    if require_transformers:
        pytest.importorskip("transformers")

    # runpy warns when a target submodule is already imported; clear only that submodule
    # to emulate first-run CLI execution without invalidating the top-level package object.
    if "." in module_name:
        sys.modules.pop(module_name, None)
    # No required args: if __main__ wiring is correct, argparse exits.
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        runpy.run_module(module_name, run_name="__main__")
