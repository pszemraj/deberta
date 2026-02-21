from __future__ import annotations

import runpy
import sys

import pytest


def test_export_cli_module_invokes_main(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("transformers")

    # No required args: if __main__ wiring is correct, argparse exits.
    monkeypatch.setattr(sys, "argv", ["deberta.export_cli"])
    with pytest.raises(SystemExit):
        runpy.run_module("deberta.export_cli", run_name="__main__")
