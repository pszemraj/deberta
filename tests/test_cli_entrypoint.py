from __future__ import annotations

import runpy
import sys

import pytest


def test_cli_module_invokes_main(monkeypatch: pytest.MonkeyPatch):
    # No args: if __main__ wiring is correct, argparse exits.
    monkeypatch.setattr(sys, "argv", ["deberta"])
    with pytest.raises(SystemExit):
        runpy.run_module("deberta", run_name="__main__")
