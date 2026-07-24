"""Route-matrix parity gate for FlashDeBERTa attention.

Each case builds one fp32 eager reference, one bf16 eager model, and one bf16
flash model from identical weights, then asserts that flash forward and backward
error stays within the eager-bf16 error envelope. This is the authoritative
numerical gate for the flash routes; it runs whenever a CUDA device is visible.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


def _parity_module() -> types.ModuleType:
    """Import the parity harness from ``tools/``.

    :return types.ModuleType: Imported parity module.
    """

    if str(_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_TOOLS_DIR))
    return importlib.import_module("flashdeberta_parity_test")


def _case_ids() -> list[str]:
    """Return parity case names for parametrization.

    :return list[str]: Case names in suite order.
    """

    return [case.name for case in _parity_module().parity_cases()]


def test_parity_case_names_match_cli_choices() -> None:
    """The ``--case`` choices must stay in sync with the executed suite."""

    parity = _parity_module()
    assert sorted(_case_ids()) == sorted(parity._PARITY_CASE_NAMES)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashDeBERTa parity.")
@pytest.mark.parametrize("case_name", _case_ids())
def test_flash_route_matches_eager_reference(case_name: str) -> None:
    """Assert one flash route stays within the eager-bf16 error envelope.

    :param str case_name: Parity case to execute.
    """

    parity = _parity_module()
    cases = {case.name: case for case in parity.parity_cases()}
    parity.run_case(cases[case_name], device=torch.device("cuda"))
