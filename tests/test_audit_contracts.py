from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_audit_contracts_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "tools" / "audit_contracts.py"
    spec = importlib.util.spec_from_file_location("audit_contracts_tool", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rtd_loss_integrity_contract_check_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_audit_contracts_module()

    result = module.check_rtd_loss_integrity(repo_root=repo_root, verbose=False)
    assert result.status == "PASS", result.details
