from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generator() -> ModuleType:
    """Load the config-reference generator module."""

    generator_path = REPO_ROOT / "tools" / "generate_config_reference.py"
    spec = importlib.util.spec_from_file_location("generate_config_reference", generator_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_reference_matches_generator() -> None:
    """Ensure the checked-in config reference is generated, not hand-maintained."""

    generator = _load_generator()
    expected = generator.render_config_reference()
    actual = (REPO_ROOT / "docs" / "guides" / "config-reference.md").read_text(encoding="utf-8")
    assert actual == expected
