#!/usr/bin/env python3
"""Launch the standard training CLI with FlashDeBERTa runtime patches enabled."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """Add the repository ``src/`` directory to ``sys.path`` for direct script execution."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from deberta.modeling.flashdeberta_patch import enable_flashdeberta_attention  # noqa: E402

enable_flashdeberta_attention(strict=True)

from deberta.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
