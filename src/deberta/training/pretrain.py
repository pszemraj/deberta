"""Backward-compatible alias for training entrypoints and helpers."""

from __future__ import annotations

import sys

from deberta.training import entrypoint as _entrypoint

sys.modules[__name__] = _entrypoint
