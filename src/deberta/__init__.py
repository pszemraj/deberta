"""deberta package: modern refresh for RTD/ELECTRA-style pretraining."""

from __future__ import annotations

__all__ = ["__version__"]

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    try:
        from importlib.metadata import PackageNotFoundError, version

        __version__ = version("deberta")
    except PackageNotFoundError:
        __version__ = "0+unknown"
