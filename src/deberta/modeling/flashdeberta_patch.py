"""Compatibility helpers for the old FlashDeBERTa patch entrypoint.

FlashDeBERTa is selected by ``model.hf.attention_impl=flash`` at module
construction time. These functions remain only so older local tools fail or
validate cleanly instead of mutating process-global model classes.
"""

from __future__ import annotations

import importlib

from deberta.modeling.flashdeberta_version import require_flashdeberta_version


def enable_flashdeberta_attention(*, strict: bool = True) -> None:
    """Validate that FlashDeBERTa can be constructed by config.

    :param bool strict: Whether missing FlashDeBERTa support should raise.
    :raises RuntimeError: If strict mode is enabled and FlashDeBERTa cannot be imported.
    """

    try:
        importlib.import_module("flashdeberta")
        require_flashdeberta_version()
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "flashdeberta is unavailable or not at the supported pin. "
                "Install the optional flash runtime with: pip install -e '.[flash]'. "
                "Enable it with: deberta train ... --model.hf.attention_impl flash"
            ) from exc
        return

    flash_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
    import_error = flash_mod.flashdeberta_import_error()
    if import_error is not None:
        if strict:
            raise RuntimeError(
                "flashdeberta was found but its flash attention operator failed to import."
            ) from import_error
        return


def disable_flashdeberta_attention() -> None:
    """Compatibility no-op for the removed runtime patch path."""


__all__ = ["disable_flashdeberta_attention", "enable_flashdeberta_attention"]
