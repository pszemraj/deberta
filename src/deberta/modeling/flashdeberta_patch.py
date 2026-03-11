"""Runtime patch helpers for enabling FlashDeBERTa in this repository.

The patch stays intentionally narrow:
- swap native ``DisentangledSelfAttention`` with the flash-backed adapter
- stop encoder-level ``(S,S)`` relative-position materialization
- keep RTD EMD masks broadcastable when they are plain padding masks
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any

from deberta.modeling.mask_utils import normalize_keep_mask


def _emd_attention_mask_passthrough(attention_mask: Any) -> Any:
    """Preserve broadcast masks for EMD instead of forcing quadratic expansion.

    2D padding masks stay in ``(B,1,1,S)`` form, while true pairwise masks keep
    their ``(B,1,S,S)`` semantics.

    :param Any attention_mask: Input mask tensor.
    :raises ValueError: If the mask rank is unsupported.
    :return Any: Normalized broadcast or pairwise keep mask.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 2:
        return mask[:, None, None, :]
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        if int(mask.shape[1]) == 1:
            return mask
        return mask.any(dim=1, keepdim=True)
    raise ValueError(f"Unsupported attention_mask rank for EMD: {mask.ndim}")


def enable_flashdeberta_attention(*, strict: bool = True) -> None:
    """Enable FlashDeBERTa-backed attention via runtime monkey patches.

    :param bool strict: Whether missing FlashDeBERTa support should raise.
    :raises RuntimeError: If strict mode is enabled and FlashDeBERTa cannot be imported.
    """

    try:
        importlib.import_module("flashdeberta")
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "flashdeberta is not installed. Install the optional flash runtime with: pip install flashdeberta triton"
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

    dv2 = importlib.import_module("deberta.modeling.deberta_v2_native")
    rtd = importlib.import_module("deberta.modeling.rtd")

    if getattr(dv2, "_FLASHDEBERTA_ENABLED", False):
        return

    dv2._FLASHDEBERTA_ORIG_DisentangledSelfAttention = dv2.DisentangledSelfAttention
    dv2.DisentangledSelfAttention = flash_mod.FlashDisentangledSelfAttention  # type: ignore[assignment]

    if hasattr(dv2, "DebertaV2Encoder") and hasattr(dv2.DebertaV2Encoder, "get_rel_pos"):
        dv2._FLASHDEBERTA_ORIG_get_rel_pos = dv2.DebertaV2Encoder.get_rel_pos

        def _get_rel_pos_passthrough(
            self: Any,
            hidden_states: Any,
            query_states: Any = None,
            relative_pos: Any = None,
        ) -> Any:
            """Return caller-supplied relative positions or ``None``.

            :param Any hidden_states: Unused hidden states.
            :param Any query_states: Unused query states.
            :param Any relative_pos: Optional precomputed relative positions.
            :return Any: The caller-provided tensor or ``None``.
            """

            del self, hidden_states, query_states
            return relative_pos

        dv2.DebertaV2Encoder.get_rel_pos = _get_rel_pos_passthrough  # type: ignore[assignment]

    if hasattr(rtd, "_ensure_emd_pairwise_attention_mask"):
        rtd._FLASHDEBERTA_ORIG_ensure_emd_pairwise_attention_mask = rtd._ensure_emd_pairwise_attention_mask
        rtd._ensure_emd_pairwise_attention_mask = _emd_attention_mask_passthrough

    dv2._FLASHDEBERTA_ENABLED = True
    rtd._FLASHDEBERTA_ENABLED = True

    if os.environ.get("FLASHDEBERTA_SHOW_CONFIG_WARNINGS", "0").strip().lower() not in {"1", "true", "yes"}:
        warnings.filterwarnings("ignore", message=r"INFO: Fixed-length forward config is.*")
        warnings.filterwarnings("ignore", message=r"INFO: Fixed-length backward config is.*")
        warnings.filterwarnings("ignore", message=r"INFO: Variable-length forward config is.*")
        warnings.filterwarnings("ignore", message=r"INFO: Varlen backward config -> .*")


def disable_flashdeberta_attention() -> None:
    """Undo previously applied FlashDeBERTa runtime patches."""

    dv2 = importlib.import_module("deberta.modeling.deberta_v2_native")
    rtd = importlib.import_module("deberta.modeling.rtd")

    orig_attn = getattr(dv2, "_FLASHDEBERTA_ORIG_DisentangledSelfAttention", None)
    if orig_attn is not None:
        dv2.DisentangledSelfAttention = orig_attn  # type: ignore[assignment]

    orig_get_rel_pos = getattr(dv2, "_FLASHDEBERTA_ORIG_get_rel_pos", None)
    if orig_get_rel_pos is not None and hasattr(dv2, "DebertaV2Encoder"):
        dv2.DebertaV2Encoder.get_rel_pos = orig_get_rel_pos  # type: ignore[assignment]

    orig_emd_mask = getattr(rtd, "_FLASHDEBERTA_ORIG_ensure_emd_pairwise_attention_mask", None)
    if orig_emd_mask is not None:
        rtd._ensure_emd_pairwise_attention_mask = orig_emd_mask

    dv2._FLASHDEBERTA_ENABLED = False
    rtd._FLASHDEBERTA_ENABLED = False


__all__ = ["disable_flashdeberta_attention", "enable_flashdeberta_attention"]
