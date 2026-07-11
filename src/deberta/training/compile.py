"""Torch compile and attention-mask helpers for pretraining."""

from __future__ import annotations

import dataclasses
import logging
import os
import types
from collections.abc import Callable
from typing import Any

import torch

from deberta.config import ModelConfig, _normalize_sdpa_kernel
from deberta.data.batch_contract import CPU_SCALAR_BATCH_KEYS, FLASH_SCALAR_BATCH_KEYS
from deberta.modeling.flash_config import flash_cfg_bool, flash_cfg_get, flash_cfg_optional_int
from deberta.modeling.flashdeberta_kernel_tuning import (
    compute_capability_key,
    configure_flashdeberta_kernel_overrides,
    flash_padding_route,
    flash_route_choice,
    flash_seq_bucket,
    tuned_capability_keys,
)
from deberta.modeling.flashdeberta_op_utils import device_compute_capability
from deberta.modeling.mask_utils import (
    FlashBatchMeta,
    build_doc_block_mask,
    build_doc_segment_metadata,
    build_validated_prefix_lengths,
    doc_segment_metadata_host_stats,
    is_pairwise_mask,
    mask_to_2d_keep_mask,
    validate_doc_segments_against_mask,
)

logger = logging.getLogger(__name__)


def _maybe_enable_tf32(enabled: bool) -> None:
    """Configure TF32 compute policy for CUDA matmul/cudnn.

    :param bool enabled: Whether to enable TF32.
    """
    torch.backends.cuda.matmul.allow_tf32 = bool(enabled)
    torch.backends.cudnn.allow_tf32 = bool(enabled)


def _maybe_configure_sdpa_kernels(policy: str, *, is_main: bool) -> None:
    """Configure PyTorch SDPA backend toggles on CUDA.

    :param str policy: SDPA kernel policy.
    :param bool is_main: Whether current process should emit logs.
    """
    if not torch.cuda.is_available():
        return

    policy = _normalize_sdpa_kernel(policy)

    enable_flash = True
    enable_mem_efficient = True
    enable_math = True

    if policy == "flash":
        enable_mem_efficient = False
        enable_math = False
    elif policy == "mem_efficient":
        enable_flash = False
    elif policy == "math":
        enable_flash = False
        enable_mem_efficient = False

    try:
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is not None:
            for name, enabled in (
                ("enable_flash_sdp", enable_flash),
                ("enable_mem_efficient_sdp", enable_mem_efficient),
                ("enable_math_sdp", enable_math),
            ):
                fn = getattr(cuda_backend, name, None)
                if callable(fn):
                    fn(bool(enabled))
    except Exception:
        # Best-effort only; do not fail training if this backend API changes.
        pass

    if is_main:
        logger.info(
            "SDPA kernel policy=%s (requested flash=%s, mem_efficient=%s, math=%s).",
            policy,
            enable_flash,
            enable_mem_efficient,
            enable_math,
        )


def _bf16_runtime_sanity_check() -> bool:
    """Check whether bf16 autocast executes a tiny CUDA matmul.

    :return bool: True when a tiny bf16 autocast path succeeds.
    """
    if not torch.cuda.is_available():
        logger.error("bf16 mixed precision requested but CUDA is not available.")
        return False
    if not torch.cuda.is_bf16_supported():
        logger.error("bf16 mixed precision requested but this CUDA device reports no bf16 support.")
        return False

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            a = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            b = torch.randn((64, 64), device="cuda", dtype=torch.float32)
            c = a @ b
            _ = c.sum().item()
        return True
    except Exception:
        logger.error("bf16 autocast preflight failed.", exc_info=True)
        return False


def _flash_route_hint_for_padding_batch(
    *,
    seq_len: int,
    active_tokens: int,
    batch_size: int,
    flash_cfg: Any | None = None,
    device: torch.device | None = None,
) -> str:
    """Select a fixed-vs-varlen route for one standard padded batch.

    :param int seq_len: Padded sequence length.
    :param int active_tokens: Total active tokens across the batch.
    :param int batch_size: Batch size.
    :param Any | None flash_cfg: Optional resolved flash config.
    :param torch.device | None device: Batch device for capability-scoped table rows.
    :return str: Either ``fixed`` or ``varlen``.
    """

    return flash_padding_route(
        seq_len=int(seq_len),
        total_tokens=int(active_tokens),
        batch_size=int(batch_size),
        force_varlen=flash_cfg_bool(flash_cfg, name="force_varlen", default="0"),
        varlen_min_seq_len=flash_cfg_optional_int(flash_cfg, name="varlen_min_seq_len", default=None),
        compute_capability=device_compute_capability(device) if device is not None else None,
    )


def _flash_route_hint_for_docblock_batch(
    *,
    seq_len: int,
    batch_size: int | None = None,
    flash_cfg: Any | None = None,
    device: torch.device | None = None,
) -> str:
    """Select the doc-block flash backend for one packed batch.

    The default policy comes from the repo-local JSON route table. Measured
    packed RTD sequence buckets choose dense ``docblock_bias`` on GPUs with
    matching capability-scoped rows (shipped: ``sm_120``); other hardware
    defaults to the segment-aware ragged ``docblock`` route. The dense rows
    carry ``max_seq_len``/``max_batch_size`` bounds because the route saves a
    ``(B,H,S,S)`` bias for backward; out-of-bounds batches stay ragged. Set
    ``docblock_bias_seq_len`` to force dense at one exact length on any GPU
    regardless of bounds, or ``0`` to force ragged for ablations.

    :param int seq_len: Packed sequence length.
    :param int | None batch_size: Packed batch size, when known.
    :param Any | None flash_cfg: Optional resolved flash config.
    :param torch.device | None device: Batch device for capability-scoped table rows.
    :return str: Either ``docblock_bias`` or ``docblock``.
    """

    override_bias_seq_len = flash_cfg_optional_int(
        flash_cfg,
        name="docblock_bias_seq_len",
        default=None,
    )
    if override_bias_seq_len is not None:
        if int(override_bias_seq_len) > 0 and int(seq_len) == int(override_bias_seq_len):
            return "docblock_bias"
        return "docblock"

    seq_bucket = flash_seq_bucket(seq_len=int(seq_len))
    table_route = flash_route_choice(
        policy="docblock",
        seq_bucket=seq_bucket,
        compute_capability=device_compute_capability(device) if device is not None else None,
        seq_len=int(seq_len),
        batch_size=int(batch_size) if batch_size is not None else None,
    )
    if table_route in {"docblock", "docblock_bias"}:
        return table_route
    return "docblock"


def _is_main_process() -> bool:
    """Check whether this process should emit once-per-process flash notices.

    Launchers (accelerate, torchrun) export ``RANK``; a single-process run has
    no ``RANK`` and counts as main. This keeps host-side flash warnings from
    repeating once per rank in distributed training.

    :return bool: True for rank 0 or unlaunched processes.
    """

    return os.environ.get("RANK", "0").strip() in {"", "0"}


_UNTUNED_FLASH_HARDWARE_NOTICED: set[str] = set()


def _notice_untuned_flash_hardware_once(device: torch.device) -> None:
    """Log once per GPU class when flash runs without measured tuning rows.

    :param torch.device device: Device hosting the flash batch.
    """

    if device.type != "cuda" or not _is_main_process():
        return
    key = compute_capability_key(device_compute_capability(device))
    if key in _UNTUNED_FLASH_HARDWARE_NOTICED:
        return
    _UNTUNED_FLASH_HARDWARE_NOTICED.add(key)
    if key in tuned_capability_keys():
        return
    logger.warning(
        "FlashDeBERTa has no measured tuning rows for %s; flash stays enabled with "
        "hardware-agnostic route defaults and generic kernel configs. "
        "See docs/advanced/gpu-support.md to tune this GPU.",
        key,
    )


_NON_PREFIX_PADDING_FALLBACK_NOTICED = False


def _notice_non_prefix_padding_fallback_once() -> None:
    """Warn once per process when non-prefix padding masks keep batches on eager.

    Batch preparation runs host-side outside compiled graphs, so this is the
    one fallback signal that survives compiled training; the per-layer
    fallback warnings are skipped inside compiled forwards by design.
    """

    global _NON_PREFIX_PADDING_FALLBACK_NOTICED
    if _NON_PREFIX_PADDING_FALLBACK_NOTICED or not _is_main_process():
        return
    _NON_PREFIX_PADDING_FALLBACK_NOTICED = True
    logger.warning(
        "FlashDeBERTa batch preparation saw a padding mask with holes or left "
        "padding; such batches run eager attention (throughput drops for them). "
        "This is logged once per process; set model.hf.flash.debug_stats=true in "
        "an uncompiled run to count occurrences."
    )


_DOCBLOCK_ROUTE_NOTICED: set[tuple[str, int, int]] = set()


def _notice_docblock_route_once(
    *,
    route_hint: str,
    seq_len: int,
    batch_size: int,
    device: torch.device | None,
    flash_cfg: Any | None = None,
) -> None:
    """Log the doc-block route decision once per batch shape.

    Route selection depends on batch shape, so a config change (batch size,
    sequence length) can silently flip dense ``docblock_bias`` to ragged
    ``docblock`` and cost the measured speedup. Batch preparation runs
    host-side outside compiled graphs, so this signal survives compiled
    training where the per-layer ``model.hf.flash.debug_stats`` counters are
    no-ops.

    :param str route_hint: Selected doc-block route for this shape.
    :param int seq_len: Packed sequence length.
    :param int batch_size: Packed batch size.
    :param torch.device | None device: Batch device for capability lookup.
    :param Any | None flash_cfg: Optional resolved flash config for override detection.
    """

    if not _is_main_process():
        return
    key = (str(route_hint), int(seq_len), int(batch_size))
    if key in _DOCBLOCK_ROUTE_NOTICED:
        return
    _DOCBLOCK_ROUTE_NOTICED.add(key)
    seq_bucket = flash_seq_bucket(seq_len=int(seq_len))
    compute_capability = device_compute_capability(device) if device is not None else None
    if route_hint == "docblock_bias":
        override_bias_seq_len = flash_cfg_optional_int(flash_cfg, name="docblock_bias_seq_len", default=None)
        if override_bias_seq_len is not None:
            bounded_route = flash_route_choice(
                policy="docblock",
                seq_bucket=seq_bucket,
                compute_capability=compute_capability,
                seq_len=int(seq_len),
                batch_size=int(batch_size),
            )
            if bounded_route != "docblock_bias":
                logger.warning(
                    "model.hf.flash.docblock_bias_seq_len forces the dense doc-block route "
                    "for (B=%d, S=%d), a shape the tuning table would keep ragged - the knob "
                    "bypasses the table's max_batch_size/max_seq_len bounds and the dense "
                    "route saves a (B,H,S,S) bias for backward. Clear the knob or verify "
                    "memory headroom for this shape.",
                    batch_size,
                    seq_len,
                )
                return
    if route_hint != "docblock_bias":
        unbounded_route = flash_route_choice(
            policy="docblock",
            seq_bucket=seq_bucket,
            compute_capability=compute_capability,
        )
        if unbounded_route == "docblock_bias":
            logger.warning(
                "FlashDeBERTa keeps packed doc-block batches (B=%d, S=%d) on the ragged "
                "'docblock' route: the tuning table's dense row exists for this shape but "
                "its max_batch_size/max_seq_len bounds exclude the batch (the dense route "
                "saves a (B,H,S,S) bias for backward). If this GPU has memory headroom, "
                "raise the row bounds via model.hf.flash.kernel_overrides_path; see "
                "docs/advanced/gpu-support.md.",
                batch_size,
                seq_len,
            )
            return
    logger.info(
        "FlashDeBERTa doc-block route for packed batches (B=%d, S=%d): %s (logged once per shape).",
        batch_size,
        seq_len,
        route_hint,
    )


def _configure_flash_kernel_overrides_from_cfg(flash_cfg: Any | None) -> None:
    """Apply config-driven FlashDeBERTa kernel override tables for route helpers.

    :param Any | None flash_cfg: Optional flash config object or mapping.
    """

    if flash_cfg is None:
        return
    value = flash_cfg_get(flash_cfg, "kernel_overrides_path", None)
    configure_flashdeberta_kernel_overrides(str(value).strip() if value is not None else None)


def _flash_active_tokens_host(value: Any) -> int | None:
    """Return a host active-token count when one is already available.

    :param Any value: Candidate scalar from the batch.
    :return int | None: Host integer or ``None``.
    """

    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, torch.Tensor) and value.ndim == 0 and value.device.type == "cpu":
        return int(value)
    raise ValueError("flash_active_tokens must be a Python int or CPU scalar tensor.")


def _flash_scalar_tensor(value: Any) -> torch.Tensor | None:
    """Return a CPU scalar tensor when one is already available.

    :param Any value: Candidate scalar tensor from the batch.
    :return torch.Tensor | None: CPU scalar int tensor or ``None``.
    """

    return (
        value if isinstance(value, torch.Tensor) and value.ndim == 0 and value.device.type == "cpu" else None
    )


def _cpu_int_scalar(value: int | None) -> torch.Tensor | None:
    """Return a CPU int32 scalar tensor for a known host integer.

    :param int | None value: Host integer value.
    :return torch.Tensor | None: CPU scalar tensor or ``None``.
    """

    return torch.tensor(int(value), dtype=torch.int32) if value is not None else None


def _reconcile_host_scalar_int(
    *,
    label: str,
    host_value: int | None,
    scalar_tensor: torch.Tensor | None,
    derived_value: int | None,
    required: bool = False,
) -> tuple[int | None, torch.Tensor | None]:
    """Reconcile duplicate integer forms without touching device tensors.

    :param str label: Field label used in errors.
    :param int | None host_value: Optional Python integer.
    :param torch.Tensor | None scalar_tensor: Optional CPU scalar tensor.
    :param int | None derived_value: Optional authoritative CPU-derived value.
    :param bool required: Whether absence is an error.
    :raises ValueError: If available values disagree.
    :raises RuntimeError: If a required value is unavailable.
    :return tuple[int | None, torch.Tensor | None]: Reconciled host and CPU scalar forms.
    """

    scalar_value = int(scalar_tensor) if scalar_tensor is not None else None
    values = [value for value in (derived_value, host_value, scalar_value) if value is not None]
    if len(set(values)) > 1:
        raise ValueError(
            f"{label} values disagree: derived={derived_value}, host={host_value}, scalar={scalar_value}."
        )
    resolved = values[0] if values else None
    if required and resolved is None:
        raise RuntimeError(
            f"{label} is missing for a device batch; build and validate it before device transfer."
        )
    return resolved, scalar_tensor if scalar_tensor is not None else _cpu_int_scalar(resolved)


def _flash_meta_with_route(
    flash_meta: FlashBatchMeta | None, route_hint: str | None
) -> FlashBatchMeta | None:
    """Return metadata with a normalized route hint override.

    :param FlashBatchMeta | None flash_meta: Existing metadata bundle.
    :param str | None route_hint: Route hint to install.
    :return FlashBatchMeta | None: Metadata bundle with the requested route.
    """

    route = str(route_hint).strip().lower() if route_hint is not None else None
    if route == "":
        route = None
    if flash_meta is None:
        return FlashBatchMeta(route_hint=route) if route is not None else None
    if flash_meta.normalized_route_hint() == route:
        return flash_meta
    return dataclasses.replace(flash_meta, route_hint=route)


def _resolve_flash_seq_lengths_and_active_tokens(
    batch: dict[str, Any],
    seq_lengths: torch.Tensor,
) -> tuple[torch.Tensor, int | None, torch.Tensor | None]:
    """Reconcile active-token counters with authoritative sequence lengths.

    CPU lengths are the validation boundary and must agree with both optional
    counter forms. Device lengths arrive only after a collator-issued CPU
    attestation, so their already-validated host/scalar counters are reused
    without synchronizing the device.

    :param dict[str, Any] batch: Batch mapping.
    :param torch.Tensor seq_lengths: Authoritative per-example lengths.
    :raises ValueError: If host/scalar counters disagree with each other or CPU lengths.
    :return tuple[torch.Tensor, int | None, torch.Tensor | None]: Sequence
        lengths, host active-token count, and CPU scalar active-token tensor.
    """

    active_tokens, active_tokens_scalar = _reconcile_host_scalar_int(
        label="Flash active-token count",
        host_value=_flash_active_tokens_host(batch.get(FLASH_SCALAR_BATCH_KEYS.active_tokens.host)),
        scalar_tensor=_flash_scalar_tensor(batch.get(CPU_SCALAR_BATCH_KEYS.active_tokens)),
        derived_value=(int(seq_lengths.sum(dtype=torch.int32)) if seq_lengths.device.type == "cpu" else None),
    )
    return seq_lengths, active_tokens, active_tokens_scalar


def _flash_doc_segment_host_stats(batch: dict[str, Any]) -> tuple[int | None, int | None]:
    """Return precomputed host doc-segment stats from the batch.

    :param dict[str, Any] batch: Batch mapping.
    :return tuple[int | None, int | None]: Segment count and max segment length.
    """

    num_segments = batch.get(FLASH_SCALAR_BATCH_KEYS.doc_num_segments.host)
    max_seqlen = batch.get(FLASH_SCALAR_BATCH_KEYS.doc_max_seqlen.host)
    if num_segments is None and max_seqlen is None:
        return None, None
    if (
        isinstance(num_segments, int)
        and not isinstance(num_segments, bool)
        and isinstance(max_seqlen, int)
        and not isinstance(max_seqlen, bool)
    ):
        return int(num_segments), int(max_seqlen)
    raise ValueError("flash_doc_num_segments and flash_doc_max_seqlen must be supplied together as integers.")


def _resolve_flash_doc_segment_stats(
    batch: dict[str, Any],
    *,
    segment_lengths: torch.Tensor,
    active_tokens: int | None,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Reconcile document segment host/scalar statistics.

    :param dict[str, Any] batch: Batch mapping.
    :param torch.Tensor segment_lengths: Validated segment lengths.
    :param int | None active_tokens: Validated active-token count.
    :raises ValueError: If duplicate host/scalar or CPU-derived values disagree.
    :raises RuntimeError: If device metadata lacks prevalidated host statistics.
    :return tuple[int, int, torch.Tensor, torch.Tensor]: Segment count, maximum
        segment length, and their CPU scalar tensors.
    """

    host_num, host_max = _flash_doc_segment_host_stats(batch)
    derived_num, derived_max, _ = doc_segment_metadata_host_stats(
        segment_lengths,
        active_tokens=active_tokens,
    )
    num_segments, num_scalar = _reconcile_host_scalar_int(
        label="Flash document segment count",
        host_value=host_num,
        scalar_tensor=_flash_scalar_tensor(batch.get(CPU_SCALAR_BATCH_KEYS.doc_num_segments)),
        derived_value=derived_num,
        required=True,
    )
    max_seqlen, max_scalar = _reconcile_host_scalar_int(
        label="Flash document max segment length",
        host_value=host_max,
        scalar_tensor=_flash_scalar_tensor(batch.get(CPU_SCALAR_BATCH_KEYS.doc_max_seqlen)),
        derived_value=derived_max,
        required=True,
    )
    assert num_segments is not None and max_seqlen is not None
    assert num_scalar is not None and max_scalar is not None
    return int(num_segments), int(max_seqlen), num_scalar, max_scalar


def _clear_flash_batch_metadata(batch: dict[str, Any]) -> None:
    """Remove every collator-attached flash metadata key from a batch.

    Early-return paths that hand the batch to non-flash consumers must not
    leave stale flash descriptors behind.

    :param dict[str, Any] batch: Batch mapping.
    """

    batch.pop("flash_seq_lengths", None)
    batch.pop("flash_mask_contract", None)
    batch.pop("flash_mask_contract_validated", None)
    for key in (
        FLASH_SCALAR_BATCH_KEYS.active_tokens.host,
        CPU_SCALAR_BATCH_KEYS.active_tokens,
        "flash_doc_segment_offsets",
        "flash_doc_segment_lengths",
        "flash_doc_cu_seqlens",
        FLASH_SCALAR_BATCH_KEYS.doc_num_segments.host,
        FLASH_SCALAR_BATCH_KEYS.doc_max_seqlen.host,
        CPU_SCALAR_BATCH_KEYS.doc_num_segments,
        CPU_SCALAR_BATCH_KEYS.doc_max_seqlen,
    ):
        batch.pop(key, None)


def _take_flash_mask_attestation(batch: dict[str, Any]) -> str | None:
    """Consume a collator-issued mask contract attestation.

    :param dict[str, Any] batch: Batch mapping.
    :raises ValueError: If an asserted contract is unknown.
    :return str | None: Validated contract name, or ``None`` when unattested.
    """

    contract = batch.pop("flash_mask_contract", None)
    validated = batch.pop("flash_mask_contract_validated", False)
    if validated is not True:
        return None
    normalized = str(contract).strip().lower()
    if normalized not in {"prefix", "docblock"}:
        raise ValueError(f"Unknown validated flash mask contract: {contract!r}.")
    return normalized


def prepare_flash_attention_batch_metadata(
    *,
    batch: dict[str, Any],
    backbone_type: str,
    flash_enabled: bool = False,
    flash_cfg: Any | None = None,
    route_device: torch.device | None = None,
) -> tuple[dict[str, Any], FlashBatchMeta | None]:
    """Attach precomputed flash metadata and return an out-of-graph metadata bundle.

    This owns ``doc_ids`` consumption for every backbone: non-``hf_deberta_v2``
    backbones get the dense pairwise doc-block mask built here, so callers must
    not pre-convert (a forgotten preamble would otherwise silently drop
    cross-document blocking).

    :param dict[str, Any] batch: Device-local batch mapping.
    :param str backbone_type: Backbone type string.
    :param bool flash_enabled: Whether the active backend can consume flash metadata.
    :param Any | None flash_cfg: Optional resolved flash config for route selection.
    :param torch.device | None route_device: Optional eventual activation device
        when preparing metadata before transfer.
    :return tuple[dict[str, Any], FlashBatchMeta | None]: Updated batch and optional metadata.
    """

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 2:
        _clear_flash_batch_metadata(batch)
        return batch, None

    btype = str(backbone_type).strip().lower()
    seq_len = int(input_ids.shape[-1])
    batch_size = int(input_ids.shape[0])
    validated_contract = _take_flash_mask_attestation(batch)
    routing_device = route_device if route_device is not None else input_ids.device
    flash_enabled = bool(flash_enabled)
    if flash_enabled and btype == "hf_deberta_v2":
        _configure_flash_kernel_overrides_from_cfg(flash_cfg)
        _notice_untuned_flash_hardware_once(routing_device)

    doc_ids = batch.pop("doc_ids", None)
    if isinstance(doc_ids, torch.Tensor) and doc_ids.ndim == 2:
        if doc_ids.shape != input_ids.shape:
            raise ValueError(
                "doc_ids must match input_ids shape; "
                f"got doc_ids={tuple(doc_ids.shape)}, input_ids={tuple(input_ids.shape)}."
            )
        source_attention_mask = batch.get("attention_mask")
        if validated_contract is not None:
            if validated_contract != "docblock":
                raise ValueError(f"Packed document batch carried mask_contract={validated_contract!r}.")
            keep_mask = doc_ids.ne(0)
        elif (btype != "hf_deberta_v2" or not flash_enabled) and doc_ids.device.type != "cpu":
            # General batch transfer happens before attention preparation. For eager
            # attention, doc_ids is itself the complete pairwise-mask source, so keep
            # mask construction on-device rather than forcing a quadratic CPU copy.
            keep_mask = doc_ids.ne(0)
        else:
            if doc_ids.device.type != "cpu":
                raise RuntimeError(
                    "Unvalidated document metadata reached a device batch. Validate it on CPU "
                    "before device transfer."
                )
            if source_attention_mask is None:
                keep_mask = torch.ones_like(doc_ids, dtype=torch.bool)
            elif isinstance(source_attention_mask, torch.Tensor):
                keep_mask = mask_to_2d_keep_mask(source_attention_mask, seq_len=seq_len)
            else:
                raise TypeError("Packed attention_mask must be a tensor or None.")
            if not torch.equal(doc_ids.ne(0), keep_mask):
                raise ValueError("doc_ids liveness disagrees with attention_mask.")

        if btype != "hf_deberta_v2" or not flash_enabled:
            batch["attention_mask"] = build_doc_block_mask(doc_ids)
            _clear_flash_batch_metadata(batch)
            return batch, None
        route_hint = _flash_route_hint_for_docblock_batch(
            seq_len=seq_len,
            batch_size=batch_size,
            flash_cfg=flash_cfg,
            device=routing_device,
        )
        _notice_docblock_route_once(
            route_hint=route_hint,
            seq_len=seq_len,
            batch_size=batch_size,
            device=routing_device,
            flash_cfg=flash_cfg,
        )

        segment_offsets = batch.get("flash_doc_segment_offsets")
        segment_lengths = batch.get("flash_doc_segment_lengths")
        cu_seqlens = batch.get("flash_doc_cu_seqlens")
        supplied_segment_fields = (
            segment_offsets is not None,
            segment_lengths is not None,
            cu_seqlens is not None,
        )
        if any(supplied_segment_fields) and not all(supplied_segment_fields):
            raise ValueError(
                "flash_doc_segment_offsets, flash_doc_segment_lengths, and "
                "flash_doc_cu_seqlens must be supplied together."
            )
        if not (
            isinstance(segment_offsets, torch.Tensor)
            and isinstance(segment_lengths, torch.Tensor)
            and isinstance(cu_seqlens, torch.Tensor)
        ):
            if doc_ids.device.type != "cpu":
                raise RuntimeError(
                    "Flash doc-block metadata is missing for a device batch. "
                    "Build flash_doc_segment_* metadata in the collator before device transfer."
                )
            segment_offsets, segment_lengths, cu_seqlens, _ = build_doc_segment_metadata(doc_ids)
        if (
            segment_offsets.ndim != 1
            or segment_lengths.ndim != 1
            or cu_seqlens.ndim != 1
            or segment_offsets.shape != segment_lengths.shape
            or int(cu_seqlens.numel()) != int(segment_lengths.numel()) + 1
        ):
            raise ValueError(
                "Flash document descriptor tensor shapes disagree: "
                f"offsets={tuple(segment_offsets.shape)}, "
                f"lengths={tuple(segment_lengths.shape)}, cu={tuple(cu_seqlens.shape)}."
            )
        if validated_contract is None:
            validate_doc_segments_against_mask(
                attention_mask=keep_mask,
                segment_offsets=segment_offsets,
                segment_lengths=segment_lengths,
                seq_len=seq_len,
                doc_ids=doc_ids,
                cu_seqlens=cu_seqlens,
            )
        derived_active_tokens = (
            int(keep_mask.sum(dtype=torch.int32)) if keep_mask.device.type == "cpu" else None
        )
        active_tokens, active_tokens_scalar = _reconcile_host_scalar_int(
            label="Flash active-token count",
            host_value=_flash_active_tokens_host(batch.get(FLASH_SCALAR_BATCH_KEYS.active_tokens.host)),
            scalar_tensor=_flash_scalar_tensor(batch.get(CPU_SCALAR_BATCH_KEYS.active_tokens)),
            derived_value=derived_active_tokens,
            required=True,
        )
        assert active_tokens is not None
        (
            doc_num_segments,
            doc_max_seqlen,
            doc_num_segments_scalar,
            doc_max_seqlen_scalar,
        ) = _resolve_flash_doc_segment_stats(
            batch,
            segment_lengths=segment_lengths,
            active_tokens=active_tokens,
        )

        if route_hint == "docblock_bias":
            batch["attention_mask"] = build_doc_block_mask(doc_ids)
            meta = FlashBatchMeta(
                active_tokens_host=active_tokens,
                active_tokens_scalar=active_tokens_scalar,
                route_hint=route_hint,
                mask_contract="docblock",
                mask_contract_validated=True,
            )
        else:
            batch["attention_mask"] = keep_mask
            meta = FlashBatchMeta(
                doc_segment_offsets=segment_offsets,
                doc_segment_lengths=segment_lengths,
                doc_cu_seqlens=cu_seqlens,
                active_tokens_host=active_tokens,
                doc_num_segments_host=doc_num_segments,
                doc_max_segment_length_host=doc_max_seqlen,
                active_tokens_scalar=active_tokens_scalar,
                doc_num_segments_scalar=doc_num_segments_scalar,
                doc_max_segment_length_scalar=doc_max_seqlen_scalar,
                route_hint=route_hint,
                mask_contract="docblock",
                mask_contract_validated=True,
            )
        _clear_flash_batch_metadata(batch)
        return batch, meta

    if doc_ids is not None:
        raise ValueError(f"doc_ids must be rank-2; got shape={getattr(doc_ids, 'shape', None)}.")

    if btype != "hf_deberta_v2":
        _clear_flash_batch_metadata(batch)
        return batch, None

    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        _clear_flash_batch_metadata(batch)
        return (
            batch,
            FlashBatchMeta(
                route_hint="dense",
                mask_contract="all_active",
                mask_contract_validated=True,
            )
            if flash_enabled
            else None,
        )
    if is_pairwise_mask(attention_mask, query_len=seq_len, key_len=seq_len):
        _clear_flash_batch_metadata(batch)
        return batch, None
    if not isinstance(attention_mask, torch.Tensor):
        _clear_flash_batch_metadata(batch)
        return batch, None
    if not flash_enabled:
        _clear_flash_batch_metadata(batch)
        return batch, None

    supplied_seq_lengths = batch.get("flash_seq_lengths")
    if supplied_seq_lengths is not None and (
        not isinstance(supplied_seq_lengths, torch.Tensor) or supplied_seq_lengths.ndim != 1
    ):
        raise ValueError("flash_seq_lengths shape disagrees with its required rank-1 contract.")
    if validated_contract is not None:
        if validated_contract != "prefix":
            raise ValueError(f"Padding batch carried mask_contract={validated_contract!r}.")
        if supplied_seq_lengths is None:
            raise ValueError("Validated prefix batch is missing flash_seq_lengths.")
        seq_lengths = supplied_seq_lengths
    else:
        if attention_mask.device.type != "cpu":
            # Ordinary device-side model calls have no trusted pre-transfer
            # metadata. Keep the original mask authoritative and use eager
            # attention rather than synchronizing it into Python.
            _clear_flash_batch_metadata(batch)
            return batch, None
        try:
            seq_lengths = build_validated_prefix_lengths(
                attention_mask,
                seq_len=seq_len,
                supplied_lengths=supplied_seq_lengths,
            )
        except ValueError as exc:
            if "right-padded prefix mask" not in str(exc):
                raise
            _notice_non_prefix_padding_fallback_once()
            _clear_flash_batch_metadata(batch)
            return batch, None

    if tuple(seq_lengths.shape) != (batch_size,):
        raise ValueError(
            "Flash sequence-length batch shape disagrees with input_ids: "
            f"lengths={tuple(seq_lengths.shape)}, batch_size={batch_size}."
        )
    seq_lengths, active_tokens, active_tokens_scalar = _resolve_flash_seq_lengths_and_active_tokens(
        batch, seq_lengths
    )
    route_active_tokens = int(active_tokens) if active_tokens is not None else int(seq_len) * batch_size
    route_hint = _flash_route_hint_for_padding_batch(
        seq_len=seq_len,
        active_tokens=route_active_tokens,
        batch_size=batch_size,
        flash_cfg=flash_cfg,
        device=routing_device,
    )
    meta = FlashBatchMeta(
        seq_lengths=seq_lengths,
        active_tokens_host=active_tokens,
        active_tokens_scalar=active_tokens_scalar,
        route_hint=route_hint,
        mask_contract="prefix",
        mask_contract_validated=True,
    )
    _clear_flash_batch_metadata(batch)
    return batch, meta


def _maybe_cudagraph_mark_step_begin() -> None:
    """Mark cudagraph step boundaries when the API is available.

    This is a no-op on PyTorch builds that do not expose ``torch.compiler``
    or ``cudagraph_mark_step_begin``.
    """
    try:
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
    except Exception:
        # Best-effort: keep training running if backend API shape changes.
        pass


def _resolve_compile_scope(
    *,
    requested_scope: str,
    model_cfg: ModelConfig,
    block_cross_document_attention: bool = False,
) -> tuple[str, str | None]:
    """Resolve effective compile scope for auto compile mode.

    :param str requested_scope: Requested canonical compile scope.
    :param ModelConfig model_cfg: Model configuration.
    :param bool block_cross_document_attention: Whether doc-blocking is enabled.
    :return tuple[str, str | None]: Effective scope and optional reason message.
    """
    if requested_scope != "auto":
        return requested_scope, None

    backbone_type = str(getattr(model_cfg, "backbone_type", "")).strip().lower()
    if bool(block_cross_document_attention) and backbone_type == "rope":
        return (
            "ffn",
            "auto scope selected FFN-only for rope + doc-blocking (mask shape churn under compile)",
        )
    return "backbones", None


def _resolve_effective_compile_scope(
    *,
    train_cfg: Any,
    model_cfg: ModelConfig,
    data_cfg: Any,
    compile_enabled: bool,
) -> tuple[str, str, str | None]:
    """Resolve requested and effective compile scope for one training run.

    :param Any train_cfg: Train config carrying ``torch_compile_scope``.
    :param ModelConfig model_cfg: Model configuration.
    :param Any data_cfg: Data config carrying ``block_cross_document_attention``.
    :param bool compile_enabled: Whether torch.compile is enabled.
    :return tuple[str, str, str | None]: Requested scope, effective scope, and
        optional downgrade reason (effective == requested when compile is off).
    """
    requested_scope = str(train_cfg.compile.scope).strip().lower()
    if not compile_enabled:
        return requested_scope, requested_scope, None
    scope, reason = _resolve_compile_scope(
        requested_scope=requested_scope,
        model_cfg=model_cfg,
        block_cross_document_attention=bool(data_cfg.packing.block_cross_document_attention),
    )
    return requested_scope, scope, reason


def _compile_backbones_for_scope(
    *,
    unwrapped_model: torch.nn.Module,
    compile_scope: str,
    compile_kwargs: dict[str, Any],
) -> list[str]:
    """Compile selected generator/discriminator submodules for a requested scope.

    :param torch.nn.Module unwrapped_model: Unwrapped RTD pretrainer model.
    :param str compile_scope: Effective compile scope.
    :param dict[str, Any] compile_kwargs: Keyword arguments passed to ``torch.compile``.
    :raises RuntimeError: If required backbone submodules are missing.
    :return list[str]: Human-readable list of compiled targets.
    """
    generator = getattr(unwrapped_model, "generator", None)
    discriminator = getattr(unwrapped_model, "discriminator", None)
    if not isinstance(generator, torch.nn.Module) or not isinstance(discriminator, torch.nn.Module):
        raise RuntimeError("RTD model must expose generator and discriminator modules for compilation.")

    compiled_targets: list[str] = []

    def _compile_module_forward(*, module: torch.nn.Module, target: str) -> None:
        """Compile one module's ``forward`` method in-place.

        Keeping module identity stable avoids ``._orig_mod`` wrapper keys in saved checkpoints.

        :param torch.nn.Module module: Module whose forward should be compiled.
        :param str target: Human-readable target name for logs.
        """
        if _install_stable_backbone_compile_dispatch(
            module=module,
            compile_kwargs=compile_kwargs,
            target=target,
            compiled_targets=compiled_targets,
        ):
            return
        forward = getattr(module, "forward", None)
        if not callable(forward):
            raise RuntimeError(f"{target}.forward is required for compile scope.")
        module.forward = torch.compile(forward, **compile_kwargs)  # type: ignore[assignment]
        compiled_targets.append(target)

    def _compile_encoder_ffn(*, backbone: torch.nn.Module, branch: str) -> None:
        """Compile FFN blocks in all encoder layers.

        Supports both HF DeBERTa-v2 (``encoder.layer[i].intermediate``/``.output``)
        and RoPE (``encoder.layers[i].mlp``) module layouts.

        :param torch.nn.Module backbone: Generator/discriminator backbone.
        :param str branch: Branch label used in compiled target names.
        :raises RuntimeError: If encoder/layer/FFN modules are missing.
        """
        encoder = getattr(backbone, "encoder", None)
        if not isinstance(encoder, torch.nn.Module):
            raise RuntimeError(f"{branch}.encoder is required for ffn-only compile scope.")

        # Try HF DeBERTa-v2 convention (encoder.layer) then RoPE (encoder.layers).
        layers = getattr(encoder, "layer", None)
        layers_attr = "layer"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            layers = getattr(encoder, "layers", None)
            layers_attr = "layers"
        if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
            raise RuntimeError(f"{branch}.encoder.layer/layers is required for ffn-only compile scope.")
        if len(layers) == 0:
            raise RuntimeError(
                f"{branch}.encoder.{layers_attr} is empty; cannot apply ffn-only compile scope."
            )

        for idx, layer in enumerate(layers):
            if not isinstance(layer, torch.nn.Module):
                raise RuntimeError(f"{branch}.encoder.{layers_attr}[{idx}] is not a torch.nn.Module.")

            # HF DeBERTa-v2: intermediate + output modules.
            intermediate = getattr(layer, "intermediate", None)
            output = getattr(layer, "output", None)
            if isinstance(intermediate, torch.nn.Module) and isinstance(output, torch.nn.Module):
                pfx = f"{branch}.encoder.{layers_attr}[{idx}]"
                _compile_module_forward(module=intermediate, target=f"{pfx}.intermediate")
                _compile_module_forward(module=output, target=f"{pfx}.output")
                continue

            # RoPE: single mlp module.
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, torch.nn.Module):
                _compile_module_forward(module=mlp, target=f"{branch}.encoder.{layers_attr}[{idx}].mlp")
                continue

            raise RuntimeError(
                f"{branch}.encoder.{layers_attr}[{idx}] has no recognized FFN module "
                "(expected intermediate+output or mlp)."
            )

    if compile_scope == "backbones":
        _compile_module_forward(module=generator, target="generator")
        _compile_module_forward(module=discriminator, target="discriminator")
        return compiled_targets

    if compile_scope in {"encoder", "gen_encoder"}:
        gen_encoder = getattr(generator, "encoder", None)
        if not isinstance(gen_encoder, torch.nn.Module):
            raise RuntimeError("generator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=gen_encoder, target="generator.encoder")

    if compile_scope in {"encoder", "disc_encoder"}:
        disc_encoder = getattr(discriminator, "encoder", None)
        if not isinstance(disc_encoder, torch.nn.Module):
            raise RuntimeError("discriminator.encoder is required for encoder-only compile scope.")
        _compile_module_forward(module=disc_encoder, target="discriminator.encoder")

    if compile_scope in {"ffn", "gen_ffn"}:
        _compile_encoder_ffn(backbone=generator, branch="generator")

    if compile_scope in {"ffn", "disc_ffn"}:
        _compile_encoder_ffn(backbone=discriminator, branch="discriminator")

    if not compiled_targets:
        raise ValueError(f"Unsupported compile scope: {compile_scope}")
    return compiled_targets


def _install_stable_backbone_compile_dispatch(
    *,
    module: torch.nn.Module,
    compile_kwargs: dict[str, Any],
    target: str,
    compiled_targets: list[str],
) -> bool:
    """Install a stable compiled dense/masked dispatcher when supported.

    Native HF DeBERTa backbones accept Python optionals in ``forward`` for
    ``attention_mask`` and output flags. Compiling that public ``forward``
    directly encourages Dynamo to guard on ``None``/bool optionals. When the
    backbone exposes resolved dense/masked helpers, compile those stable
    entrypoints instead and leave ``forward`` as a tiny Python dispatcher.

    :param torch.nn.Module module: Candidate backbone module.
    :param dict[str, Any] compile_kwargs: Keyword arguments passed to ``torch.compile``.
    :param str target: Human-readable target name for logs.
    :param list[str] compiled_targets: Accumulator for compiled target labels.
    :return bool: True when a stable dispatcher was installed.
    """

    resolve_options = getattr(module, "_resolve_forward_options", None)
    dense_hs0 = getattr(module, "_forward_dense_hs0", None)
    dense_hs1 = getattr(module, "_forward_dense_hs1", None)
    masked_hs0 = getattr(module, "_forward_masked_hs0", None)
    masked_hs1 = getattr(module, "_forward_masked_hs1", None)
    if not (
        callable(resolve_options)
        and callable(dense_hs0)
        and callable(dense_hs1)
        and callable(masked_hs0)
        and callable(masked_hs1)
    ):
        return False
    dense_hs0_fn = dense_hs0
    dense_hs1_fn = dense_hs1
    masked_hs0_fn = masked_hs0
    masked_hs1_fn = masked_hs1

    def _make_routed_masked_fn(base_fn: Callable[..., Any], route: str) -> Callable[..., Any]:
        """Bind a fixed flash route onto one stable masked entrypoint.

        Each returned closure is a distinct function object, so ``torch.compile``
        keeps one compiled artifact per (route, hidden-states) combination.

        :param Callable[..., Any] base_fn: Stable masked helper to wrap.
        :param str route: Flash route literal stamped onto ``flash_meta``.
        :return Callable[..., Any]: Route-bound masked entrypoint.
        """

        def _routed_masked_fn(
            *,
            input_ids: torch.Tensor | None = None,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
            flash_meta: FlashBatchMeta | None = None,
        ) -> Any:
            """Call the masked helper with a fixed flash route.

            :param torch.Tensor | None input_ids: Optional input token ids.
            :param torch.Tensor attention_mask: Attention mask tensor.
            :param torch.Tensor | None token_type_ids: Optional token type ids.
            :param torch.Tensor | None position_ids: Optional position ids.
            :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
            :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
            :return Any: Masked-path backbone outputs.
            """

            return base_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_meta=_flash_meta_with_route(flash_meta, route),
            )

        return _routed_masked_fn

    def _compile_routed_masked_pair(route: str) -> dict[bool, Any]:
        """Compile the hs0/hs1 masked entrypoints bound to one flash route.

        :param str route: Flash route literal.
        :return dict[bool, Any]: Compiled entrypoints keyed by ``output_hidden_states``.
        """

        return {
            False: torch.compile(_make_routed_masked_fn(masked_hs0_fn, route), **compile_kwargs),
            True: torch.compile(_make_routed_masked_fn(masked_hs1_fn, route), **compile_kwargs),
        }

    compiled_dense = {
        False: torch.compile(dense_hs0_fn, **compile_kwargs),
        True: torch.compile(dense_hs1_fn, **compile_kwargs),
    }
    compiled_masked = {
        False: torch.compile(masked_hs0_fn, **compile_kwargs),
        True: torch.compile(masked_hs1_fn, **compile_kwargs),
    }
    # Routes with a dedicated compiled specialization; adding a route family
    # means adding one entry here. Hints outside this mapping (or None) run the
    # generic masked entrypoint with the hint re-attached, so the model-side
    # adapter still resolves them.
    compiled_masked_routed = {
        route: _compile_routed_masked_pair(route)
        for route in ("fixed", "varlen", "docblock", "docblock_bias")
    }

    def _dispatch_forward(
        self: torch.nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> Any:
        """Normalize public options, then dispatch into stable compiled entrypoints.

        :param torch.nn.Module self: Backbone module instance.
        :param torch.Tensor | None input_ids: Optional input token ids.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor | None token_type_ids: Optional token type ids.
        :param torch.Tensor | None position_ids: Optional position ids.
        :param torch.Tensor | None inputs_embeds: Optional precomputed embeddings.
        :param bool | None output_attentions: Optional attention-output flag.
        :param bool | None output_hidden_states: Optional hidden-state-output flag.
        :param bool | None return_dict: Optional return-format flag.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return Any: Module outputs from either a compiled fast path or the generic resolved path.
        """

        (
            resolved_output_attentions,
            resolved_output_hidden_states,
            resolved_return_dict,
        ) = self._resolve_forward_options(  # type: ignore[attr-defined]
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # The fast compiled path specializes on the fixed training contract:
        # return_dict=True, output_attentions=False, output_hidden_states in {False, True}.
        # Other combinations are correct but uncommon in training, so keep them
        # on the uncompiled resolved path instead of exploding compile variants.
        if resolved_output_attentions or not resolved_return_dict:
            return self._forward_resolved(  # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                output_attentions=resolved_output_attentions,
                output_hidden_states=resolved_output_hidden_states,
                return_dict=resolved_return_dict,
                flash_meta=flash_meta,
            )
        if attention_mask is None:
            return compiled_dense[resolved_output_hidden_states](
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
            )
        normalized_route = flash_meta.normalized_route_hint() if flash_meta is not None else None
        routed = compiled_masked_routed.get(normalized_route) if normalized_route is not None else None
        if routed is not None:
            assert flash_meta is not None, "A compiled routed target requires explicit FlashBatchMeta."
            return routed[resolved_output_hidden_states](
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                flash_meta=flash_meta,
            )
        return compiled_masked[resolved_output_hidden_states](
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            flash_meta=_flash_meta_with_route(flash_meta, normalized_route),
        )

    module._compiled_forward_dense = compiled_dense
    module._compiled_forward_masked = compiled_masked
    module._compiled_forward_masked_routed = compiled_masked_routed
    module.forward = types.MethodType(_dispatch_forward, module)  # type: ignore[assignment]
    compiled_targets.extend(
        [
            f"{target}[dense_hs0]",
            f"{target}[dense_hs1]",
            f"{target}[masked_hs0]",
            f"{target}[masked_hs1]",
            *(
                f"{target}[masked_{route}_hs{int(hs)}]"
                for route in compiled_masked_routed
                for hs in (False, True)
            ),
        ]
    )
    return True


def _dtype_for_mixed_precision(mode: str) -> torch.dtype:
    """Map configured mixed-precision mode to runtime compute dtype.

    :param str mode: Effective mixed-precision mode.
    :return torch.dtype: Expected activation dtype used in forward passes.
    """
    normalized = str(mode).strip().lower()
    if normalized == "bf16":
        return torch.bfloat16
    return torch.float32


def _prefill_rotary_caches_for_compile(
    *,
    model: torch.nn.Module,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """Prefill rotary caches on all RoPE attention modules before compile.

    :param torch.nn.Module model: Unwrapped RTD model.
    :param int seq_len: Maximum runtime sequence length.
    :param torch.device device: Runtime device.
    :param torch.dtype dtype: Runtime compute dtype.
    :return int: Number of rotary modules prefilled.
    """
    if int(seq_len) <= 0:
        return 0

    prefilled = 0
    for module in model.modules():
        rope = getattr(module, "rope", None)
        prefill = getattr(rope, "prefill_cache", None)
        if callable(prefill):
            prefill(int(seq_len), device=device, dtype=dtype)
            prefilled += 1
    return prefilled


def _stabilize_compile_attention_mask(
    *,
    batch: dict[str, torch.Tensor],
    compile_enabled: bool,
    compile_scope: str,
    backbone_type: str,
) -> dict[str, torch.Tensor]:
    """Canonicalize attention-mask dtype for compiled attention paths.

    For HF DeBERTa-v2: normalizes existing masks to bool but does **not**
    materialize a mask when absent — the backbone's no-mask fast path handles
    ``None`` directly.

    RoPE + doc-blocking mask shape churn is handled by auto-downgrading compile
    scope to FFN in ``_resolve_compile_scope`` instead of materializing S² masks.

    :param dict[str, torch.Tensor] batch: Device-local batch mapping.
    :param bool compile_enabled: Whether torch.compile is active.
    :param str compile_scope: Effective compile scope.
    :param str backbone_type: Model backbone type.
    :return dict[str, torch.Tensor]: Possibly updated batch mapping.
    """
    if not bool(compile_enabled):
        return batch

    scope = str(compile_scope).strip().lower()
    if scope not in {"backbones", "encoder", "gen_encoder", "disc_encoder"}:
        return batch

    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        return batch

    btype = str(backbone_type).strip().lower()

    if btype == "hf_deberta_v2":
        attn = batch.get("attention_mask")
        if isinstance(attn, torch.Tensor) and attn.dtype != torch.bool:
            batch["attention_mask"] = attn.to(dtype=torch.bool)
        return batch

    return batch
