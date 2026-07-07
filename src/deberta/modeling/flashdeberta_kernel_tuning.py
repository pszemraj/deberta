"""FlashDeBERTa kernel tuning policy loaded from repo-local JSON data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_TUNING_PATH = Path(__file__).with_name("flashdeberta_kernel_tuning.json")
_ACTIVE_OVERRIDES_PATH: str | None = None


@dataclass(frozen=True)
class FlashKernelContext:
    """Runtime facts used to match a FlashDeBERTa kernel tuning entry."""

    compute_capability: tuple[int, int]
    route: str
    kind: str
    seq_len: int
    head_dim: int
    total_tokens: int | None = None
    batch_size: int | None = None
    query_len: int | None = None
    key_len: int | None = None
    num_heads: int | None = None
    dtype: str | None = None
    causal: bool | None = None
    disentangled: bool | None = None
    att_span: int | None = None
    has_mask: bool | None = None


def configure_flashdeberta_kernel_overrides(path: str | None) -> None:
    """Set the process-local FlashDeBERTa kernel override table.

    Reapplying the already-active path is a no-op, so hot-path callers such as
    per-step batch preparation do not invalidate the cached tuning table.

    :param str | None path: JSON table path, or None to use only the package default.
    """

    global _ACTIVE_OVERRIDES_PATH
    normalized = str(path).strip() if path is not None else ""
    resolved = normalized or None
    if resolved == _ACTIVE_OVERRIDES_PATH:
        return
    _ACTIVE_OVERRIDES_PATH = resolved
    _load_tuning_payload.cache_clear()


def active_flashdeberta_kernel_overrides_path() -> str | None:
    """Return the active process-local kernel override table path.

    :return str | None: Override path, or None when only package defaults are active.
    """

    return _ACTIVE_OVERRIDES_PATH


def compute_capability_key(capability: tuple[int, int]) -> str:
    """Return the table key for a CUDA compute capability tuple.

    :param tuple[int, int] capability: ``(major, minor)`` capability.
    :return str: Key such as ``"sm_120"``.
    """

    major, minor = capability
    return f"sm_{int(major)}{int(minor)}"


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping from disk.

    :param Path path: File path.
    :raises ValueError: If the file does not contain a JSON object.
    :return dict[str, Any]: Parsed mapping.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"FlashDeBERTa tuning table must be a JSON object: {path}")
    return payload


def _load_tuning_payload() -> dict[str, Any]:
    """Load the default table plus optional user override entries.

    :return dict[str, Any]: Merged tuning payload.
    """

    payload = _read_json(_DEFAULT_TUNING_PATH)
    override_path = _ACTIVE_OVERRIDES_PATH
    if override_path is None:
        return payload

    overrides = _read_json(Path(override_path).expanduser())
    merged: dict[str, Any] = dict(payload)
    for key in ("seq_buckets", "route_policies", "kernels"):
        base_value = payload.get(key)
        override_value = overrides.get(key)
        if isinstance(base_value, list) or isinstance(override_value, list):
            merged[key] = [
                *(base_value if isinstance(base_value, list) else []),
                *(override_value if isinstance(override_value, list) else []),
            ]
        elif isinstance(base_value, dict) or isinstance(override_value, dict):
            combined = dict(base_value if isinstance(base_value, dict) else {})
            combined.update(override_value if isinstance(override_value, dict) else {})
            merged[key] = combined
    return merged


try:
    from functools import cache
except ImportError:  # pragma: no cover - Python 3.8 fallback kept harmless for tooling.
    from functools import lru_cache as cache


_load_tuning_payload = cache(_load_tuning_payload)


def flash_seq_bucket(*, seq_len: int, total_tokens: int | None = None, batch_size: int | None = None) -> str:
    """Resolve the measured-policy sequence bucket for one batch shape.

    :param int seq_len: Padded sequence length.
    :param int | None total_tokens: Active token count, when known.
    :param int | None batch_size: Batch size, when known.
    :return str: Bucket name. ``"default"`` means no measured bucket matched.
    """

    capacity = max(1, int(seq_len) * max(1, int(batch_size or 1)))
    density = None if total_tokens is None else float(total_tokens) / float(capacity)
    buckets = _load_tuning_payload().get("seq_buckets", [])
    for raw in buckets:
        if not isinstance(raw, dict):
            continue
        min_seq = raw.get("min_seq_len")
        max_seq = raw.get("max_seq_len")
        if min_seq is not None and int(seq_len) < int(min_seq):
            continue
        if max_seq is not None and int(seq_len) > int(max_seq):
            continue
        if density is not None:
            min_density = raw.get("min_density")
            max_density = raw.get("max_density")
            max_density_exclusive = raw.get("max_density_exclusive")
            if min_density is not None and density < float(min_density):
                continue
            if max_density is not None and density > float(max_density):
                continue
            if max_density_exclusive is not None and density >= float(max_density_exclusive):
                continue
        elif any(key in raw for key in ("min_density", "max_density", "max_density_exclusive")):
            continue
        name = raw.get("name")
        if isinstance(name, str) and name:
            return name
    return "default"


def flash_route_policy(*, policy: str, seq_bucket: str) -> dict[str, Any] | None:
    """Resolve a route-policy row from the active tuning table.

    :param str policy: Route policy namespace such as ``"padding"`` or ``"docblock"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :return dict[str, Any] | None: Matching policy row, or None when the table has no entry.
    """

    policies = _load_tuning_payload().get("route_policies", {})
    choices = policies.get(str(policy).strip().lower()) if isinstance(policies, dict) else None
    if not isinstance(choices, list):
        return None
    for raw in reversed(choices):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("seq_bucket", "")).strip() != str(seq_bucket).strip():
            continue
        return dict(raw)
    return None


def flash_route_choice(*, policy: str, seq_bucket: str) -> str | None:
    """Resolve a route choice from the active tuning table.

    :param str policy: Route policy namespace such as ``"padding"`` or ``"docblock"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :return str | None: Route choice, or None when the table has no entry.
    """

    raw = flash_route_policy(policy=policy, seq_bucket=seq_bucket)
    if raw is not None:
        choice = raw.get("choice")
        return str(choice).strip() if choice is not None else None
    return None


def _entry_matches(context: FlashKernelContext, entry: dict[str, Any]) -> bool:
    """Return whether a table entry applies to one kernel context.

    :param FlashKernelContext context: Runtime kernel context.
    :param dict[str, Any] entry: Candidate JSON entry.
    :return bool: True when the entry should be used.
    """

    cc_key = compute_capability_key(context.compute_capability)
    if str(entry.get("compute_capability", "*")).strip().lower() not in {cc_key, "*"}:
        return False
    if str(entry.get("route", "")).strip().lower() != str(context.route).strip().lower():
        return False
    if str(entry.get("kind", "")).strip().lower() != str(context.kind).strip().lower():
        return False
    if str(entry.get("seq_bucket", "")).strip() != flash_seq_bucket(
        seq_len=context.seq_len,
        total_tokens=context.total_tokens,
        batch_size=context.batch_size,
    ):
        return False
    head_dim = entry.get("head_dim", "*")
    if head_dim != "*" and int(head_dim) != int(context.head_dim):
        return False
    exact_int_fields = (
        ("batch_size", context.batch_size),
        ("query_len", context.query_len),
        ("key_len", context.key_len),
        ("num_heads", context.num_heads),
    )
    for key, value in exact_int_fields:
        expected = entry.get(key)
        if expected is not None and value is not None and int(expected) != int(value):
            return False
    range_int_fields = (
        ("batch_size", context.batch_size),
        ("query_len", context.query_len),
        ("key_len", context.key_len),
        ("num_heads", context.num_heads),
    )
    for key, value in range_int_fields:
        if value is None:
            continue
        min_expected = entry.get(f"min_{key}")
        max_expected = entry.get(f"max_{key}")
        if min_expected is not None and int(value) < int(min_expected):
            return False
        if max_expected is not None and int(value) > int(max_expected):
            return False
    exact_bool_fields = (
        ("causal", context.causal),
        ("disentangled", context.disentangled),
        ("has_mask", context.has_mask),
    )
    for key, value in exact_bool_fields:
        expected = entry.get(key)
        if expected is not None and value is not None and bool(expected) != bool(value):
            return False
    dtype = entry.get("dtype")
    if dtype is not None and context.dtype is not None and str(dtype).strip() != str(context.dtype).strip():
        return False
    att_span_min = entry.get("att_span_min")
    if (
        att_span_min is not None
        and context.att_span is not None
        and int(context.att_span) < int(att_span_min)
    ):
        return False
    return True


def resolve_flash_kernel_config(context: FlashKernelContext) -> tuple[int, int, int, int] | None:
    """Resolve a measured kernel launch tuple from the active tuning table.

    :param FlashKernelContext context: Runtime kernel context.
    :return tuple[int, int, int, int] | None: ``(BLOCK_M, BLOCK_N, stages, warps)`` or None.
    """

    kernels = _load_tuning_payload().get("kernels", [])
    if not isinstance(kernels, list):
        return None
    for raw in reversed(kernels):
        if not isinstance(raw, dict) or not _entry_matches(context, raw):
            continue
        try:
            return (
                int(raw["block_m"]),
                int(raw["block_n"]),
                int(raw["num_stages"]),
                int(raw["num_warps"]),
            )
        except Exception:
            return None
    return None
