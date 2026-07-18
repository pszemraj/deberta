"""FlashDeBERTa kernel tuning policy loaded from repo-local JSON data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_TUNING_PATH = Path(__file__).with_name("flashdeberta_kernel_tuning.json")
_ACTIVE_OVERRIDES_PATH: str | None = None

# Shape-keyed lookups take per-batch values (total_tokens, batch_size) in their
# cache keys, so those caches must be bounded LRUs: a long variably-packed run
# otherwise accretes one permanent entry per distinct batch shape.
_SHAPE_KEYED_CACHE_MAXSIZE = 4096

# Safe launch used when no capability/shape row has been measured. Keep this
# repo-owned and deterministic: upstream selectors also expose environment
# variable overrides, which are deliberately not part of this project's
# configuration contract.
CONSERVATIVE_FLASH_KERNEL_CONFIG = (16, 16, 1, 4)


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

    :param str | None path: JSON table path, or None to use only the package default.
    """

    global _ACTIVE_OVERRIDES_PATH
    normalized = str(path).strip() if path is not None else ""
    resolved = normalized or None
    if resolved == _ACTIVE_OVERRIDES_PATH:
        return
    _ACTIVE_OVERRIDES_PATH = resolved
    _load_tuning_payload.cache_clear()
    flash_seq_bucket.cache_clear()
    flash_route_policy.cache_clear()
    resolve_flash_kernel_config.cache_clear()


def compute_capability_key(capability: tuple[int, int]) -> str:
    """Return the table key for a CUDA compute capability tuple.

    :param tuple[int, int] capability: ``(major, minor)`` capability.
    :return str: Key such as ``"sm_120"``.
    """

    major, minor = capability
    return f"sm_{int(major)}{int(minor)}"


def _read_json(path: Path) -> Any:
    """Read JSON from disk.

    :param Path path: File path.
    :return Any: Parsed JSON value.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _load_tuning_payload() -> dict[str, Any]:
    """Load the default table plus optional user override entries.

    ``route_policies`` and ``kernels`` override rows append to the shipped
    rows rather than replacing them, so an override table can promote a
    single row (for example one capability-scoped route choice) without
    restating the shipped policy; appended rows outrank shipped rows of the
    same specificity. ``seq_buckets`` override rows prepend, and a row whose
    ``name`` matches a shipped bucket replaces that bucket entirely.

    :return dict[str, Any]: Merged tuning payload.
    """

    payload = _read_json(_DEFAULT_TUNING_PATH)
    override_path = _ACTIVE_OVERRIDES_PATH
    if override_path is None:
        return payload

    expanded_override_path = Path(override_path).expanduser()
    overrides = _read_json(expanded_override_path)
    merged: dict[str, Any] = dict(payload)

    override_buckets = overrides.get("seq_buckets", [])
    override_names = {row.get("name") for row in override_buckets if row.get("name")}
    base_buckets = [row for row in payload.get("seq_buckets", []) if row.get("name") not in override_names]
    merged["seq_buckets"] = [*override_buckets, *base_buckets]

    merged_policies = {name: list(rows) for name, rows in payload.get("route_policies", {}).items()}
    for name, rows in overrides.get("route_policies", {}).items():
        merged_policies[name] = [*merged_policies.get(name, []), *rows]
    merged["route_policies"] = merged_policies
    merged["kernels"] = [*payload.get("kernels", []), *overrides.get("kernels", [])]
    return merged


_load_tuning_payload = cache(_load_tuning_payload)


def flash_seq_bucket(*, seq_len: int, total_tokens: int | None = None, batch_size: int | None = None) -> str:
    """Resolve the measured-policy sequence bucket for one batch shape.

    Results are cached in a bounded LRU (keys include per-batch token counts)
    until :func:`configure_flashdeberta_kernel_overrides` changes the active table.

    :param int seq_len: Padded sequence length.
    :param int | None total_tokens: Active token count, when known.
    :param int | None batch_size: Batch size, when known.
    :return str: Bucket name. ``"default"`` means no measured bucket matched.
    """

    capacity = max(1, int(seq_len) * max(1, int(batch_size or 1)))
    density = None if total_tokens is None else float(total_tokens) / float(capacity)
    buckets = _load_tuning_payload().get("seq_buckets", [])
    for raw in buckets:
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
        return str(raw["name"])
    return "default"


flash_seq_bucket = lru_cache(maxsize=_SHAPE_KEYED_CACHE_MAXSIZE)(flash_seq_bucket)


def flash_route_policy(
    *,
    policy: str,
    seq_bucket: str,
    compute_capability: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Resolve a route-policy row from the active tuning table.

    Rows may scope themselves to one GPU class with a ``compute_capability``
    key such as ``"sm_120"``; rows without the key (or with ``"*"``) apply to
    any hardware. An exact-capability row outranks a wildcard row; within the
    same specificity, later rows win so appended override-table rows take
    precedence.

    Results are cached until :func:`configure_flashdeberta_kernel_overrides`
    changes the active table, so callers must treat the returned row as
    read-only.

    :param str policy: Route policy namespace such as ``"padding"`` or ``"docblock"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :param tuple[int, int] | None compute_capability: Device capability, or None
        to match only hardware-agnostic rows.
    :return dict[str, Any] | None: Matching policy row, or None when the table has no entry.
    """

    policies = _load_tuning_payload().get("route_policies", {})
    choices = policies.get(str(policy).strip().lower())
    if choices is None:
        return None
    cc_key = compute_capability_key(compute_capability) if compute_capability is not None else None
    wildcard_match: dict[str, Any] | None = None
    for raw in reversed(choices):
        if str(raw["seq_bucket"]).strip() != str(seq_bucket).strip():
            continue
        entry_cc = str(raw.get("compute_capability", "*")).strip().lower()
        if entry_cc == "*":
            if wildcard_match is None:
                wildcard_match = raw
            continue
        if cc_key is not None and entry_cc == cc_key:
            return raw
    return wildcard_match


flash_route_policy = cache(flash_route_policy)


def _route_policy_row_allows(
    row: dict[str, Any],
    *,
    seq_len: int | None = None,
    batch_size: int | None = None,
) -> bool:
    """Check a policy row's optional shape bounds against known batch facts.

    Rows may carry ``min_seq_len``/``max_seq_len``/``max_batch_size`` keys;
    each bound is only enforced when the caller supplies the matching fact.

    :param dict[str, Any] row: Route-policy row from the tuning table.
    :param int | None seq_len: Padded sequence length, when known.
    :param int | None batch_size: Batch size, when known.
    :return bool: False when a row bound excludes the supplied facts.
    """

    if seq_len is not None:
        min_seq = row.get("min_seq_len")
        if min_seq is not None and seq_len < int(min_seq):
            return False
        max_seq = row.get("max_seq_len")
        if max_seq is not None and seq_len > int(max_seq):
            return False
    if batch_size is not None:
        max_batch = row.get("max_batch_size")
        if max_batch is not None and batch_size > int(max_batch):
            return False
    return True


def flash_route_choice(
    *,
    policy: str,
    seq_bucket: str,
    compute_capability: tuple[int, int] | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
) -> str | None:
    """Resolve a route choice from the active tuning table.

    :param str policy: Route policy namespace such as ``"padding"`` or ``"docblock"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :param tuple[int, int] | None compute_capability: Device capability, or None
        to match only hardware-agnostic rows.
    :param int | None seq_len: Padded sequence length, when known. Policy rows
        may carry ``min_seq_len``/``max_seq_len`` bounds tighter than their
        bucket (buckets like ``4096_plus`` are open-ended); a row whose bounds
        exclude this length resolves as if the namespace had no entry, so the
        consumer's conservative default applies.
    :param int | None batch_size: Batch size, when known. Rows may carry a
        ``max_batch_size`` bound with the same excluded-row semantics.
    :return str | None: Route choice, or None when the table has no entry.
    """

    raw = flash_route_policy(policy=policy, seq_bucket=seq_bucket, compute_capability=compute_capability)
    if raw is not None:
        if not _route_policy_row_allows(raw, seq_len=seq_len, batch_size=batch_size):
            return None
        return str(raw["choice"]).strip()
    return None


def flash_padding_route(
    *,
    seq_len: int,
    total_tokens: int | None = None,
    batch_size: int | None = None,
    compute_capability: tuple[int, int] | None = None,
) -> str:
    """Resolve the fixed-vs-varlen route for one padded batch shape.

    This is the single policy resolver shared by the training-loop route-hint
    path (which knows batch density) and the model-internal fallback (which
    does not). When ``total_tokens`` is unknown, density-gated table buckets
    are skipped, so callers with density information should always pass it.

    :param int seq_len: Padded sequence length.
    :param int | None total_tokens: Active token count, when known.
    :param int | None batch_size: Batch size, when known.
    :param tuple[int, int] | None compute_capability: Device capability for
        capability-scoped table rows; None matches only hardware-agnostic rows.
    :return str: Either ``"fixed"`` or ``"varlen"``.
    """

    seq_bucket = flash_seq_bucket(
        seq_len=int(seq_len),
        total_tokens=total_tokens,
        batch_size=batch_size,
    )
    table_route = flash_route_choice(
        policy="padding",
        seq_bucket=seq_bucket,
        compute_capability=compute_capability,
        seq_len=int(seq_len),
        batch_size=int(batch_size) if batch_size is not None else None,
    )
    if table_route in {"fixed", "varlen"}:
        return table_route
    return "fixed"


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


def _kernel_launch_tuple(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Parse the launch tuple from one kernel table entry.

    :param dict[str, Any] entry: Matching JSON entry.
    :return tuple[int, int, int, int]: ``(BLOCK_M, BLOCK_N, stages, warps)``.
    """

    return (
        int(entry["block_m"]),
        int(entry["block_n"]),
        int(entry["num_stages"]),
        int(entry["num_warps"]),
    )


def resolve_flash_kernel_config(context: FlashKernelContext) -> tuple[int, int, int, int] | None:
    """Resolve a measured kernel launch tuple from the active tuning table.

    An exact-capability row outranks a wildcard row; within the same
    specificity, later rows win so appended override-table rows take
    precedence, matching :func:`flash_route_policy` and the documented
    override workflow.

    Results are cached in a bounded LRU (contexts carry per-batch token counts)
    until :func:`configure_flashdeberta_kernel_overrides` changes the active table,
    keeping the per-layer forward/backward table scans off the hot path.

    :param FlashKernelContext context: Runtime kernel context.
    :return tuple[int, int, int, int] | None: ``(BLOCK_M, BLOCK_N, stages, warps)`` or None.
    """

    kernels = _load_tuning_payload().get("kernels", [])
    wildcard_match: dict[str, Any] | None = None
    for raw in reversed(kernels):
        if not _entry_matches(context, raw):
            continue
        if str(raw.get("compute_capability", "*")).strip().lower() == "*":
            if wildcard_match is None:
                wildcard_match = raw
            continue
        # _entry_matches already rejected rows scoped to other GPU classes, so
        # any non-wildcard match here is exact for this capability.
        return _kernel_launch_tuple(raw)
    if wildcard_match is not None:
        return _kernel_launch_tuple(wildcard_match)
    return None


resolve_flash_kernel_config = lru_cache(maxsize=_SHAPE_KEYED_CACHE_MAXSIZE)(resolve_flash_kernel_config)
