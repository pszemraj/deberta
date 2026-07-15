"""FlashDeBERTa kernel tuning policy loaded from repo-local JSON data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_TUNING_PATH = Path(__file__).with_name("flashdeberta_kernel_tuning.json")
_ACTIVE_OVERRIDES_PATH: str | None = None
_ACTIVE_OVERRIDES_SIGNATURE: tuple[int, int] | None = None

# Shape-keyed lookups take per-batch values (total_tokens, batch_size) in their
# cache keys, so those caches must be bounded LRUs: a long variably-packed run
# otherwise accretes one permanent entry per distinct batch shape.
_SHAPE_KEYED_CACHE_MAXSIZE = 4096

# Safe launch used when no capability/shape row has been measured. Keep this
# repo-owned and deterministic: upstream selectors also expose environment
# variable overrides, which are deliberately not part of this project's
# configuration contract.
CONSERVATIVE_FLASH_KERNEL_CONFIG = (16, 16, 1, 4)

_SUPPORTED_KERNEL_KINDS = {
    "bias": frozenset({"fwd", "bwd", "bwd_q", "bwd_kv"}),
    "bias_docblock_specialized": frozenset({"bwd", "bwd_q", "bwd_kv"}),
    "dense_bias": frozenset({"fwd"}),
    "docblock": frozenset({"fwd", "bwd_q", "bwd_kv"}),
    "fixed": frozenset({"fwd", "bwd"}),
    "varlen": frozenset({"fwd", "bwd_q", "bwd_kv"}),
}


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


def _overrides_file_signature(path: str) -> tuple[int, int] | None:
    """Return ``(mtime_ns, size)`` for an override table, or None when unreadable.

    :param str path: Override table path.
    :return tuple[int, int] | None: File content signature, or None if stat fails.
    """

    try:
        stat = Path(path).expanduser().stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def configure_flashdeberta_kernel_overrides(
    path: str | None,
    *,
    reload_if_changed: bool = True,
) -> None:
    """Set the process-local FlashDeBERTa kernel override table.

    Runtime hot-path callers should pass ``reload_if_changed=False`` so reapplying the
    configured path is a string comparison rather than a filesystem operation.
    Tuning tools that intentionally rewrite one path can request an mtime/size
    check explicitly.

    :param str | None path: JSON table path, or None to use only the package default.
    :param bool reload_if_changed: Whether to stat and reload an already-active
        path when its content signature changed, defaults to True.
    """

    global _ACTIVE_OVERRIDES_PATH, _ACTIVE_OVERRIDES_SIGNATURE
    normalized = str(path).strip() if path is not None else ""
    resolved = normalized or None
    if resolved == _ACTIVE_OVERRIDES_PATH:
        if not reload_if_changed:
            return
        signature = _overrides_file_signature(resolved) if resolved is not None else None
        if signature == _ACTIVE_OVERRIDES_SIGNATURE:
            return
    else:
        signature = (
            _overrides_file_signature(resolved) if resolved is not None and reload_if_changed else None
        )

    previous_path = _ACTIVE_OVERRIDES_PATH
    previous_signature = _ACTIVE_OVERRIDES_SIGNATURE
    _ACTIVE_OVERRIDES_PATH = resolved
    _ACTIVE_OVERRIDES_SIGNATURE = signature
    _load_tuning_payload.cache_clear()
    flash_seq_bucket.cache_clear()
    flash_route_policy.cache_clear()
    resolve_flash_kernel_config.cache_clear()
    try:
        # Load eagerly so malformed override tables fail at configuration time
        # with their filename and row path, not later inside route selection.
        _load_tuning_payload()
    except Exception:
        _ACTIVE_OVERRIDES_PATH = previous_path
        _ACTIVE_OVERRIDES_SIGNATURE = previous_signature
        _load_tuning_payload.cache_clear()
        flash_seq_bucket.cache_clear()
        flash_route_policy.cache_clear()
        resolve_flash_kernel_config.cache_clear()
        raise


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

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid FlashDeBERTa tuning JSON {path}:{exc.lineno}:{exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"FlashDeBERTa tuning table must be a JSON object: {path}")
    return payload


def _override_error(path: Path, location: str, message: str) -> ValueError:
    """Build a contextual override-table validation error.

    :param Path path: Override table path.
    :param str location: JSON-style section or row path.
    :param str message: Validation failure detail.
    :return ValueError: Contextual validation exception.
    """

    return ValueError(f"Invalid FlashDeBERTa tuning override {path} at {location}: {message}")


def _validate_override_row_mapping(*, path: Path, location: str, row: Any) -> dict[str, Any]:
    """Return one override row after requiring a JSON object.

    :param Path path: Override table path.
    :param str location: JSON-style row path.
    :param Any row: Candidate row.
    :raises ValueError: If the row is not a mapping.
    :return dict[str, Any]: Validated row mapping.
    """

    if not isinstance(row, dict):
        raise _override_error(path, location, "expected a JSON object")
    return row


def _validate_override_text(
    row: dict[str, Any],
    *,
    key: str,
    path: Path,
    location: str,
) -> str:
    """Return one required non-empty string override field.

    :param dict[str, Any] row: Override row.
    :param str key: Required field name.
    :param Path path: Override table path.
    :param str location: JSON-style row path.
    :raises ValueError: If the field is absent or empty.
    :return str: Validated text.
    """

    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _override_error(path, f"{location}.{key}", "expected a non-empty string")
    return value.strip()


def _validate_override_int_fields(
    row: dict[str, Any],
    *,
    keys: tuple[str, ...],
    path: Path,
    location: str,
    minimum: int = 0,
) -> None:
    """Validate present integer fields in one override row.

    :param dict[str, Any] row: Override row.
    :param tuple[str, ...] keys: Integer field names.
    :param Path path: Override table path.
    :param str location: JSON-style row path.
    :param int minimum: Minimum accepted value, defaults to 0.
    :raises ValueError: If a present field is not an integer in range.
    """

    for key in keys:
        if key not in row:
            continue
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < int(minimum):
            raise _override_error(path, f"{location}.{key}", f"expected an integer >= {minimum}")


def _validate_override_payload(payload: dict[str, Any], *, path: Path) -> None:
    """Validate user-supplied tuning sections before merging shipped defaults.

    :param dict[str, Any] payload: Override-only JSON payload.
    :param Path path: Override table path.
    :raises ValueError: If a supported section or row is malformed.
    """

    allowed_sections = {"version", "seq_buckets", "route_policies", "kernels"}
    unknown_sections = sorted(set(payload) - allowed_sections)
    if unknown_sections:
        raise _override_error(path, "$", f"unknown section(s): {', '.join(unknown_sections)}")

    if "version" in payload:
        version = payload["version"]
        if isinstance(version, bool) or not isinstance(version, int):
            raise _override_error(path, "version", "expected an integer")

    if "seq_buckets" in payload:
        buckets = payload["seq_buckets"]
        if not isinstance(buckets, list):
            raise _override_error(path, "seq_buckets", "expected a list of row objects")
        for index, raw in enumerate(buckets):
            location = f"seq_buckets[{index}]"
            row = _validate_override_row_mapping(path=path, location=location, row=raw)
            _validate_override_text(row, key="name", path=path, location=location)
            _validate_override_int_fields(
                row,
                keys=("min_seq_len", "max_seq_len"),
                path=path,
                location=location,
            )
            for key in ("min_density", "max_density", "max_density_exclusive"):
                if key not in row:
                    continue
                value = row[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise _override_error(path, f"{location}.{key}", "expected a number")

    if "route_policies" in payload:
        policies = payload["route_policies"]
        if not isinstance(policies, dict):
            raise _override_error(path, "route_policies", "expected an object of policy row lists")
        allowed_choices = {
            "padding": {"fixed", "varlen"},
            "docblock": {"docblock", "docblock_bias"},
            "local_bias": {"local_bias"},
        }
        for namespace, rows in policies.items():
            namespace_location = f"route_policies.{namespace}"
            if namespace not in allowed_choices:
                raise _override_error(path, namespace_location, "unknown route-policy namespace")
            if not isinstance(rows, list):
                raise _override_error(path, namespace_location, "expected a list of row objects")
            for index, raw in enumerate(rows):
                location = f"{namespace_location}[{index}]"
                row = _validate_override_row_mapping(path=path, location=location, row=raw)
                _validate_override_text(row, key="seq_bucket", path=path, location=location)
                choice = _validate_override_text(row, key="choice", path=path, location=location)
                if choice not in allowed_choices[namespace]:
                    expected = ", ".join(sorted(allowed_choices[namespace]))
                    raise _override_error(
                        path,
                        f"{location}.choice",
                        f"expected one of: {expected}; got {choice!r}",
                    )
                _validate_override_int_fields(
                    row,
                    keys=("min_seq_len", "max_seq_len", "max_batch_size"),
                    path=path,
                    location=location,
                )

    if "kernels" in payload:
        kernels = payload["kernels"]
        if not isinstance(kernels, list):
            raise _override_error(path, "kernels", "expected a list of row objects")
        launch_fields = ("block_m", "block_n", "num_stages", "num_warps")
        integer_fields = (
            "batch_size",
            "query_len",
            "key_len",
            "num_heads",
            "min_batch_size",
            "max_batch_size",
            "min_query_len",
            "max_query_len",
            "min_key_len",
            "max_key_len",
            "min_num_heads",
            "max_num_heads",
            "att_span_min",
        )
        for index, raw in enumerate(kernels):
            location = f"kernels[{index}]"
            row = _validate_override_row_mapping(path=path, location=location, row=raw)
            route = _validate_override_text(row, key="route", path=path, location=location).lower()
            kind = _validate_override_text(row, key="kind", path=path, location=location).lower()
            _validate_override_text(row, key="seq_bucket", path=path, location=location)
            allowed_kinds = _SUPPORTED_KERNEL_KINDS.get(route)
            if allowed_kinds is None:
                expected = ", ".join(sorted(_SUPPORTED_KERNEL_KINDS))
                raise _override_error(
                    path,
                    f"{location}.route",
                    f"expected one of: {expected}; got {route!r}",
                )
            if kind not in allowed_kinds:
                expected = ", ".join(sorted(allowed_kinds))
                raise _override_error(
                    path,
                    f"{location}.kind",
                    f"expected one of for route {route!r}: {expected}; got {kind!r}",
                )
            missing_launch = [key for key in launch_fields if key not in row]
            if missing_launch:
                raise _override_error(
                    path,
                    location,
                    f"incomplete kernel launch tuple; missing: {', '.join(missing_launch)}",
                )
            _validate_override_int_fields(
                row,
                keys=launch_fields,
                path=path,
                location=location,
                minimum=1,
            )
            _validate_override_int_fields(
                row,
                keys=integer_fields,
                path=path,
                location=location,
            )
            head_dim = row.get("head_dim", "*")
            if head_dim != "*" and (
                isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0
            ):
                raise _override_error(
                    path,
                    f"{location}.head_dim",
                    "expected '*' or an integer >= 1",
                )


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
    _validate_override_payload(overrides, path=expanded_override_path)
    merged: dict[str, Any] = dict(payload)
    for key in ("seq_buckets", "route_policies", "kernels"):
        base_value = payload.get(key)
        override_value = overrides.get(key)
        if isinstance(base_value, list) or isinstance(override_value, list):
            base_rows = base_value if isinstance(base_value, list) else []
            override_rows = override_value if isinstance(override_value, list) else []
            if key == "seq_buckets":
                # Buckets resolve first-match in list order and the shipped
                # set covers every length, so appended override buckets would
                # be unreachable; prepend them instead. route_policies/kernels
                # iterate reversed, so appending keeps "later rows win" there.
                # A same-name override replaces the shipped bucket outright:
                # keeping the shipped row would make a narrowing override
                # (tighter max_seq_len/density window) silently fall through
                # to the original shipped range.
                override_names = {
                    row.get("name") for row in override_rows if isinstance(row, dict) and row.get("name")
                }
                kept_base_rows = [
                    row
                    for row in base_rows
                    if not (isinstance(row, dict) and row.get("name") in override_names)
                ]
                merged[key] = [*override_rows, *kept_base_rows]
            else:
                merged[key] = [*base_rows, *override_rows]
        elif isinstance(base_value, dict) or isinstance(override_value, dict):
            combined = dict(base_value if isinstance(base_value, dict) else {})
            for namespace, override_rows in (
                override_value if isinstance(override_value, dict) else {}
            ).items():
                base_rows = combined.get(namespace)
                if isinstance(base_rows, list) and isinstance(override_rows, list):
                    combined[namespace] = [*base_rows, *override_rows]
                else:
                    combined[namespace] = override_rows
            merged[key] = combined
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
    choices = policies.get(str(policy).strip().lower()) if isinstance(policies, dict) else None
    if not isinstance(choices, list):
        return None
    cc_key = compute_capability_key(compute_capability) if compute_capability is not None else None
    wildcard_match: dict[str, Any] | None = None
    for raw in reversed(choices):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("seq_bucket", "")).strip() != str(seq_bucket).strip():
            continue
        entry_cc = str(raw.get("compute_capability", "*")).strip().lower()
        if entry_cc == "*":
            if wildcard_match is None:
                wildcard_match = dict(raw)
            continue
        if cc_key is not None and entry_cc == cc_key:
            return dict(raw)
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
        choice = raw.get("choice")
        return str(choice).strip() if choice is not None else None
    return None


def tuned_capability_keys() -> frozenset[str]:
    """Return the explicit compute-capability keys present in the active table.

    :return frozenset[str]: Keys such as ``{"sm_120"}``; wildcard rows are excluded.
    """

    payload = _load_tuning_payload()
    rows: list[Any] = []
    kernels = payload.get("kernels", [])
    if isinstance(kernels, list):
        rows.extend(kernels)
    policies = payload.get("route_policies", {})
    if isinstance(policies, dict):
        for choices in policies.values():
            if isinstance(choices, list):
                rows.extend(choices)
    keys: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        value = str(raw.get("compute_capability", "*")).strip().lower()
        if value and value != "*":
            keys.add(value)
    return frozenset(keys)


def flash_padding_route(
    *,
    seq_len: int,
    total_tokens: int | None = None,
    batch_size: int | None = None,
    force_varlen: bool = False,
    varlen_min_seq_len: int | None = None,
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
    :param bool force_varlen: Config override that forces the varlen route.
    :param int | None varlen_min_seq_len: Optional config threshold overriding the table.
    :param tuple[int, int] | None compute_capability: Device capability for
        capability-scoped table rows; None matches only hardware-agnostic rows.
    :return str: Either ``"fixed"`` or ``"varlen"``.
    """

    if force_varlen:
        return "varlen"
    if varlen_min_seq_len is not None:
        threshold = max(1, int(varlen_min_seq_len))
        return "varlen" if int(seq_len) >= threshold else "fixed"
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


def _kernel_launch_tuple(entry: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Parse the launch tuple from one kernel table entry.

    :param dict[str, Any] entry: Matching JSON entry.
    :return tuple[int, int, int, int] | None: ``(BLOCK_M, BLOCK_N, stages, warps)``
        or None when the entry is malformed.
    """

    try:
        return (
            int(entry["block_m"]),
            int(entry["block_n"]),
            int(entry["num_stages"]),
            int(entry["num_warps"]),
        )
    except Exception:
        return None


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
    if not isinstance(kernels, list):
        return None
    wildcard_match: dict[str, Any] | None = None
    for raw in reversed(kernels):
        if not isinstance(raw, dict) or not _entry_matches(context, raw):
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


def resolve_repo_tuned_config(
    *,
    guard: Callable[[], bool],
    compute_capability: Callable[[], tuple[int, int]],
    route: str,
    kind: str,
    seq_len: int,
    head_dim: int,
    total_tokens: int | None = None,
    batch_size: int | None = None,
    query_len: int | None = None,
    key_len: int | None = None,
    num_heads: int | None = None,
    dtype: str | None = None,
    causal: bool | None = None,
    disentangled: bool | None = None,
    att_span: int | None = None,
    has_mask: bool | None = None,
) -> tuple[int, int, int, int] | None:
    """Resolve a tuning row after one route applies its explicit safety guard.

    :param Callable guard: Route-specific launch safety guard.
    :param Callable compute_capability: Deferred CUDA compute-capability resolver.
    :param str route: Tuning-table route namespace.
    :param str kind: Kernel kind within the route.
    :param int seq_len: Representative sequence length.
    :param int head_dim: Attention head dimension.
    :param int | None total_tokens: Active or capacity token count, defaults to None.
    :param int | None batch_size: Batch size, defaults to None.
    :param int | None query_len: Query length, defaults to None.
    :param int | None key_len: Key length, defaults to None.
    :param int | None num_heads: Attention head count, defaults to None.
    :param str | None dtype: Normalized kernel dtype, defaults to None.
    :param bool | None causal: Causal-attention flag, defaults to None.
    :param bool | None disentangled: Disentangled-attention flag, defaults to None.
    :param int | None att_span: Relative-position span, defaults to None.
    :param bool | None has_mask: Dense-mask presence flag, defaults to None.
    :return tuple[int, int, int, int] | None: Tuned launch tuple, or None when guarded or unmatched.
    """

    if not guard():
        return None
    return resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=compute_capability(),
            route=str(route),
            kind=str(kind),
            seq_len=int(seq_len),
            head_dim=int(head_dim),
            total_tokens=int(total_tokens) if total_tokens is not None else None,
            batch_size=int(batch_size) if batch_size is not None else None,
            query_len=int(query_len) if query_len is not None else None,
            key_len=int(key_len) if key_len is not None else None,
            num_heads=int(num_heads) if num_heads is not None else None,
            dtype=str(dtype) if dtype is not None else None,
            causal=bool(causal) if causal is not None else None,
            disentangled=bool(disentangled) if disentangled is not None else None,
            att_span=int(att_span) if att_span is not None else None,
            has_mask=bool(has_mask) if has_mask is not None else None,
        )
    )
