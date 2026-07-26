"""FlashDeBERTa kernel tuning policy loaded from repo-local JSON data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

_DEFAULT_TUNING_PATH = Path(__file__).with_name("flashdeberta_kernel_tuning.json")
_POLICY_KEY_PREFIX = "flash-policy:"
_MATERIALIZED_POLICY_PAYLOADS: dict[str, dict[str, Any]] = {}
_TRITON_MAX_TENSOR_NUMEL = 1_048_576

# Shape-keyed lookups take per-batch values (total_tokens, batch_size) in their
# cache keys, so those caches must be bounded LRUs: a long variably-packed run
# otherwise accretes one permanent entry per distinct batch shape.
_SHAPE_KEYED_CACHE_MAXSIZE = 4096

# Safe launch used when no capability/shape row has been measured. Keep this
# repo-owned and deterministic: upstream selectors also expose environment
# variable overrides, which are deliberately not part of this project's
# configuration contract.
CONSERVATIVE_FLASH_KERNEL_CONFIG = (16, 16, 1, 4)

_ROUTE_POLICY_CHOICES = {
    "padding": {"fixed", "varlen"},
    "local_bias": {"local_bias"},
}
_KERNEL_KIND_CHOICES = {
    "bias": {"fwd", "bwd", "bwd_q", "bwd_kv"},
    "bias_docblock_specialized": {"bwd", "bwd_q", "bwd_kv"},
    "dense_bias": {"fwd"},
    "docblock": {"fwd", "bwd_q", "bwd_kv"},
    "fixed": {"fwd", "bwd"},
    "varlen": {"fwd", "bwd_q", "bwd_kv"},
}

_SEQ_BUCKET_FIELDS = {
    "name",
    "min_seq_len",
    "max_seq_len",
    "min_density",
    "max_density",
    "max_density_exclusive",
}
_ROUTE_POLICY_FIELDS = {
    "seq_bucket",
    "choice",
    "compute_capability",
    "min_seq_len",
    "max_seq_len",
    "max_batch_size",
}
_KERNEL_FIELDS = {
    "route",
    "kind",
    "seq_bucket",
    "compute_capability",
    "head_dim",
    "batch_size",
    "min_batch_size",
    "max_batch_size",
    "query_len",
    "min_query_len",
    "max_query_len",
    "key_len",
    "min_key_len",
    "max_key_len",
    "num_heads",
    "min_num_heads",
    "max_num_heads",
    "dtype",
    "causal",
    "disentangled",
    "att_span_min",
    "has_mask",
    "block_m",
    "block_n",
    "num_stages",
    "num_warps",
}


@dataclass(frozen=True)
class FlashKernelContext:
    """Runtime facts used to match a FlashDeBERTa kernel tuning entry."""

    compute_capability: tuple[int, int]
    route: str
    kind: str
    seq_len: int
    head_dim: int
    seq_bucket: str | None = None
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


@dataclass(frozen=True)
class FlashKernelPolicy:
    """One validated, immutable FlashDeBERTa override-table snapshot."""

    source_path: str
    key: str


def normalize_flash_kernel_policy_path(path: str | None) -> str:
    """Return one canonical override path or the shipped-policy sentinel.

    :param str | None path: JSON table path, or None to use only the package default.
    :return str: Absolute expanded override path, or ``""`` for the shipped policy.
    """

    normalized = str(path).strip() if path is not None else ""
    if not normalized:
        return ""
    return str(Path(normalized).expanduser().resolve())


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


def _require_mapping(value: Any, *, location: str) -> dict[str, Any]:
    """Return one tuning-table mapping or raise a contextual validation error.

    :param Any value: Candidate JSON value.
    :param str location: Human-readable payload location.
    :raises ValueError: If the value is not a mapping.
    :return dict[str, Any]: Validated mapping.
    """

    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object, got {type(value).__name__}.")
    return value


def _require_rows(value: Any, *, location: str) -> list[Any]:
    """Return one tuning-table row list or raise a contextual validation error.

    :param Any value: Candidate JSON value.
    :param str location: Human-readable payload location.
    :raises ValueError: If the value is not a list.
    :return list[Any]: Validated row list.
    """

    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON array, got {type(value).__name__}.")
    return value


def _require_fields(row: dict[str, Any], *, required: set[str], location: str) -> None:
    """Require the fields consumed directly by one tuning-table resolver.

    :param dict[str, Any] row: Candidate tuning row.
    :param set[str] required: Required field names.
    :param str location: Human-readable payload location.
    :raises ValueError: If any required field is absent.
    """

    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{location} is missing required field(s): {', '.join(missing)}.")


def _reject_unknown_fields(
    row: dict[str, Any],
    *,
    allowed: set[str],
    location: str,
) -> None:
    """Reject fields that no executable-policy resolver consumes.

    :param dict[str, Any] row: Candidate tuning row.
    :param set[str] allowed: Complete supported field set.
    :param str location: Human-readable payload location.
    :raises ValueError: If the row contains an unknown field.
    """

    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"{location} has unknown field(s): {', '.join(unknown)}.")


def _require_positive_int(row: dict[str, Any], *, field_name: str, location: str) -> int | None:
    """Validate one optional positive-integer tuning field.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Field to validate when present.
    :param str location: Human-readable payload location.
    :raises ValueError: If the field is not a positive JSON integer.
    :return int | None: Validated value, or None when absent.
    """

    value = row.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{location}.{field_name} must be a positive integer.")
    return value


def _require_nonempty_string(
    row: dict[str, Any],
    *,
    field_name: str,
    location: str,
) -> str:
    """Validate one required non-empty string field.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Required field to validate.
    :param str location: Human-readable payload location.
    :raises ValueError: If the field is absent or is not a non-empty string.
    :return str: Stripped field value.
    """

    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field_name} must be a non-empty string.")
    return value.strip()


def _require_power_of_two(row: dict[str, Any], *, field_name: str, location: str) -> int:
    """Validate one required power-of-two launch field.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Required field to validate.
    :param str location: Human-readable payload location.
    :raises ValueError: If the field is absent or is not a positive power of two.
    :return int: Validated launch value.
    """

    value = _require_positive_int(row, field_name=field_name, location=location)
    if value is None or value & (value - 1):
        raise ValueError(f"{location}.{field_name} must be a positive power of two.")
    return value


def _require_bool(row: dict[str, Any], *, field_name: str, location: str) -> bool | None:
    """Validate one optional boolean matching constraint.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Field to validate when present.
    :param str location: Human-readable payload location.
    :raises ValueError: If the field is not a JSON boolean.
    :return bool | None: Validated value, or None when absent.
    """

    value = row.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{location}.{field_name} must be a boolean.")
    return value


def _require_unit_interval_number(
    row: dict[str, Any],
    *,
    field_name: str,
    location: str,
) -> float | None:
    """Validate one optional finite density in the closed unit interval.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Field to validate when present.
    :param str location: Human-readable payload location.
    :raises ValueError: If the field is not a finite JSON number in ``[0, 1]``.
    :return float | None: Validated value, or None when absent.
    """

    value = row.get(field_name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}.{field_name} must be a finite number in [0, 1].")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{location}.{field_name} must be a finite number in [0, 1].")
    return parsed


def _validate_positive_int_range(
    row: dict[str, Any],
    *,
    field_name: str,
    location: str,
) -> tuple[int | None, int | None]:
    """Validate optional ``min_*``/``max_*`` positive-integer bounds.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Unprefixed bounded field name.
    :param str location: Human-readable payload location.
    :raises ValueError: If a bound is invalid or the minimum exceeds the maximum.
    :return tuple[int | None, int | None]: Validated minimum and maximum.
    """

    min_name = f"min_{field_name}"
    max_name = f"max_{field_name}"
    min_value = _require_positive_int(row, field_name=min_name, location=location)
    max_value = _require_positive_int(row, field_name=max_name, location=location)
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{location}.{min_name} must be <= {max_name}.")
    return min_value, max_value


def _validate_exact_positive_int_range(
    row: dict[str, Any],
    *,
    field_name: str,
    location: str,
) -> None:
    """Validate one optional exact integer against its optional range.

    :param dict[str, Any] row: Candidate tuning row.
    :param str field_name: Unprefixed exact and bounded field name.
    :param str location: Human-readable payload location.
    :raises ValueError: If the exact value cannot satisfy its declared range.
    """

    exact = _require_positive_int(row, field_name=field_name, location=location)
    minimum, maximum = _validate_positive_int_range(
        row,
        field_name=field_name,
        location=location,
    )
    if exact is not None and minimum is not None and exact < minimum:
        raise ValueError(f"{location}.{field_name} must be >= min_{field_name}.")
    if exact is not None and maximum is not None and exact > maximum:
        raise ValueError(f"{location}.{field_name} must be <= max_{field_name}.")


def _validate_override_payload(payload: dict[str, Any], *, source: Path) -> None:
    """Validate the override-table structure consumed by route and kernel lookup.

    :param dict[str, Any] payload: Parsed override payload.
    :param Path source: Source path used in diagnostics.
    :raises ValueError: If the required override schema is invalid.
    """

    unknown = sorted(set(payload) - {"version", "seq_buckets", "route_policies", "kernels"})
    if unknown:
        raise ValueError(f"Unknown top-level field(s) in {source}: {', '.join(unknown)}.")

    override_bucket_names: set[str] = set()
    for index, raw in enumerate(_require_rows(payload.get("seq_buckets", []), location="seq_buckets")):
        location = f"seq_buckets[{index}]"
        row = _require_mapping(raw, location=location)
        _reject_unknown_fields(row, allowed=_SEQ_BUCKET_FIELDS, location=location)
        _require_fields(row, required={"name"}, location=location)
        override_bucket_names.add(_require_nonempty_string(row, field_name="name", location=location))
        _validate_positive_int_range(row, field_name="seq_len", location=location)
        min_density = _require_unit_interval_number(
            row,
            field_name="min_density",
            location=location,
        )
        for max_name in ("max_density", "max_density_exclusive"):
            max_density = _require_unit_interval_number(
                row,
                field_name=max_name,
                location=location,
            )
            if min_density is not None and max_density is not None and min_density > max_density:
                raise ValueError(f"{location}.min_density must be <= {max_name}.")

    shipped_payload = _require_mapping(
        _read_json(_DEFAULT_TUNING_PATH),
        location=f"shipped tuning table at {_DEFAULT_TUNING_PATH}",
    )
    shipped_bucket_names = {
        str(row["name"]).strip()
        for row in _require_rows(
            shipped_payload.get("seq_buckets", []),
            location="shipped seq_buckets",
        )
    }
    known_bucket_names = {"default", *shipped_bucket_names, *override_bucket_names}

    policies = _require_mapping(payload.get("route_policies", {}), location="route_policies")
    for policy, raw_rows in policies.items():
        choices = _ROUTE_POLICY_CHOICES.get(str(policy))
        if choices is None:
            allowed = ", ".join(sorted(_ROUTE_POLICY_CHOICES))
            raise ValueError(f"Unknown route policy {policy!r}; expected one of: {allowed}.")
        for index, raw in enumerate(_require_rows(raw_rows, location=f"route_policies.{policy}")):
            location = f"route_policies.{policy}[{index}]"
            row = _require_mapping(raw, location=location)
            _reject_unknown_fields(row, allowed=_ROUTE_POLICY_FIELDS, location=location)
            _require_fields(row, required={"seq_bucket", "choice"}, location=location)
            seq_bucket = _require_nonempty_string(
                row,
                field_name="seq_bucket",
                location=location,
            )
            if seq_bucket not in known_bucket_names:
                raise ValueError(f"{location}.seq_bucket references unknown bucket {seq_bucket!r}.")
            choice = _require_nonempty_string(row, field_name="choice", location=location)
            if choice not in choices:
                allowed = ", ".join(sorted(choices))
                raise ValueError(f"{location}.choice must be one of: {allowed}. Got {choice!r}.")
            if "compute_capability" in row:
                _require_nonempty_string(
                    row,
                    field_name="compute_capability",
                    location=location,
                )
            _validate_positive_int_range(row, field_name="seq_len", location=location)
            _require_positive_int(row, field_name="max_batch_size", location=location)

    for index, raw in enumerate(_require_rows(payload.get("kernels", []), location="kernels")):
        location = f"kernels[{index}]"
        row = _require_mapping(raw, location=location)
        _reject_unknown_fields(row, allowed=_KERNEL_FIELDS, location=location)
        _require_fields(
            row,
            required={
                "route",
                "kind",
                "seq_bucket",
                "block_m",
                "block_n",
                "num_stages",
                "num_warps",
            },
            location=location,
        )
        route = _require_nonempty_string(row, field_name="route", location=location)
        kind = _require_nonempty_string(row, field_name="kind", location=location)
        kinds = _KERNEL_KIND_CHOICES.get(route)
        if kinds is None or kind not in kinds:
            raise ValueError(f"{location} has unsupported route/kind pair {route!r}/{kind!r}.")
        seq_bucket = _require_nonempty_string(
            row,
            field_name="seq_bucket",
            location=location,
        )
        if seq_bucket not in known_bucket_names:
            raise ValueError(f"{location}.seq_bucket references unknown bucket {seq_bucket!r}.")
        block_m = _require_power_of_two(row, field_name="block_m", location=location)
        block_n = _require_power_of_two(row, field_name="block_n", location=location)
        if block_m < 16 or block_n < 16:
            raise ValueError(
                f"{location}.block_m and block_n must be at least 16; Triton's tl.dot "
                "requires 16 per dimension, so a smaller tile crashes at kernel compile."
            )
        if route == "dense_bias" and block_m * block_n > _TRITON_MAX_TENSOR_NUMEL:
            raise ValueError(
                f"{location} dense_bias tile has {block_m * block_n} elements; "
                f"Triton tensors support at most {_TRITON_MAX_TENSOR_NUMEL}."
            )
        _require_positive_int(row, field_name="num_stages", location=location)
        num_warps = _require_power_of_two(row, field_name="num_warps", location=location)
        if num_warps not in {1, 2, 4, 8}:
            raise ValueError(f"{location}.num_warps must be one of 1, 2, 4, or 8.")
        if row.get("head_dim") != "*":
            _require_positive_int(row, field_name="head_dim", location=location)
        for field_name in ("batch_size", "query_len", "key_len", "num_heads"):
            _validate_exact_positive_int_range(
                row,
                field_name=field_name,
                location=location,
            )
        for field_name in ("compute_capability", "dtype"):
            if field_name in row:
                _require_nonempty_string(
                    row,
                    field_name=field_name,
                    location=location,
                )
        _require_positive_int(row, field_name="att_span_min", location=location)
        for field_name in ("causal", "disentangled", "has_mask"):
            _require_bool(row, field_name=field_name, location=location)


def _load_override_payload(path: Path) -> dict[str, Any]:
    """Read and validate one user-provided tuning override table.

    :param Path path: Override-table path.
    :raises ValueError: If the file cannot be loaded or its required schema is invalid.
    :return dict[str, Any]: Validated override payload.
    """

    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load FlashDeBERTa kernel overrides from {path}: {exc}") from exc
    payload = _require_mapping(raw, location=f"FlashDeBERTa kernel overrides at {path}")
    _validate_override_payload(payload, source=path)
    return payload


def validate_flashdeberta_kernel_overrides(path: str) -> None:
    """Validate one configured FlashDeBERTa kernel override table.

    :param str path: JSON override-table path.
    :raises ValueError: If the file cannot be loaded or its required schema is invalid.
    """

    policy = materialize_flash_kernel_policy(path)
    if not policy.source_path:
        raise ValueError("FlashDeBERTa kernel override path must be non-empty.")


def _merge_tuning_payload(overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge one validated override payload onto the shipped tuning table.

    ``route_policies`` and ``kernels`` override rows append to the shipped
    rows rather than replacing them, so an override table can promote a
    single row (for example one capability-scoped route choice) without
    restating the shipped policy; appended rows outrank shipped rows of the
    same specificity. ``seq_buckets`` override rows prepend, and a row whose
    ``name`` matches a shipped bucket replaces that bucket entirely.

    :param dict[str, Any] overrides: Validated override-table payload.
    :return dict[str, Any]: Merged tuning payload.
    """

    payload = _read_json(_DEFAULT_TUNING_PATH)
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


def materialize_flash_kernel_policy(path: str | None) -> FlashKernelPolicy:
    """Load one override path into an immutable content-addressed policy snapshot.

    :param str | None path: Optional override-table path.
    :return FlashKernelPolicy: Normalized source path and immutable policy key.
    """

    source_path = normalize_flash_kernel_policy_path(path)
    if not source_path:
        return FlashKernelPolicy(source_path="", key="")

    overrides = _load_override_payload(Path(source_path))
    canonical = json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode("utf-8")
    policy_key = f"{_POLICY_KEY_PREFIX}{hashlib.sha256(canonical).hexdigest()}"
    if policy_key not in _MATERIALIZED_POLICY_PAYLOADS:
        _MATERIALIZED_POLICY_PAYLOADS[policy_key] = _merge_tuning_payload(overrides)
    return FlashKernelPolicy(source_path=source_path, key=policy_key)


def _normalize_flash_kernel_policy_key(path_or_key: str | None) -> str:
    """Return an immutable policy key from either a source path or existing key.

    :param str | None path_or_key: Override path, immutable key, or None.
    :raises ValueError: If an unknown immutable key is supplied.
    :return str: Immutable policy key, or ``""`` for the shipped policy.
    """

    value = str(path_or_key).strip() if path_or_key is not None else ""
    if not value:
        return ""
    if value.startswith(_POLICY_KEY_PREFIX):
        if value not in _MATERIALIZED_POLICY_PAYLOADS:
            raise ValueError(f"Unknown materialized FlashDeBERTa kernel policy key: {value}.")
        return value
    return materialize_flash_kernel_policy(value).key


def _load_tuning_payload(policy_key: str) -> dict[str, Any]:
    """Load the shipped table or one materialized immutable override snapshot.

    :param str policy_key: Immutable policy key, or ``""`` for shipped policy.
    :raises ValueError: If the key was not materialized in this process.
    :return dict[str, Any]: Tuning payload.
    """

    if not policy_key:
        return _read_json(_DEFAULT_TUNING_PATH)
    payload = _MATERIALIZED_POLICY_PAYLOADS.get(policy_key)
    if payload is None:
        raise ValueError(f"Unknown materialized FlashDeBERTa kernel policy key: {policy_key}.")
    return payload


_load_tuning_payload = cache(_load_tuning_payload)


def _flash_seq_bucket(
    *,
    policy_path: str,
    seq_len: int,
    total_tokens: int | None,
    batch_size: int | None,
) -> str:
    """Resolve one sequence bucket for a normalized policy path.

    :param str policy_path: Normalized override path, or ``""``.
    :param int seq_len: Padded sequence length.
    :param int | None total_tokens: Active token count, when known.
    :param int | None batch_size: Batch size, when known.
    :return str: Matching bucket name, or ``"default"``.
    """

    capacity = max(1, int(seq_len) * max(1, int(batch_size or 1)))
    density = None if total_tokens is None else float(total_tokens) / float(capacity)
    buckets = _load_tuning_payload(policy_path).get("seq_buckets", [])
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


_flash_seq_bucket = lru_cache(maxsize=_SHAPE_KEYED_CACHE_MAXSIZE)(_flash_seq_bucket)


def flash_seq_bucket(
    *,
    seq_len: int,
    total_tokens: int | None = None,
    batch_size: int | None = None,
    policy_path: str | None = None,
) -> str:
    """Resolve the measured-policy sequence bucket for one batch shape.

    Results are cached in a bounded LRU (keys include per-batch token counts)
    independently for each immutable policy path.

    :param int seq_len: Padded sequence length.
    :param int | None total_tokens: Active token count, when known.
    :param int | None batch_size: Batch size, when known.
    :param str | None policy_path: Optional kernel-override table path.
    :return str: Bucket name. ``"default"`` means no measured bucket matched.
    """

    return _flash_seq_bucket(
        policy_path=_normalize_flash_kernel_policy_key(policy_path),
        seq_len=int(seq_len),
        total_tokens=total_tokens,
        batch_size=batch_size,
    )


flash_seq_bucket.cache_clear = _flash_seq_bucket.cache_clear  # type: ignore[attr-defined]
flash_seq_bucket.cache_info = _flash_seq_bucket.cache_info  # type: ignore[attr-defined]


def _flash_route_policy(
    *,
    policy_path: str,
    policy: str,
    seq_bucket: str,
    compute_capability: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Resolve a route-policy row from one model-scoped tuning table.

    Rows may scope themselves to one GPU class with a ``compute_capability``
    key such as ``"sm_120"``; rows without the key (or with ``"*"``) apply to
    any hardware. An exact-capability row outranks a wildcard row; within the
    same specificity, later rows win so appended override-table rows take
    precedence.

    :param str policy_path: Normalized override path, or ``""``.
    :param str policy: Route policy namespace such as ``"padding"`` or ``"local_bias"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :param tuple[int, int] | None compute_capability: Device capability, or None
        to match only hardware-agnostic rows.
    :return dict[str, Any] | None: Matching policy row, or None when the table has no entry.
    """

    policies = _load_tuning_payload(policy_path).get("route_policies", {})
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


_flash_route_policy = cache(_flash_route_policy)


def flash_route_policy(
    *,
    policy: str,
    seq_bucket: str,
    compute_capability: tuple[int, int] | None = None,
    policy_path: str | None = None,
) -> dict[str, Any] | None:
    """Resolve one read-only route-policy row under an explicit policy.

    :param str policy: Route policy namespace such as ``"padding"`` or ``"local_bias"``.
    :param str seq_bucket: Sequence bucket returned by :func:`flash_seq_bucket`.
    :param tuple[int, int] | None compute_capability: Device capability, when known.
    :param str | None policy_path: Optional kernel-override table path.
    :return dict[str, Any] | None: Matching policy row, or None.
    """

    return _flash_route_policy(
        policy_path=_normalize_flash_kernel_policy_key(policy_path),
        policy=policy,
        seq_bucket=seq_bucket,
        compute_capability=compute_capability,
    )


def _route_policy_row_allows(
    row: dict[str, Any],
    *,
    seq_len: int | None = None,
    batch_size: int | None = None,
) -> bool:
    """Check a policy row's optional shape bounds against known batch facts.

    Rows may carry ``min_seq_len``/``max_seq_len``/``max_batch_size`` keys.
    A declared bound fails closed when the caller does not supply its runtime fact.

    :param dict[str, Any] row: Route-policy row from the tuning table.
    :param int | None seq_len: Padded sequence length, when known.
    :param int | None batch_size: Batch size, when known.
    :return bool: False when a row bound excludes the supplied facts.
    """

    min_seq = row.get("min_seq_len")
    max_seq = row.get("max_seq_len")
    if min_seq is not None or max_seq is not None:
        if seq_len is None:
            return False
        if min_seq is not None and seq_len < int(min_seq):
            return False
        if max_seq is not None and seq_len > int(max_seq):
            return False
    max_batch = row.get("max_batch_size")
    if max_batch is not None:
        if batch_size is None:
            return False
        if batch_size > int(max_batch):
            return False
    return True


def flash_route_choice(
    *,
    policy: str,
    seq_bucket: str,
    compute_capability: tuple[int, int] | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    policy_path: str | None = None,
) -> str | None:
    """Resolve a route choice from one model-scoped tuning table.

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
    :param str | None policy_path: Optional kernel-override table path.
    :return str | None: Route choice, or None when the table has no entry.
    """

    raw = flash_route_policy(
        policy=policy,
        seq_bucket=seq_bucket,
        compute_capability=compute_capability,
        policy_path=policy_path,
    )
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
    policy_path: str | None = None,
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
    :param str | None policy_path: Optional kernel-override table path.
    :return str: Either ``"fixed"`` or ``"varlen"``.
    """

    seq_bucket = flash_seq_bucket(
        seq_len=int(seq_len),
        total_tokens=total_tokens,
        batch_size=batch_size,
        policy_path=policy_path,
    )
    table_route = flash_route_choice(
        policy="padding",
        seq_bucket=seq_bucket,
        compute_capability=compute_capability,
        seq_len=int(seq_len),
        batch_size=int(batch_size) if batch_size is not None else None,
        policy_path=policy_path,
    )
    if table_route in {"fixed", "varlen"}:
        return table_route
    return "fixed"


def _entry_matches(
    context: FlashKernelContext,
    entry: dict[str, Any],
    *,
    policy_path: str,
) -> bool:
    """Return whether a table entry applies to one kernel context.

    :param FlashKernelContext context: Runtime kernel context.
    :param dict[str, Any] entry: Candidate JSON entry.
    :param str policy_path: Normalized override path, or ``""``.
    :return bool: True when the entry should be used.
    """

    cc_key = compute_capability_key(context.compute_capability)
    if str(entry.get("compute_capability", "*")).strip().lower() not in {cc_key, "*"}:
        return False
    if str(entry.get("route", "")).strip().lower() != str(context.route).strip().lower():
        return False
    if str(entry.get("kind", "")).strip().lower() != str(context.kind).strip().lower():
        return False
    seq_bucket = context.seq_bucket or flash_seq_bucket(
        seq_len=context.seq_len,
        total_tokens=context.total_tokens,
        batch_size=context.batch_size,
        policy_path=policy_path,
    )
    if str(entry.get("seq_bucket", "")).strip() != seq_bucket:
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
        if expected is not None:
            if value is None or int(expected) != int(value):
                return False
    range_int_fields = (
        ("batch_size", context.batch_size),
        ("query_len", context.query_len),
        ("key_len", context.key_len),
        ("num_heads", context.num_heads),
    )
    for key, value in range_int_fields:
        min_expected = entry.get(f"min_{key}")
        max_expected = entry.get(f"max_{key}")
        if min_expected is None and max_expected is None:
            continue
        if value is None:
            return False
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
        if expected is not None:
            if value is None or bool(expected) != bool(value):
                return False
    dtype = entry.get("dtype")
    if dtype is not None:
        if context.dtype is None or str(dtype).strip() != str(context.dtype).strip():
            return False
    att_span_min = entry.get("att_span_min")
    if att_span_min is not None:
        if context.att_span is None or int(context.att_span) < int(att_span_min):
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


def _resolve_flash_kernel_config(
    context: FlashKernelContext,
    *,
    policy_path: str,
) -> tuple[int, int, int, int] | None:
    """Resolve a measured kernel launch tuple from one model-scoped tuning table.

    An exact-capability row outranks a wildcard row; within the same
    specificity, later rows win so appended override-table rows take
    precedence, matching :func:`flash_route_policy` and the documented
    override workflow.

    :param FlashKernelContext context: Runtime kernel context.
    :param str policy_path: Normalized override path, or ``""``.
    :return tuple[int, int, int, int] | None: ``(BLOCK_M, BLOCK_N, stages, warps)`` or None.
    """

    kernels = _load_tuning_payload(policy_path).get("kernels", [])
    wildcard_match: dict[str, Any] | None = None
    for raw in reversed(kernels):
        if not _entry_matches(context, raw, policy_path=policy_path):
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


_resolve_flash_kernel_config = lru_cache(maxsize=_SHAPE_KEYED_CACHE_MAXSIZE)(_resolve_flash_kernel_config)


def resolve_flash_kernel_config(
    context: FlashKernelContext,
    *,
    policy_path: str | None = None,
) -> tuple[int, int, int, int] | None:
    """Resolve one launch tuple without process-global policy selection.

    :param FlashKernelContext context: Runtime kernel context.
    :param str | None policy_path: Optional kernel-override table path.
    :return tuple[int, int, int, int] | None: Matching launch tuple, or None.
    """

    return _resolve_flash_kernel_config(
        context,
        policy_path=_normalize_flash_kernel_policy_key(policy_path),
    )


resolve_flash_kernel_config.cache_clear = _resolve_flash_kernel_config.cache_clear  # type: ignore[attr-defined]
resolve_flash_kernel_config.cache_info = _resolve_flash_kernel_config.cache_info  # type: ignore[attr-defined]
