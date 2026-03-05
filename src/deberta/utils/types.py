"""Shared helpers for scalar parsing/coercion across CLI/config/training."""

from __future__ import annotations

import types
from typing import Any, Union, get_args, get_origin

TRUE_STRINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_STRINGS = frozenset({"0", "false", "f", "no", "n", "off"})
_NONE_STRINGS = frozenset({"none", "null"})


def parse_bool(value: Any, *, allow_numeric: bool = False) -> bool:
    """Parse bool-like values into ``bool``.

    :param Any value: Value to parse.
    :param bool allow_numeric: Whether numeric ``0``/``1`` inputs are accepted.
    :raises ValueError: If the value cannot be parsed as boolean.
    :return bool: Parsed boolean.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        text = str(value).strip().lower()
    else:
        if allow_numeric and isinstance(value, (int, float)) and value in {0, 1}:
            return bool(int(value))
        raise ValueError(f"Unsupported boolean value: {value!r}")

    if text in TRUE_STRINGS:
        return True
    if text in FALSE_STRINGS:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def unwrap_optional_type(field_type: Any) -> tuple[Any, bool]:
    """Unwrap ``Optional[T]``-style type annotations.

    :param Any field_type: Raw type annotation.
    :return tuple[Any, bool]: Base annotation type and whether ``None`` is allowed.
    """
    origin = get_origin(field_type)
    if origin in {Union, types.UnionType}:
        args = get_args(field_type)
        allows_none = any(a is type(None) for a in args)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0], allows_none
    return field_type, False


def coerce_scalar(
    raw: Any,
    target_type: Any,
    *,
    allow_none: bool = False,
    allow_bool_numeric: bool = False,
) -> Any:
    """Coerce a scalar value to a target annotation type.

    :param Any raw: Raw scalar value.
    :param Any target_type: Target annotation type.
    :param bool allow_none: Whether ``None``/``none``/``null`` are accepted.
    :param bool allow_bool_numeric: Whether bool coercion accepts numeric ``0``/``1``.
    :raises ValueError: If coercion fails for supported scalar target types.
    :return Any: Coerced value (or original value for unsupported target types).
    """
    target_t, optional_allows_none = unwrap_optional_type(target_type)
    allows_none = bool(allow_none or optional_allows_none)

    if raw is None:
        if allows_none:
            return None
        raise ValueError("None is not allowed for this field.")

    text = str(raw).strip() if isinstance(raw, str) else raw
    if isinstance(text, str) and allows_none and text.lower() in _NONE_STRINGS:
        return None

    if target_t is bool:
        bool_input = text if isinstance(text, str) else raw
        return parse_bool(bool_input, allow_numeric=allow_bool_numeric)

    if target_t is int:
        if isinstance(raw, bool):
            raise ValueError(f"Expected integer value, got bool: {raw!r}.")
        if isinstance(text, str):
            return int(text)
        return int(raw)

    if target_t is float:
        if isinstance(raw, bool):
            raise ValueError(f"Expected numeric value, got bool: {raw!r}.")
        if isinstance(text, str):
            return float(text)
        return float(raw)

    if target_t is str:
        if isinstance(text, str):
            return text
        return str(raw)

    return raw


__all__ = ["FALSE_STRINGS", "TRUE_STRINGS", "coerce_scalar", "parse_bool", "unwrap_optional_type"]
