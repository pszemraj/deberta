"""Shared accessors for materialized FlashDeBERTa configuration values."""

from __future__ import annotations

from typing import Any

_TRUTHY = {"1", "true", "yes", "y", "on"}


def flash_cfg_get(flash_cfg: Any | None, name: str, default: Any) -> Any:
    """Return one flash config value from a mapping, dataclass, or object.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Field name.
    :param Any default: Default value.
    :return Any: Resolved value.
    """

    if flash_cfg is None:
        return default
    if isinstance(flash_cfg, dict):
        return flash_cfg.get(name, default)
    return getattr(flash_cfg, name, default)


def flash_cfg_bool(flash_cfg: Any | None, *, name: str, default: str) -> bool:
    """Resolve one boolean flash option from config or a declared default.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Config field name.
    :param str default: Default text when config is absent.
    :return bool: Resolved boolean value.
    """

    if flash_cfg is None:
        return str(default).strip().lower() in _TRUTHY
    return bool(flash_cfg_get(flash_cfg, name, False))


def flash_cfg_optional_int(
    flash_cfg: Any | None,
    *,
    name: str,
    default: int | None = None,
) -> int | None:
    """Resolve one optional integer flash option.

    :param Any | None flash_cfg: Optional config source.
    :param str name: Config field name.
    :param int | None default: Default value when config is absent.
    :return int | None: Resolved integer, or None when unset.
    """

    value = flash_cfg_get(flash_cfg, name, default)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return default


__all__ = ["flash_cfg_bool", "flash_cfg_get", "flash_cfg_optional_int"]
