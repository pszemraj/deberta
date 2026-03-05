"""Shared serialization helpers for config/runtime payloads."""

import dataclasses
from dataclasses import fields
from typing import Any, get_type_hints

from deberta.utils.types import coerce_scalar, unwrap_optional_type


def asdict_without_private(value: Any) -> Any:
    """Convert nested dataclasses to dicts while skipping private fields.

    :param Any value: Dataclass or nested value.
    :return Any: Mapping/list/scalar payload.
    """
    if dataclasses.is_dataclass(value):
        out: dict[str, Any] = {}
        for f in fields(value):
            if str(f.name).startswith("_"):
                continue
            out[str(f.name)] = asdict_without_private(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {k: asdict_without_private(v) for k, v in value.items()}
    if isinstance(value, list):
        return [asdict_without_private(v) for v in value]
    if isinstance(value, tuple):
        return [asdict_without_private(v) for v in value]
    return value


def mapping_from_config_obj(config_obj: Any) -> dict[str, Any]:
    """Convert config-like objects to plain mappings.

    :param Any config_obj: Config object exposing ``to_dict``/``__dict__``.
    :return dict[str, Any]: Best-effort plain mapping.
    """
    if config_obj is None:
        return {}
    if isinstance(config_obj, dict):
        return dict(config_obj)
    to_dict_fn = getattr(config_obj, "to_dict", None)
    if callable(to_dict_fn):
        raw = to_dict_fn()
        if isinstance(raw, dict):
            return dict(raw)
    raw_dict = getattr(config_obj, "__dict__", None)
    if isinstance(raw_dict, dict):
        return dict(raw_dict)
    return {}


def coerce_dataclass_payload_types(cfg_obj: Any) -> dict[str, Any]:
    """Serialize dataclass payloads with best-effort scalar coercion.

    :param Any cfg_obj: Dataclass config object or mapping.
    :return dict[str, Any]: Serialized mapping with scalar coercion.
    """

    def _coerce_dataclass_instance(obj: Any) -> dict[str, Any]:
        """Recursively coerce nested dataclass payload fields.

        :param Any obj: Dataclass instance.
        :return dict[str, Any]: Serialized/coerced mapping.
        """
        payload: dict[str, Any] = {}
        type_hints = get_type_hints(type(obj))
        for f in fields(type(obj)):
            name = str(f.name)
            value = getattr(obj, name)
            target_t, _allows_none = unwrap_optional_type(type_hints.get(name, f.type))
            if dataclasses.is_dataclass(value):
                payload[name] = _coerce_dataclass_instance(value)
            else:
                try:
                    payload[name] = coerce_scalar(
                        value,
                        target_t,
                        allow_none=True,
                        allow_bool_numeric=True,
                    )
                except Exception:
                    payload[name] = value
        return payload

    if not dataclasses.is_dataclass(cfg_obj):
        if isinstance(cfg_obj, dict):
            return {str(key): value for key, value in cfg_obj.items()}
        return {}
    return _coerce_dataclass_instance(cfg_obj)


def drop_none_recursive(value: Any) -> Any:
    """Recursively drop ``None`` entries from mappings/lists.

    :param Any value: Arbitrary nested value.
    :return Any: Value with ``None`` keys/items removed.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            cleaned = drop_none_recursive(item)
            if cleaned is None:
                continue
            out[str(key)] = cleaned
        return out
    if isinstance(value, list):
        out_list: list[Any] = []
        for item in value:
            cleaned = drop_none_recursive(item)
            if cleaned is None:
                continue
            out_list.append(cleaned)
        return out_list
    if isinstance(value, tuple):
        out_tuple: list[Any] = []
        for item in value:
            cleaned = drop_none_recursive(item)
            if cleaned is None:
                continue
            out_tuple.append(cleaned)
        return out_tuple
    return value


__all__ = [
    "asdict_without_private",
    "coerce_dataclass_payload_types",
    "drop_none_recursive",
    "mapping_from_config_obj",
]
