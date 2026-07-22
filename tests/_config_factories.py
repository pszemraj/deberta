"""Concise config builders for behavior-focused tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

from deberta.config import DataConfig, LoggingConfig, ModelConfig, OptimConfig, TrainConfig

T = TypeVar("T")


def _replace_path(config: T, path: str, value: Any) -> T:
    head, *tail = path.split(".", 1)
    if not tail:
        return replace(config, **{head: value})
    child = getattr(config, head)
    return replace(config, **{head: _replace_path(child, tail[0], value)})


def _build(config: T, overrides: dict[str, Any]) -> T:
    valid = {item.name for item in fields(config)}
    for key, value in overrides.items():
        if key in valid and isinstance(value, dict) and is_dataclass(getattr(config, key)):
            config = replace(config, **{key: _build(getattr(config, key), value)})
        else:
            config = _replace_path(config, key, value)
    return config


def make_model_config(**overrides: Any) -> ModelConfig:
    return _build(ModelConfig(), overrides)


def make_data_config(**overrides: Any) -> DataConfig:
    return _build(DataConfig(), overrides)


def make_train_config(**overrides: Any) -> TrainConfig:
    return _build(TrainConfig(), overrides)


def make_optim_config(**overrides: Any) -> OptimConfig:
    return _build(OptimConfig(), overrides)


def make_logging_config(**overrides: Any) -> LoggingConfig:
    return _build(LoggingConfig(), overrides)
