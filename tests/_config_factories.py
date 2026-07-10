"""Concise config builders for behavior-focused tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, TypeVar

from deberta.config import DataConfig, LoggingConfig, ModelConfig, OptimConfig, TrainConfig
from deberta.utils.mapping import flatten_mapping

T = TypeVar("T")

_OPTIM_PATHS = {
    "learning_rate": "lr.base",
    "generator_learning_rate": "lr.generator",
    "discriminator_learning_rate": "lr.discriminator",
    "adam_beta1": "adam.beta1",
    "adam_beta2": "adam.beta2",
    "adam_epsilon": "adam.epsilon",
    "lr_scheduler_type": "scheduler.type",
    "warmup_steps": "scheduler.warmup_steps",
}
_LOGGING_PATHS = {
    "wandb_watch": "wandb.watch",
    "wandb_watch_log_freq": "wandb.watch_log_freq",
    "debug_metrics": "debug.metrics",
}


def _replace_path(config: T, path: str, value: Any) -> T:
    head, *tail = path.split(".", 1)
    if not tail:
        return replace(config, **{head: value})
    child = getattr(config, head)
    return replace(config, **{head: _replace_path(child, tail[0], value)})


def _build(config: T, overrides: dict[str, Any], moved: dict[str, str] | None = None) -> T:
    valid = {item.name for item in fields(config)}
    paths: dict[str, Any] = {}
    for key, value in overrides.items():
        path = (moved or {}).get(key, key)
        if path in valid and isinstance(value, dict) and is_dataclass(getattr(config, path)):
            paths.update({f"{path}.{child}": item for child, item in flatten_mapping(value).items()})
        else:
            paths[path] = value
    for path, value in paths.items():
        config = _replace_path(config, path, value)
    return config


def make_model_config(**overrides: Any) -> ModelConfig:
    return _build(ModelConfig(), overrides, ModelConfig._MOVED_KEYS)


def make_data_config(**overrides: Any) -> DataConfig:
    return _build(DataConfig(), overrides, DataConfig._MOVED_KEYS)


def make_train_config(**overrides: Any) -> TrainConfig:
    return _build(TrainConfig(), overrides, TrainConfig._MOVED_KEYS)


def make_optim_config(**overrides: Any) -> OptimConfig:
    return _build(OptimConfig(), overrides, _OPTIM_PATHS)


def make_logging_config(**overrides: Any) -> LoggingConfig:
    report_to = overrides.pop("report_to", None)
    config = _build(LoggingConfig(), overrides, _LOGGING_PATHS)
    if report_to is None:
        return config
    target = str(report_to).lower()
    config = _replace_path(config, "wandb.enabled", target == "wandb")
    return replace(config, backend="none" if target == "wandb" else target)
