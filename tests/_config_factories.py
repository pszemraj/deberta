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


def make_native_deberta_config(*, flash: bool = False, **overrides: Any) -> Any:
    """Build a tiny native DeBERTa config while keeping test-specific behavior explicit."""

    from deberta.modeling.deberta_v2_native import DebertaV2Config

    defaults = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
        "max_position_embeddings": 16,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
    }
    config = DebertaV2Config(**(defaults | overrides))
    if flash:
        config.hf_flash = {}
    return config
