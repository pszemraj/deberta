"""Materialized model/tokenizer artifacts owned by one training run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from deberta.config import ModelConfig
from deberta.modeling.deberta_v2_native import DebertaV2Config
from deberta.modeling.rope_encoder import DebertaRoPEConfig
from deberta.run_layout import (
    DISCRIMINATOR_CONFIG_FILENAME,
    GENERATOR_CONFIG_FILENAME,
    TOKENIZER_DIRNAME,
)
from deberta.utils.io import dump_json, load_json_mapping


def persist_materialized_run_artifacts(
    *,
    run_dir: Path,
    tokenizer: Any,
    discriminator_config: Any,
    generator_config: Any,
) -> None:
    """Persist the exact tokenizer and component configs used by one run.

    :param Path run_dir: Run directory that owns the artifacts.
    :param Any tokenizer: Materialized tokenizer exposing ``save_pretrained``.
    :param Any discriminator_config: Materialized discriminator config exposing ``to_dict``.
    :param Any generator_config: Materialized generator config exposing ``to_dict``.
    :return None: None.
    """

    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(discriminator_config.to_dict(), run_dir / DISCRIMINATOR_CONFIG_FILENAME)
    dump_json(generator_config.to_dict(), run_dir / GENERATOR_CONFIG_FILENAME)
    tokenizer.save_pretrained(str(run_dir / TOKENIZER_DIRNAME))


def load_materialized_backbone_configs(
    *,
    run_dir: Path,
    model_cfg: ModelConfig,
) -> tuple[Any, Any]:
    """Load the exact component configs persisted by one run.

    :param Path run_dir: Run directory that owns the materialized configs.
    :param ModelConfig model_cfg: High-level model config selecting the backbone family.
    :raises FileNotFoundError: If either materialized config is missing.
    :return tuple[Any, Any]: Discriminator and generator config objects.
    """

    discriminator_path = run_dir / DISCRIMINATOR_CONFIG_FILENAME
    generator_path = run_dir / GENERATOR_CONFIG_FILENAME
    for path in (discriminator_path, generator_path):
        if not path.is_file():
            raise FileNotFoundError(f"Run is missing required materialized backbone config: {path}")

    config_cls = (
        DebertaV2Config
        if str(model_cfg.backbone_type).strip().lower() == "hf_deberta_v2"
        else DebertaRoPEConfig
    )
    return (
        config_cls.from_dict(load_json_mapping(discriminator_path)),
        config_cls.from_dict(load_json_mapping(generator_path)),
    )


def materialized_tokenizer_path(run_dir: Path) -> Path:
    """Return the required run-owned tokenizer directory.

    :param Path run_dir: Run directory that owns the tokenizer.
    :raises FileNotFoundError: If the tokenizer directory is missing.
    :return Path: Tokenizer directory.
    """

    path = run_dir / TOKENIZER_DIRNAME
    if not path.is_dir():
        raise FileNotFoundError(f"Run is missing required materialized tokenizer directory: {path}")
    return path
