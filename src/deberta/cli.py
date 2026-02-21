from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

from deberta.config import DataConfig, ModelConfig, TrainConfig
from deberta.training import run_pretraining


def _split_flat_dict(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split a flat dict into model/data/train sub-dicts based on dataclass field names."""

    model_keys = {f.name for f in fields(ModelConfig)}
    data_keys = {f.name for f in fields(DataConfig)}
    train_keys = {f.name for f in fields(TrainConfig)}

    model_dict: dict[str, Any] = {}
    data_dict: dict[str, Any] = {}
    train_dict: dict[str, Any] = {}

    unknown: dict[str, Any] = {}

    for k, v in raw.items():
        if k in model_keys:
            model_dict[k] = v
        elif k in data_keys:
            data_dict[k] = v
        elif k in train_keys:
            train_dict[k] = v
        else:
            unknown[k] = v

    if unknown:
        keys = ", ".join(sorted(unknown.keys()))
        raise ValueError(
            "Unknown keys in config file (not in ModelConfig/DataConfig/TrainConfig): "
            f"{keys}"
        )

    return model_dict, data_dict, train_dict


def _load_yaml(path: Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyyaml is required for YAML config files. Install with `pip install pyyaml`.") from e

    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("YAML config must parse to a dict.")

    if any(k in raw for k in ("model", "data", "train")):
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if not isinstance(model_dict, dict) or not isinstance(data_dict, dict) or not isinstance(train_dict, dict):
            raise ValueError("YAML config sections model/data/train must be dicts.")
    else:
        model_dict, data_dict, train_dict = _split_flat_dict(raw)

    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


def _load_json(path: Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    raw = json.loads(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("JSON config must parse to a dict.")

    # Support either nested {model:..., data:..., train:...} or flat.
    if any(k in raw for k in ("model", "data", "train")):
        model_dict = raw.get("model", {}) or {}
        data_dict = raw.get("data", {}) or {}
        train_dict = raw.get("train", {}) or {}
        if not isinstance(model_dict, dict) or not isinstance(data_dict, dict) or not isinstance(train_dict, dict):
            raise ValueError("JSON config sections model/data/train must be dicts.")
        return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)

    # Back-compat: try HF's flat json format by splitting keys.
    model_dict, data_dict, train_dict = _split_flat_dict(raw)
    return ModelConfig(**model_dict), DataConfig(**data_dict), TrainConfig(**train_dict)


def main() -> None:
    """Entry point for `deberta-pretrain`.

    Supported invocation styles:

      1) CLI flags (HfArgumentParser):
         deberta-pretrain --dataset_name c4 --streaming true ...

      2) Single config file arg (.yaml/.yml/.json):
         deberta-pretrain configs/pretrain_rope.yaml

    Note: Config-file mode is intentionally simple: it does not merge CLI overrides.
    """

    # Config-file shortcut mode.
    if len(sys.argv) == 2 and Path(sys.argv[1]).suffix.lower() in {".json", ".yaml", ".yml"}:
        cfg_path = Path(sys.argv[1]).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(str(cfg_path))

        if cfg_path.suffix.lower() == ".json":
            model_cfg, data_cfg, train_cfg = _load_json(cfg_path)
        else:
            model_cfg, data_cfg, train_cfg = _load_yaml(cfg_path)

        run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)
        return

    # Default: parse from CLI flags.
    try:
        from transformers import HfArgumentParser
    except Exception as e:  # pragma: no cover
        raise RuntimeError("transformers is required for CLI parsing.") from e

    parser = HfArgumentParser((ModelConfig, DataConfig, TrainConfig))
    model_cfg, data_cfg, train_cfg = parser.parse_args_into_dataclasses()

    run_pretraining(model_cfg=model_cfg, data_cfg=data_cfg, train_cfg=train_cfg)


if __name__ == "__main__":
    main()
