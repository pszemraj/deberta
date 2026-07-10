from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from transformers.activations import ACT2FN

from deberta import config as cfg_mod
from deberta.config import (
    Config,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    iter_leaf_paths_for_dataclass,
    load_config,
    validate_model_config,
    validate_optim_config,
    validate_train_config,
)
from deberta.utils.mapping import flatten_mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = REPO_ROOT / "configs" / "config-reference.yaml"
MARKER_RE = re.compile(r"^\s*# \[([a-z0-9_.]+)\]\s*$")
LABEL_RE = re.compile(r"^\s*# (type|schema default|default|required|values|aliases|contract):\s*(.*)$")

CLOSED_SETS: dict[str, set[str]] = {
    "model.backbone_type": cfg_mod._BACKBONE_CHOICES,
    "model.embedding_sharing": cfg_mod._EMBED_SHARING_CHOICES,
    "model.hf.model_size": cfg_mod._HF_MODEL_SIZE_CHOICES,
    "model.hf.attention_kernel": cfg_mod._HF_ATTN_KERNEL_CHOICES,
    "model.hf.attention_impl": cfg_mod._HF_ATTN_IMPL_CHOICES,
    "model.rope.hidden_act": set(ACT2FN),
    "model.rope.norm_arch": cfg_mod._NORM_ARCH_CHOICES,
    "model.rope.attention_implementation": cfg_mod._ATTN_IMPL_CHOICES,
    "model.rope.ffn_type": cfg_mod._FFN_CHOICES,
    "model.rope.pretrained.norm_arch": cfg_mod._NORM_ARCH_CHOICES,
    "model.rope.pretrained.ffn_type": cfg_mod._FFN_CHOICES,
    "train.mixed_precision": cfg_mod._MIXED_PRECISION_CANONICAL,
    "train.sdpa_kernel": cfg_mod._SDPA_KERNEL_CHOICES,
    "train.compile.mode": cfg_mod._TORCH_COMPILE_MODE_CHOICES,
    "train.compile.scope": cfg_mod._TORCH_COMPILE_SCOPE_CHOICES,
    "train.compile.backend": cfg_mod._TORCH_COMPILE_BACKEND_CHOICES,
    "train.checkpoint.resume_data_strategy": cfg_mod._RESUME_DATA_STRATEGY_CHOICES,
    "optim.scheduler.type": cfg_mod._LR_SCHEDULER_CHOICES,
    "logging.backend": cfg_mod._LOGGING_BACKEND_CHOICES,
    "logging.wandb.watch": cfg_mod._WANDB_WATCH_CHOICES,
}

ALIASES: dict[str, dict[str, str]] = {
    "model.hf.attention_kernel": cfg_mod._HF_ATTN_KERNEL_ALIASES,
    "train.mixed_precision": cfg_mod._MIXED_PRECISION_ALIASES,
    "train.sdpa_kernel": cfg_mod._SDPA_KERNEL_ALIASES,
    "train.compile.mode": cfg_mod._TORCH_COMPILE_MODE_ALIASES,
    "train.compile.scope": cfg_mod._TORCH_COMPILE_SCOPE_ALIASES,
    "train.compile.backend": cfg_mod._TORCH_COMPILE_BACKEND_ALIASES,
    "logging.wandb.watch": cfg_mod._WANDB_WATCH_ALIASES,
}


def _type_name(tp: Any) -> str:
    """Render the config annotation spelling used by the manual reference."""

    text = str(tp).replace("<class '", "").replace("'>", "")
    return text.replace("types.NoneType", "None").replace("NoneType", "None")


def _reference_blocks(text: str) -> dict[str, dict[str, str]]:
    """Parse labeled comment blocks keyed by dotted config path."""

    blocks: dict[str, dict[str, str]] = {}
    current_path: str | None = None
    current_label: str | None = None
    for line in text.splitlines():
        marker = MARKER_RE.match(line)
        if marker:
            current_path = marker.group(1)
            assert current_path not in blocks, f"duplicate reference block: {current_path}"
            blocks[current_path] = {}
            current_label = None
            continue
        if current_path is None:
            continue
        label = LABEL_RE.match(line)
        if label:
            current_label = label.group(1)
            blocks[current_path][current_label] = label.group(2).strip()
            continue
        stripped = line.strip()
        if current_label is not None and stripped.startswith("#   "):
            blocks[current_path][current_label] += " " + stripped[4:].strip()
    return blocks


def _assert_schema_default(raw: str, expected: Any, *, path: str) -> None:
    """Compare one hand-written schema-default marker with the dataclass value."""

    if isinstance(expected, float):
        assert float(raw) == pytest.approx(expected), path
        return
    assert yaml.safe_load(raw) == expected, path


def test_manual_reference_covers_schema_and_metadata() -> None:
    """Require one complete manual contract block for every accepted leaf."""

    text = REFERENCE_PATH.read_text(encoding="utf-8")
    blocks = _reference_blocks(text)
    schema = dict(iter_leaf_paths_for_dataclass(Config))
    assert set(blocks) == set(schema)

    defaults = flatten_mapping(asdict(Config()))
    required_labels = {"type", "schema default", "default", "required", "values", "contract"}
    for path, field_type in schema.items():
        block = blocks[path]
        assert required_labels <= set(block), path
        assert block["type"] == _type_name(field_type), path
        _assert_schema_default(block["schema default"], defaults[path], path=path)
        assert block["default"], path
        assert block["required"], path
        assert block["values"], path
        assert block["contract"], path


def test_manual_reference_enumerates_every_closed_set() -> None:
    """Keep finite value lists synchronized with executable validator constants."""

    blocks = _reference_blocks(REFERENCE_PATH.read_text(encoding="utf-8"))
    for path, expected in CLOSED_SETS.items():
        documented = set(re.findall(r'"([^"]+)"', blocks[path]["values"]))
        assert documented == set(expected), path


def test_manual_reference_lists_every_nontrivial_alias() -> None:
    """Keep normalization aliases visible on the fields that accept them."""

    blocks = _reference_blocks(REFERENCE_PATH.read_text(encoding="utf-8"))
    for path, aliases in ALIASES.items():
        documented = blocks[path].get("aliases", "")
        for alias, canonical in aliases.items():
            if alias == canonical:
                continue
            assert alias in documented, (path, alias)
            assert canonical in documented, (path, canonical)


def test_manual_reference_yaml_has_exact_runtime_inventory() -> None:
    """Ensure the runnable YAML contains every accepted leaf and no undocumented runtime leaf."""

    raw = yaml.safe_load(REFERENCE_PATH.read_text(encoding="utf-8"))
    runtime = {key: raw[key] for key in ("model", "data", "train", "optim", "logging")}
    assert set(flatten_mapping(runtime)) == {
        path for path, _field_type in iter_leaf_paths_for_dataclass(Config)
    }


def test_manual_reference_is_a_valid_config() -> None:
    """Keep the hand-written reference directly runnable without network access during validation."""

    cfg = load_config(REFERENCE_PATH)
    assert cfg.model.backbone_type == "hf_deberta_v2"
    assert cfg.data.source.dataset_name == "HuggingFaceFW/fineweb-edu"
    assert cfg.train.objective.mask_token_prob == pytest.approx(1.0)
    assert cfg.train.objective.random_token_prob == pytest.approx(0.0)
    assert cfg.train.objective.disc_loss_weight == pytest.approx(10.0)
    assert cfg.optim.adam.epsilon == pytest.approx(1e-6)
    assert cfg.optim.scheduler.warmup_steps == 10_000


def test_config_reference_has_no_documentation_generator() -> None:
    """Prevent reintroducing generated prose for the manual YAML reference."""

    assert not (REPO_ROOT / "tools" / "generate_config_reference.py").exists()


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (ModelConfig(generator={"hidden_size": 0}), "model.generator.hidden_size"),
        (ModelConfig(rope={"rope_theta": 0.0}), "model.rope.rope_theta"),
        (ModelConfig(dropout={"hidden_prob": 1.1}), "model.dropout.hidden_prob"),
        (
            ModelConfig(
                hf={"attention_impl": "flash"},
                dropout={"hidden_prob": None, "attention_probs_prob": 0.0},
            ),
            "explicitly set to 0.0",
        ),
    ],
)
def test_documented_model_constraints_fail_during_config_validation(cfg: ModelConfig, match: str) -> None:
    """Keep documented model boundaries in the early validation layer."""

    with pytest.raises(ValueError, match=match):
        validate_model_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (TrainConfig(objective={"gen_loss_weight": -1.0}), "loss weights"),
        (
            TrainConfig(objective={"gen_loss_weight": 0.0, "disc_loss_weight": 0.0}),
            "At least one",
        ),
    ],
)
def test_documented_train_constraints_fail_during_config_validation(cfg: TrainConfig, match: str) -> None:
    """Keep documented objective boundaries in the early validation layer."""

    with pytest.raises(ValueError, match=match):
        validate_train_config(cfg)


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (OptimConfig(adam={"beta1": 1.0}), "optim.adam.beta1"),
        (OptimConfig(adam={"epsilon": 0.0}), "optim.adam.epsilon"),
    ],
)
def test_documented_optim_constraints_fail_during_config_validation(cfg: OptimConfig, match: str) -> None:
    """Keep documented Adam boundaries in the early validation layer."""

    with pytest.raises(ValueError, match=match):
        validate_optim_config(cfg)
