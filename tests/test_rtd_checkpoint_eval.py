from __future__ import annotations

import math
import runpy
from pathlib import Path

import torch

_TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "evaluate_rtd_checkpoint.py"))
_checkpoint_step = _TOOL["_checkpoint_step"]
_discriminator_metrics = _TOOL["_discriminator_metrics"]
_ranking_metrics = _TOOL["_ranking_metrics"]


def test_ranking_metrics_match_hand_calculation() -> None:
    logits = torch.tensor([0.9, 0.8, 0.7, 0.1])
    labels = torch.tensor([1, 0, 1, 0])

    roc_auc, average_precision = _ranking_metrics(logits, labels)

    assert math.isclose(roc_auc, 0.75)
    assert math.isclose(average_precision, (1.0 + 2.0 / 3.0) / 2.0)


def test_ranking_metrics_group_tied_scores() -> None:
    logits = torch.tensor([1.0, 1.0, 0.0, 0.0])
    labels = torch.tensor([1, 0, 1, 0])

    roc_auc, average_precision = _ranking_metrics(logits, labels)

    assert math.isclose(roc_auc, 0.5)
    assert math.isclose(average_precision, 0.5)


def test_discriminator_metrics_report_prior_gain() -> None:
    logits = torch.tensor([2.0, -2.0, 1.0, -1.0])
    labels = torch.tensor([1, 0, 1, 0])

    metrics = _discriminator_metrics(logits, labels)

    assert metrics["positive_rate"] == 0.5
    assert metrics["loss_gain"] > 0.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0


def test_checkpoint_step_uses_directory_suffix() -> None:
    assert _checkpoint_step(Path("run/checkpoint-12000")) == 12000
