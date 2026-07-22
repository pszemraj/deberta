from __future__ import annotations

import math
import runpy
from pathlib import Path

import torch

_TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "evaluate_rtd_checkpoint.py"))
_checkpoint_step = _TOOL["_checkpoint_step"]
_discriminator_metrics = _TOOL["_discriminator_metrics"]
_forward_discriminator_with_diagnostics = _TOOL["_forward_discriminator_with_diagnostics"]
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


def test_discriminator_metrics_define_zero_recall_without_positive_targets() -> None:
    metrics = _discriminator_metrics(
        torch.tensor([-2.0, -1.0, 1.0]),
        torch.zeros(3, dtype=torch.long),
    )

    assert metrics["positives"] == 0
    assert metrics["recall_at_zero"] == 0.0
    assert metrics["roc_auc"] is None
    assert metrics["average_precision"] is None


def test_ranking_metrics_mark_roc_auc_undefined_without_negative_targets() -> None:
    roc_auc, average_precision = _ranking_metrics(
        torch.tensor([2.0, 1.0, -1.0]),
        torch.ones(3, dtype=torch.long),
    )

    assert roc_auc is None
    assert average_precision == 1.0


def test_discriminator_diagnostics_use_shared_phase_path() -> None:
    class _Head(torch.nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None,
            doc_context_index: torch.Tensor | None,
        ) -> torch.Tensor:
            del attention_mask, doc_context_index
            return hidden_states.squeeze(-1) + 1.0

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.discriminator_head = _Head()
            self.phase_calls = 0

        def forward_discriminator_phase(
            self,
            *,
            input_ids: torch.Tensor,
            corrupted_input_ids: torch.Tensor,
            disc_labels: torch.Tensor,
            attention_mask: torch.Tensor | None,
            token_type_ids: torch.Tensor | None,
            position_ids: torch.Tensor | None,
            doc_context_index: torch.Tensor | None,
        ) -> None:
            del input_ids, disc_labels, token_type_ids, position_ids
            self.phase_calls += 1
            hidden_states = corrupted_input_ids.float().unsqueeze(-1)
            self.discriminator_head(
                hidden_states,
                attention_mask=attention_mask,
                doc_context_index=doc_context_index,
            )

    model = _Model()
    corrupted_input_ids = torch.tensor([[3, 5, 7]])
    hidden_states, logits = _forward_discriminator_with_diagnostics(
        model,
        input_ids=corrupted_input_ids,
        corrupted_input_ids=corrupted_input_ids,
        disc_labels=torch.zeros_like(corrupted_input_ids, dtype=torch.float32),
        attention_mask=torch.ones_like(corrupted_input_ids),
        token_type_ids=None,
        position_ids=None,
        doc_context_index=None,
        precision="fp32",
    )

    assert model.phase_calls == 1
    torch.testing.assert_close(hidden_states, corrupted_input_ids.float().unsqueeze(-1))
    torch.testing.assert_close(logits, corrupted_input_ids.float() + 1.0)


def test_checkpoint_step_uses_directory_suffix() -> None:
    assert _checkpoint_step(Path("run/checkpoint-12000")) == 12000
