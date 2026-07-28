from __future__ import annotations

import math
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from deberta.modeling.mask_utils import attention_mask_to_active_tokens

_TOOL = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "evaluate_rtd_checkpoint.py"))
_assemble_eval_batch = _TOOL["_assemble_eval_batch"]
_checkpoint_step = _TOOL["_checkpoint_step"]
_discriminator_metrics = _TOOL["_discriminator_metrics"]
_evaluate_checkpoint = _TOOL["_evaluate_checkpoint"]
_evaluation_provenance = _TOOL["_evaluation_provenance"]
_forward_discriminator_with_diagnostics = _TOOL["_forward_discriminator_with_diagnostics"]
_load_checkpoint_artifacts = _TOOL["_load_checkpoint_artifacts"]
_precision_mismatch_warning = _TOOL["_precision_mismatch_warning"]
_ranking_metrics = _TOOL["_ranking_metrics"]
_replacement_rate = _TOOL["_replacement_rate"]


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


def test_load_checkpoint_artifacts_uses_run_owned_tokenizer_and_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint-12000"
    checkpoint.mkdir(parents=True)
    tokenizer_dir = run_dir / "tokenizer"
    tokenizer_dir.mkdir()
    model_cfg = SimpleNamespace(tokenizer=SimpleNamespace(name_or_path="mutable-external-tokenizer"))
    cfg = SimpleNamespace(model=model_cfg)
    tokenizer = object()
    disc_cfg = object()
    gen_cfg = object()
    seen: dict[str, object] = {}

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source: Path, *, use_fast: bool) -> object:
            seen["tokenizer_source"] = source
            seen["use_fast"] = use_fast
            return tokenizer

    def _load_configs(*, run_dir: Path, model_cfg: object) -> tuple[object, object]:
        seen["config_run_dir"] = run_dir
        seen["model_cfg"] = model_cfg
        return disc_cfg, gen_cfg

    monkeypatch.setitem(_load_checkpoint_artifacts.__globals__, "AutoTokenizer", _AutoTokenizer)
    monkeypatch.setitem(
        _load_checkpoint_artifacts.__globals__,
        "load_materialized_backbone_configs",
        _load_configs,
    )

    loaded = _load_checkpoint_artifacts(cfg, checkpoint)

    assert loaded == (tokenizer, disc_cfg, gen_cfg)
    assert seen == {
        "tokenizer_source": tokenizer_dir,
        "use_fast": True,
        "config_run_dir": run_dir,
        "model_cfg": model_cfg,
    }


def test_replacement_rate_divides_by_masked_tokens_not_all_positions() -> None:
    # Padding-heavy (2, 4) grid: only one of two masked positions was actually
    # replaced by the generator; the other six entries are unmasked/padding
    # zeros. The old `.mean()` computation divides by all 8 grid positions.
    disc_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    masked_tokens = 2.0

    rate = _replacement_rate(disc_labels, masked_tokens)

    assert math.isclose(rate, 0.5)
    assert not math.isclose(rate, float(disc_labels.mean().item()))


def test_replacement_rate_clamps_zero_masked_tokens() -> None:
    disc_labels = torch.zeros((2, 4))

    assert _replacement_rate(disc_labels, 0.0) == 0.0


def test_precision_mismatch_warning_flags_disagreement() -> None:
    assert _precision_mismatch_warning("bf16", "bf16") is None
    assert _precision_mismatch_warning("fp32", "no") is None

    message = _precision_mismatch_warning("bf16", "no")
    assert message is not None
    assert "bf16" in message
    assert "no" in message

    message = _precision_mismatch_warning("fp32", "bf16")
    assert message is not None
    assert "fp32" in message
    assert "bf16" in message


def test_evaluation_provenance_reports_eager_attention_and_values() -> None:
    provenance = _evaluation_provenance(
        precision="fp32",
        sampling_temperature=0.85,
        configured_mixed_precision="no",
    )

    assert provenance == {
        "attention_impl": "eager",
        "flash_enabled": False,
        "precision": "fp32",
        "sampling_temperature": 0.85,
        "configured_mixed_precision": "no",
    }


def test_assemble_eval_batch_converts_doc_ids_to_doc_block_mask() -> None:
    rows = [
        {
            "input_ids": torch.tensor([[11, 12, 13, 14]]),
            "labels": torch.tensor([[-100, 12, -100, -100]]),
            "doc_ids": torch.tensor([[1, 1, 2, 0]]),
        },
        {
            "input_ids": torch.tensor([[21, 22, 23, 24]]),
            "labels": torch.tensor([[21, -100, -100, -100]]),
            "doc_ids": torch.tensor([[1, 1, 1, 1]]),
        },
    ]

    batch = _assemble_eval_batch(rows, backbone_type="hf_deberta_v2")

    # doc_ids are consumed into a pairwise mask, mirroring the training loop.
    assert "doc_ids" not in batch
    mask = batch["attention_mask"]
    assert mask.shape == (2, 4, 4)
    assert mask.dtype == torch.bool

    # Cross-document attention is blocked in both directions.
    assert mask[0, 0, 2].item() is False
    assert mask[0, 2, 0].item() is False

    # Padding stays inactive and loss accounting sees the doc_ids liveness.
    active = attention_mask_to_active_tokens(
        input_ids=batch["input_ids"],
        attention_mask=mask,
    )
    assert torch.equal(
        active,
        torch.tensor([[True, True, True, False], [True, True, True, True]]),
    )


def test_assemble_eval_batch_restores_dropped_all_ones_masks() -> None:
    rows = [
        {
            "input_ids": torch.tensor([[11, 12, 13]]),
            "labels": torch.tensor([[-100, 12, -100]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        },
        {
            # The collator drops all-ones masks, so this row arrives without one.
            "input_ids": torch.tensor([[21, 22, 23]]),
            "labels": torch.tensor([[21, -100, -100]]),
        },
    ]

    batch = _assemble_eval_batch(rows, backbone_type="hf_deberta_v2")

    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 0], [1, 1, 1]]))


def test_assemble_eval_batch_raises_on_partial_key_presence() -> None:
    # Same failure class fixed for doc_ids in commit 2319fe1: a key present on
    # only some rows must not be silently dropped from the evaluation batch.
    rows = [
        {
            "input_ids": torch.tensor([[11, 12, 13, 14]]),
            "labels": torch.tensor([[-100, 12, -100, -100]]),
            "doc_ids": torch.tensor([[1, 1, 2, 0]]),
        },
        {
            "input_ids": torch.tensor([[21, 22, 23, 24]]),
            "labels": torch.tensor([[21, -100, -100, -100]]),
            # doc_ids missing on this row.
        },
    ]

    with pytest.raises(ValueError, match="doc_ids"):
        _assemble_eval_batch(rows, backbone_type="hf_deberta_v2")


def test_evaluate_checkpoint_threads_sampling_temperature_and_fixes_replacement_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeGeneratorOutput:
        def __init__(
            self,
            corrupted_input_ids: torch.Tensor,
            disc_labels: torch.Tensor,
            gen_token_count: torch.Tensor,
            gen_loss_raw: torch.Tensor,
        ) -> None:
            self.corrupted_input_ids = corrupted_input_ids
            self.disc_labels = disc_labels
            self.gen_token_count = gen_token_count
            self.gen_loss_raw = gen_loss_raw

    class _Head(torch.nn.Module):
        def forward(
            self,
            hidden_states: torch.Tensor,
            *,
            attention_mask: torch.Tensor | None,
            doc_context_index: torch.Tensor | None,
        ) -> torch.Tensor:
            del attention_mask, doc_context_index
            return hidden_states.squeeze(-1)

    class _FakeModel(torch.nn.Module):
        def __init__(self, disc_labels: torch.Tensor) -> None:
            super().__init__()
            self.discriminator_head = _Head()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self._disc_labels = disc_labels
            self.seen_temperatures: list[float] = []

        def sync_discriminator_embeddings_from_generator(self) -> None:
            pass

        def forward_generator_phase(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
            labels: torch.Tensor,
            token_type_ids: torch.Tensor | None,
            position_ids: torch.Tensor | None,
            sampling_temperature: float,
        ) -> _FakeGeneratorOutput:
            del attention_mask, token_type_ids, position_ids
            self.seen_temperatures.append(sampling_temperature)
            gen_token_count = labels.ne(-100).sum().to(torch.float32)
            return _FakeGeneratorOutput(
                corrupted_input_ids=input_ids.clone(),
                disc_labels=self._disc_labels,
                gen_token_count=gen_token_count,
                gen_loss_raw=torch.tensor(0.4),
            )

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
            del disc_labels, token_type_ids, position_ids
            hidden_states = corrupted_input_ids.float().unsqueeze(-1)
            self.discriminator_head(
                hidden_states,
                attention_mask=attention_mask,
                doc_context_index=doc_context_index,
            )

    # (2, 4) grid: one masked position per row, but only one of the two
    # masked positions is actually marked "replaced". This mirrors the
    # regression above at the full `_evaluate_checkpoint` call site: the old
    # `.mean()` computation (1 / 8 = 0.125) differs from the fixed rate
    # (1 masked-position replacement / 2 masked tokens = 0.5).
    disc_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    model = _FakeModel(disc_labels)
    # `_evaluate_checkpoint` is wrapped by `@torch.no_grad()`, which does not
    # forward `__globals__`; reach the original function via `__wrapped__` so
    # the monkeypatched global is actually visible to its body.
    monkeypatch.setitem(
        _evaluate_checkpoint.__wrapped__.__globals__,
        "load_model_state_with_compile_key_remap",
        lambda _model, _checkpoint: {"missing": [], "unexpected": []},
    )

    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        "labels": torch.tensor([[-100, 11, -100, -100], [-100, -100, -100, 23]]),
    }

    result = _evaluate_checkpoint(
        model,
        batch=batch,
        checkpoint=Path("fake-checkpoint-1"),
        batch_size=2,
        precision="fp32",
        sampling_temperature=0.37,
        seed=0,
    )

    assert model.seen_temperatures == [0.37]
    assert math.isclose(result["generator"]["replacement_rate"], 0.5)
    assert result["generator"]["masked_tokens"] == 2
    assert result["generator"]["replacements"] == 1
