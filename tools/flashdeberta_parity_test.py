#!/usr/bin/env python3
"""CUDA parity check for config-driven FlashDeBERTa attention routes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import _bench_common  # noqa: E402,F401  (inserts src/ on sys.path at import)
import torch

from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model  # noqa: E402
from deberta.modeling.mask_utils import (  # noqa: E402
    FlashBatchMeta,
    build_doc_block_mask,
    build_doc_segment_metadata,
    doc_segment_metadata_host_stats,
)


@dataclass(frozen=True)
class ParityCase:
    """One FlashDeBERTa parity scenario."""

    name: str
    seq_len: int
    batch_size: int
    route_hint: str
    pad_tail: int = 0
    docblock: bool = False
    strict_ratio: bool = False


def _build_tiny_config(*, seq_len: int, flash: bool) -> DebertaV2Config:
    """Build a small DeBERTa config suitable for parity testing."""

    cfg = DebertaV2Config(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        max_position_embeddings=seq_len,
        type_vocab_size=0,
        layer_norm_eps=1e-7,
        relative_attention=True,
        position_buckets=32,
        max_relative_positions=seq_len,
        pos_att_type=["c2p", "p2c"],
        pad_token_id=0,
        position_biased_input=False,
    )
    cfg.hf_attention_impl = "flash" if bool(flash) else "eager"
    cfg.hf_flash = {
        "force_varlen": False,
        "varlen_min_seq_len": None,
        "docblock_bias_seq_len": None,
        "eager_dense_max_seq_len": 0,
    }
    return cfg


@torch.no_grad()
def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy model weights between variants."""

    dst.load_state_dict(src.state_dict(), strict=True)


def _doc_ids_for_case(case: ParityCase) -> torch.Tensor:
    """Build packed document ids with at least three docs per active row.

    :param ParityCase case: Doc-block parity case.
    :return torch.Tensor: CPU doc ids in ``(B,S)`` layout.
    """

    if not case.docblock:
        raise ValueError("_doc_ids_for_case requires a doc-block case.")
    active_len = int(case.seq_len) - int(case.pad_tail)
    if active_len < 3:
        raise ValueError(f"Doc-block parity case needs at least three active tokens: {case}")
    first = max(1, active_len // 5)
    second = max(1, active_len // 3)
    third = active_len - first - second
    if third <= 0:
        third = 1
        second = max(1, active_len - first - third)
    lengths = (first, second, third)

    doc_ids = torch.zeros((case.batch_size, case.seq_len), dtype=torch.long)
    for row in range(case.batch_size):
        cursor = 0
        rotation = row % len(lengths)
        row_lengths = lengths[rotation:] + lengths[:rotation]
        for doc_idx, length in enumerate(row_lengths, start=1):
            next_cursor = min(active_len, cursor + int(length))
            doc_ids[row, cursor:next_cursor] = int(doc_idx)
            cursor = next_cursor
        if cursor < active_len:
            doc_ids[row, cursor:active_len] = len(row_lengths)
    return doc_ids


def _case_payload(case: ParityCase, *, cfg: DebertaV2Config, device: torch.device) -> dict[str, Any]:
    """Build inputs and flash metadata for one route case."""

    input_ids = torch.randint(5, cfg.vocab_size, (case.batch_size, case.seq_len), device=device)
    attention_mask: torch.Tensor | None = None
    seq_lengths: torch.Tensor | None = None
    doc_segment_offsets: torch.Tensor | None = None
    doc_segment_lengths: torch.Tensor | None = None
    doc_cu_seqlens: torch.Tensor | None = None
    active_tokens: int | None = None
    doc_num_segments: int | None = None
    doc_max_seqlen: int | None = None

    if case.docblock:
        doc_ids_cpu = _doc_ids_for_case(case)
        if case.pad_tail > 0:
            input_ids[:, -case.pad_tail :] = int(cfg.pad_token_id)
        doc_ids = doc_ids_cpu.to(device=device)
        seq_lengths = doc_ids.ne(0).sum(-1, dtype=torch.int32)
        if case.route_hint == "docblock_bias":
            attention_mask = build_doc_block_mask(doc_ids)
        else:
            attention_mask = doc_ids.ne(0)
            (
                doc_segment_offsets,
                doc_segment_lengths,
                doc_cu_seqlens,
                active_tokens,
            ) = build_doc_segment_metadata(doc_ids_cpu)
            doc_num_segments, doc_max_seqlen, _ = doc_segment_metadata_host_stats(
                doc_segment_lengths,
                active_tokens=active_tokens,
            )
            doc_segment_offsets = doc_segment_offsets.to(device=device)
            doc_segment_lengths = doc_segment_lengths.to(device=device)
            doc_cu_seqlens = doc_cu_seqlens.to(device=device)
    elif case.pad_tail > 0:
        attention_mask = torch.ones((case.batch_size, case.seq_len), device=device, dtype=torch.bool)
        attention_mask[1, -case.pad_tail :] = False
        input_ids[1, -case.pad_tail :] = int(cfg.pad_token_id)
        seq_lengths = attention_mask.sum(-1, dtype=torch.int32)

    flash_meta = FlashBatchMeta(
        seq_lengths=seq_lengths,
        doc_segment_offsets=doc_segment_offsets,
        doc_segment_lengths=doc_segment_lengths,
        doc_cu_seqlens=doc_cu_seqlens,
        active_tokens_host=active_tokens,
        doc_num_segments_host=doc_num_segments,
        doc_max_segment_length_host=doc_max_seqlen,
        route_hint=case.route_hint,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "flash_meta": flash_meta,
        "loss_mask": doc_ids_cpu.ne(0).to(device=device)
        if case.docblock
        else (
            attention_mask.bool()
            if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2
            else None
        ),
    }


def _run(model: DebertaV2Model, payload: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward/backward pass and return selected gradients."""

    model.zero_grad(set_to_none=True)
    model_payload = dict(payload)
    loss_mask = model_payload.pop("loss_mask", None)
    out = model(**model_payload).last_hidden_state
    if isinstance(loss_mask, torch.Tensor):
        loss = out.float()[loss_mask.bool()].pow(2).mean()
    else:
        loss = out.float().pow(2).mean()
    loss.backward()
    grads = {
        "word_embeddings": model.embeddings.word_embeddings.weight.grad,
        "rel_embeddings": model.encoder.rel_embeddings.weight.grad,
        "query": model.encoder.layer[0].attention.self.query_proj.weight.grad,
        "value": model.encoder.layer[0].attention.self.value_proj.weight.grad,
    }
    if any(value is None for value in grads.values()):
        missing = sorted(key for key, value in grads.items() if value is None)
        raise RuntimeError(f"Missing gradients for {missing}")
    return out.detach(), {key: value.detach().float().clone() for key, value in grads.items()}  # type: ignore[union-attr]


def _assert_close_to_reference(
    *,
    case_name: str,
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    max_abs_limit: float,
    mean_abs_limit: float,
) -> tuple[float, float]:
    """Assert one tensor is close enough to the fp32 eager reference.

    :return tuple[float, float]: Observed max and mean absolute errors.
    """

    diff = (actual.float() - reference.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    print(f"{case_name:14s} {label:18s} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e}")
    if max_abs > float(max_abs_limit) or mean_abs > float(mean_abs_limit):
        raise AssertionError(
            f"{case_name} {label} exceeded parity limits: "
            f"max_abs={max_abs:.4e} > {max_abs_limit:.4e} or "
            f"mean_abs={mean_abs:.4e} > {mean_abs_limit:.4e}"
        )
    return max_abs, mean_abs


def _assert_ratio_to_reference(
    *,
    case_name: str,
    label: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    eager_err: tuple[float, float],
) -> tuple[float, float]:
    """Assert flash error stays within the eager-bf16 error envelope.

    :param str case_name: Case name.
    :param str label: Tensor label.
    :param torch.Tensor actual: Flash tensor.
    :param torch.Tensor reference: fp32 eager reference tensor.
    :param tuple[float, float] eager_err: Eager ``(max_abs, mean_abs)`` errors.
    :return tuple[float, float]: Observed flash errors.
    """

    diff = (actual.float() - reference.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    max_limit = max(3.0 * float(eager_err[0]), 1e-7)
    mean_limit = max(3.0 * float(eager_err[1]), 1e-8)
    print(
        f"{case_name:14s} {label:18s} max_abs={max_abs:.4e} "
        f"mean_abs={mean_abs:.4e} limits=({max_limit:.4e},{mean_limit:.4e})"
    )
    if max_abs > max_limit or mean_abs > mean_limit:
        raise AssertionError(
            f"{case_name} {label} exceeded 3x eager-bf16 error: "
            f"max_abs={max_abs:.4e} > {max_limit:.4e} or "
            f"mean_abs={mean_abs:.4e} > {mean_limit:.4e}"
        )
    return max_abs, mean_abs


def _scaled_grad_limits(reference: torch.Tensor, *, max_rel: float, mean_rel: float) -> tuple[float, float]:
    """Return scale-aware absolute limits for a gradient tensor."""

    ref_f = reference.detach().float()
    max_scale = float(ref_f.abs().max().item())
    mean_scale = float(ref_f.abs().mean().item())
    return max(max_rel * max_scale, 5e-4), max(mean_rel * mean_scale, 5e-5)


def _run_case(case: ParityCase, *, device: torch.device) -> None:
    cfg_ref = _build_tiny_config(seq_len=case.seq_len, flash=False)
    cfg_flash = _build_tiny_config(seq_len=case.seq_len, flash=True)

    torch.manual_seed(0)
    ref = DebertaV2Model(cfg_ref).to(device=device, dtype=torch.float32).train()
    eager = DebertaV2Model(cfg_ref).to(device=device, dtype=torch.bfloat16).train()
    flash = DebertaV2Model(cfg_flash).to(device=device, dtype=torch.bfloat16).train()
    _copy_weights(ref, eager)
    _copy_weights(ref, flash)

    payload = _case_payload(case, cfg=cfg_ref, device=device)
    ref_payload = dict(payload)
    ref_payload.pop("flash_meta")
    if case.route_hint == "docblock":
        doc_ids = _doc_ids_for_case(case).to(device=device)
        ref_payload["attention_mask"] = build_doc_block_mask(doc_ids)

    ref_out, ref_grads = _run(ref, ref_payload)
    eager_out, eager_grads = _run(eager, ref_payload)
    flash_out, flash_grads = _run(flash, payload)
    compare_mask = payload.get("loss_mask")
    if isinstance(compare_mask, torch.Tensor):
        mask = compare_mask.bool()
        ref_out = ref_out[mask]
        eager_out = eager_out[mask]
        flash_out = flash_out[mask]

    eager_out_max, eager_out_mean = _assert_close_to_reference(
        case_name=case.name,
        label="eager_bf16_out",
        actual=eager_out,
        reference=ref_out,
        max_abs_limit=5e-2,
        mean_abs_limit=8e-3,
    )
    if case.strict_ratio:
        _assert_ratio_to_reference(
            case_name=case.name,
            label="flash_bf16_out",
            actual=flash_out,
            reference=ref_out,
            eager_err=(eager_out_max, eager_out_mean),
        )
    else:
        _assert_close_to_reference(
            case_name=case.name,
            label="flash_bf16_out",
            actual=flash_out,
            reference=ref_out,
            max_abs_limit=max(3.0 * eager_out_max, 7e-2),
            mean_abs_limit=max(3.0 * eager_out_mean, 1.2e-2),
        )
    for key in ("word_embeddings", "rel_embeddings", "query", "value"):
        eager_max_limit, eager_mean_limit = _scaled_grad_limits(
            ref_grads[key],
            max_rel=0.35,
            mean_rel=0.35,
        )
        eager_grad_max, eager_grad_mean = _assert_close_to_reference(
            case_name=case.name,
            label=f"eager_grad_{key}",
            actual=eager_grads[key],
            reference=ref_grads[key],
            max_abs_limit=eager_max_limit,
            mean_abs_limit=eager_mean_limit,
        )
        if case.strict_ratio:
            _assert_ratio_to_reference(
                case_name=case.name,
                label=f"flash_grad_{key}",
                actual=flash_grads[key],
                reference=ref_grads[key],
                eager_err=(eager_grad_max, eager_grad_mean),
            )
        else:
            flash_max_limit, flash_mean_limit = _scaled_grad_limits(
                ref_grads[key],
                max_rel=0.5,
                mean_rel=0.5,
            )
            _assert_close_to_reference(
                case_name=case.name,
                label=f"flash_grad_{key}",
                actual=flash_grads[key],
                reference=ref_grads[key],
                max_abs_limit=max(3.0 * eager_grad_max, flash_max_limit),
                mean_abs_limit=max(3.0 * eager_grad_mean, flash_mean_limit),
            )


def main() -> None:
    """Run forward/backward parity checks on a CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tools/flashdeberta_parity_test.py.")

    device = torch.device("cuda")
    cases = [
        ParityCase("dense", seq_len=256, batch_size=2, route_hint="dense"),
        ParityCase("fixed_padded", seq_len=256, batch_size=2, route_hint="fixed", pad_tail=64),
        ParityCase("varlen", seq_len=256, batch_size=2, route_hint="varlen", pad_tail=64),
        ParityCase("local_bias", seq_len=1024, batch_size=2, route_hint="dense"),
        ParityCase("docblock", seq_len=256, batch_size=2, route_hint="docblock", pad_tail=32, docblock=True),
        ParityCase(
            "docblock_1024",
            seq_len=1024,
            batch_size=1,
            route_hint="docblock",
            pad_tail=96,
            docblock=True,
            strict_ratio=True,
        ),
        ParityCase(
            "docblock_2048",
            seq_len=2048,
            batch_size=1,
            route_hint="docblock",
            pad_tail=160,
            docblock=True,
            strict_ratio=True,
        ),
        ParityCase(
            "docblock_4096",
            seq_len=4096,
            batch_size=1,
            route_hint="docblock",
            pad_tail=256,
            docblock=True,
            strict_ratio=True,
        ),
    ]
    include_docblock_bias = str(
        os.environ.get("FLASHDEBERTA_SKIP_DOCBLOCK_BIAS", "")
    ).strip().lower() not in {
        "1",
        "true",
        "yes",
    }
    if include_docblock_bias:
        cases.append(
            ParityCase("docblock_bias", seq_len=1024, batch_size=2, route_hint="docblock_bias", docblock=True)
        )
        cases.extend(
            [
                ParityCase(
                    "docbias_1024_b4",
                    seq_len=1024,
                    batch_size=4,
                    route_hint="docblock_bias",
                    docblock=True,
                    strict_ratio=True,
                ),
                ParityCase(
                    "docbias_1024",
                    seq_len=1024,
                    batch_size=1,
                    route_hint="docblock_bias",
                    pad_tail=96,
                    docblock=True,
                    strict_ratio=True,
                ),
                ParityCase(
                    "docbias_2048",
                    seq_len=2048,
                    batch_size=1,
                    route_hint="docblock_bias",
                    pad_tail=160,
                    docblock=True,
                    strict_ratio=True,
                ),
                ParityCase(
                    "docbias_4096",
                    seq_len=4096,
                    batch_size=1,
                    route_hint="docblock_bias",
                    pad_tail=256,
                    docblock=True,
                    strict_ratio=True,
                ),
            ]
        )
    requested_cases = {
        item.strip()
        for item in str(os.environ.get("FLASHDEBERTA_PARITY_CASES", "")).split(",")
        if item.strip()
    }
    if requested_cases:
        cases = [case for case in cases if case.name in requested_cases]
        missing = requested_cases.difference(case.name for case in cases)
        if missing:
            raise ValueError(f"Unknown parity cases requested: {sorted(missing)}")
    for case in cases:
        _run_case(case, device=device)
    print("OK")


if __name__ == "__main__":
    main()
