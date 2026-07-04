#!/usr/bin/env python3
"""CUDA parity check for config-driven FlashDeBERTa attention routes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


def _ensure_src_on_path() -> None:
    """Add the repository ``src/`` directory to ``sys.path`` for direct script execution."""

    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()

from deberta.modeling.deberta_v2_native import DebertaV2Config, DebertaV2Model  # noqa: E402
from deberta.modeling.mask_utils import build_doc_block_mask, build_doc_segment_metadata  # noqa: E402


@dataclass(frozen=True)
class ParityCase:
    """One FlashDeBERTa parity scenario."""

    name: str
    seq_len: int
    batch_size: int
    route_hint: str
    pad_tail: int = 0
    docblock: bool = False


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
        "varlen_min_seq_len": 2048,
        "docblock_bias_seq_len": 1024,
        "eager_dense_max_seq_len": 0,
    }
    return cfg


@torch.no_grad()
def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy model weights between variants."""

    dst.load_state_dict(src.state_dict(), strict=True)


def _case_payload(
    case: ParityCase, *, cfg: DebertaV2Config, device: torch.device
) -> dict[str, torch.Tensor | str | None]:
    """Build inputs and flash metadata for one route case."""

    input_ids = torch.randint(5, cfg.vocab_size, (case.batch_size, case.seq_len), device=device)
    attention_mask: torch.Tensor | None = None
    flash_seq_lengths: torch.Tensor | None = None
    flash_doc_segment_offsets: torch.Tensor | None = None
    flash_doc_segment_lengths: torch.Tensor | None = None
    flash_doc_cu_seqlens: torch.Tensor | None = None

    if case.docblock:
        doc_ids = torch.zeros((case.batch_size, case.seq_len), device=device, dtype=torch.long)
        split = max(2, case.seq_len // 2)
        doc_ids[:, :split] = 1
        doc_ids[:, split : case.seq_len - case.pad_tail] = 2
        if case.pad_tail > 0:
            input_ids[:, -case.pad_tail :] = int(cfg.pad_token_id)
        flash_seq_lengths = doc_ids.ne(0).sum(-1, dtype=torch.int32)
        if case.route_hint == "docblock_bias":
            attention_mask = build_doc_block_mask(doc_ids)
        else:
            attention_mask = doc_ids.ne(0)
            (
                flash_doc_segment_offsets,
                flash_doc_segment_lengths,
                flash_doc_cu_seqlens,
                _active_tokens,
            ) = build_doc_segment_metadata(doc_ids)
    elif case.pad_tail > 0:
        attention_mask = torch.ones((case.batch_size, case.seq_len), device=device, dtype=torch.bool)
        attention_mask[1, -case.pad_tail :] = False
        input_ids[1, -case.pad_tail :] = int(cfg.pad_token_id)
        flash_seq_lengths = attention_mask.sum(-1, dtype=torch.int32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "flash_seq_lengths": flash_seq_lengths,
        "flash_doc_segment_offsets": flash_doc_segment_offsets,
        "flash_doc_segment_lengths": flash_doc_segment_lengths,
        "flash_doc_cu_seqlens": flash_doc_cu_seqlens,
        "flash_route_hint": case.route_hint,
    }


def _run(
    model: DebertaV2Model,
    payload: dict[str, torch.Tensor | str | None],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward/backward pass and return selected gradients."""

    model.zero_grad(set_to_none=True)
    out = model(**payload).last_hidden_state
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
) -> None:
    """Assert one tensor is close enough to the fp32 eager reference."""

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
    ref_payload.pop("flash_seq_lengths")
    ref_payload.pop("flash_doc_segment_offsets")
    ref_payload.pop("flash_doc_segment_lengths")
    ref_payload.pop("flash_doc_cu_seqlens")
    ref_payload.pop("flash_route_hint")
    if case.route_hint == "docblock":
        doc_ids = torch.zeros((case.batch_size, case.seq_len), device=device, dtype=torch.long)
        split = max(2, case.seq_len // 2)
        doc_ids[:, :split] = 1
        doc_ids[:, split : case.seq_len - case.pad_tail] = 2
        ref_payload["attention_mask"] = build_doc_block_mask(doc_ids)

    ref_out, ref_grads = _run(ref, ref_payload)
    eager_out, eager_grads = _run(eager, ref_payload)
    flash_out, flash_grads = _run(flash, payload)

    _assert_close_to_reference(
        case_name=case.name,
        label="eager_bf16_out",
        actual=eager_out,
        reference=ref_out,
        max_abs_limit=5e-2,
        mean_abs_limit=8e-3,
    )
    _assert_close_to_reference(
        case_name=case.name,
        label="flash_bf16_out",
        actual=flash_out,
        reference=ref_out,
        max_abs_limit=7e-2,
        mean_abs_limit=1.2e-2,
    )
    for key in ("word_embeddings", "rel_embeddings", "query", "value"):
        _assert_close_to_reference(
            case_name=case.name,
            label=f"eager_grad_{key}",
            actual=eager_grads[key],
            reference=ref_grads[key],
            max_abs_limit=8e-2,
            mean_abs_limit=1.5e-2,
        )
        _assert_close_to_reference(
            case_name=case.name,
            label=f"flash_grad_{key}",
            actual=flash_grads[key],
            reference=ref_grads[key],
            max_abs_limit=1.2e-1,
            mean_abs_limit=2e-2,
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
        ParityCase("docblock_bias", seq_len=1024, batch_size=2, route_hint="docblock_bias", docblock=True),
    ]
    for case in cases:
        _run_case(case, device=device)
    print("OK")


if __name__ == "__main__":
    main()
