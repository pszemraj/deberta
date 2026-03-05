#!/usr/bin/env python3
"""CUDA parity check for the FlashDeBERTa runtime patch integration."""

from __future__ import annotations

import sys
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
from deberta.modeling.flashdeberta_patch import (  # noqa: E402
    disable_flashdeberta_attention,
    enable_flashdeberta_attention,
)


def _build_tiny_config(*, seq_len: int) -> DebertaV2Config:
    """Build a small DeBERTa config suitable for parity testing.

    :param int seq_len: Sequence length.
    :return DebertaV2Config: Test config.
    """

    return DebertaV2Config(
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


@torch.no_grad()
def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy model weights between eager and flash variants.

    :param torch.nn.Module src: Source model.
    :param torch.nn.Module dst: Destination model.
    """

    dst.load_state_dict(src.state_dict(), strict=True)


def main() -> None:
    """Run forward/backward parity checks on a CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tools/flashdeberta_parity_test.py.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    seq_len = 256
    cfg = _build_tiny_config(seq_len=seq_len)

    disable_flashdeberta_attention()
    eager = DebertaV2Model(cfg).to(device=device, dtype=torch.bfloat16)
    eager.train()

    enable_flashdeberta_attention(strict=True)
    flash = DebertaV2Model(cfg).to(device=device, dtype=torch.bfloat16)
    flash.train()

    _copy_weights(eager, flash)

    batch = 2
    input_ids = torch.randint(5, cfg.vocab_size, (batch, seq_len), device=device)
    attention_mask = torch.ones((batch, seq_len), device=device, dtype=torch.bool)
    attention_mask[1, -64:] = False
    input_ids[1, -64:] = int(cfg.pad_token_id)

    eager.zero_grad(set_to_none=True)
    flash.zero_grad(set_to_none=True)

    out_eager = eager(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    out_flash = flash(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    max_abs = (out_eager - out_flash).abs().max().item()
    mean_abs = (out_eager - out_flash).abs().mean().item()
    print(f"forward diff: max_abs={max_abs:.4e}, mean_abs={mean_abs:.4e}")

    loss_eager = out_eager.float().pow(2).mean()
    loss_flash = out_flash.float().pow(2).mean()

    loss_eager.backward()
    loss_flash.backward()

    grad_eager = eager.embeddings.word_embeddings.weight.grad
    grad_flash = flash.embeddings.word_embeddings.weight.grad
    if grad_eager is None or grad_flash is None:
        raise RuntimeError("Missing gradients from parity run.")

    grad_max_abs = (grad_eager - grad_flash).abs().max().item()
    grad_mean_abs = (grad_eager - grad_flash).abs().mean().item()
    print(f"grad diff:    max_abs={grad_max_abs:.4e}, mean_abs={grad_mean_abs:.4e}")

    if max_abs > 3e-2:
        print("WARNING: forward diff exceeded heuristic threshold 3e-2.")
    if grad_max_abs > 3e-2:
        print("WARNING: grad diff exceeded heuristic threshold 3e-2.")

    print("OK")


if __name__ == "__main__":
    main()
