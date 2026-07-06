"""Tests for optional FlashDeBERTa runtime integration."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import sys
import types
import warnings

import pytest
import torch

from deberta.modeling.mask_utils import FlashBatchMeta


def _install_fake_flashdeberta(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install a minimal in-memory FlashDeBERTa module tree for tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    :return dict[str, int]: Mutable call counters for fake flash operators.
    """

    flash_pkg = types.ModuleType("flashdeberta")
    flash_pkg.__path__ = []  # type: ignore[attr-defined]
    flash_pkg.__version__ = "0.0.7"  # type: ignore[attr-defined]
    ops_pkg = types.ModuleType("flashdeberta.ops")
    ops_pkg.__path__ = []  # type: ignore[attr-defined]
    flash_attention_mod = types.ModuleType("flashdeberta.ops.flash_attention")
    flash_attention_varlen_mod = types.ModuleType("flashdeberta.ops.flash_attention_varlen")
    flash_attention_bias_mod = types.ModuleType("flashdeberta.ops.flash_attention_bias")
    calls = {"fixed": 0, "varlen": 0, "bias": 0}

    def _fake_flash_attention_with_disentangled(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_lengths: torch.Tensor | None,
        k_pos: torch.Tensor | None,
        q_pos: torch.Tensor | None,
        causal: bool = False,
        sm_scale: float | None = None,
        position_buckets: int = 0,
        max_relative_distance: int = 0,
    ) -> torch.Tensor:
        """Return value-shaped zeros for adapter tests.

        :param torch.Tensor q: Query tensor.
        :param torch.Tensor k: Key tensor.
        :param torch.Tensor v: Value tensor.
        :param torch.Tensor | None seq_lengths: Optional sequence lengths.
        :param torch.Tensor | None k_pos: Optional c2p tensor.
        :param torch.Tensor | None q_pos: Optional p2c tensor.
        :param bool causal: Unused causal flag.
        :param float | None sm_scale: Optional softmax scale.
        :param int position_buckets: Optional bucket count.
        :param int max_relative_distance: Optional max relative distance.
        :return torch.Tensor: Zero tensor shaped like ``q``.
        """

        calls["fixed"] += 1
        del k, v, seq_lengths, k_pos, q_pos, causal, sm_scale, position_buckets, max_relative_distance
        return torch.zeros_like(q)

    def _fake_flash_attention_with_disentangled_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_pos: torch.Tensor | None,
        q_pos: torch.Tensor | None,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool = False,
        sm_scale: float | None = None,
        position_buckets: int = 0,
        max_relative_distance: int = 0,
    ) -> torch.Tensor:
        """Return value-shaped zeros for varlen adapter tests.

        :param torch.Tensor q: Unpadded query tensor.
        :param torch.Tensor k: Unpadded key tensor.
        :param torch.Tensor v: Unpadded value tensor.
        :param torch.Tensor | None k_pos: Optional c2p tensor.
        :param torch.Tensor | None q_pos: Optional p2c tensor.
        :param torch.Tensor cu_seqlens_q: Query cumulative lengths.
        :param torch.Tensor cu_seqlens_k: Key cumulative lengths.
        :param int max_seqlen_q: Max query length in batch.
        :param int max_seqlen_k: Max key length in batch.
        :param bool causal: Unused causal flag.
        :param float | None sm_scale: Optional softmax scale.
        :param int position_buckets: Optional bucket count.
        :param int max_relative_distance: Optional max relative distance.
        :return torch.Tensor: Zero tensor shaped like ``q``.
        """

        calls["varlen"] += 1
        del (
            k,
            v,
            k_pos,
            q_pos,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            sm_scale,
            position_buckets,
            max_relative_distance,
        )
        return torch.zeros_like(q)

    def _fake_flash_attention_with_bias(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
        causal: bool = False,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        """Return value-shaped zeros for dense local-bias adapter tests.

        :param torch.Tensor q: Query tensor.
        :param torch.Tensor k: Key tensor.
        :param torch.Tensor v: Value tensor.
        :param torch.Tensor bias: Additive attention bias tensor.
        :param bool causal: Unused causal flag.
        :param float | None sm_scale: Optional softmax scale.
        :return torch.Tensor: Zero tensor shaped like ``q``.
        """

        calls["bias"] += 1
        del k, v, bias, causal, sm_scale
        return torch.zeros_like(q)

    flash_attention_mod.flash_attention_with_disentangled = _fake_flash_attention_with_disentangled
    flash_attention_varlen_mod.flash_attention_with_disentangled_varlen = (
        _fake_flash_attention_with_disentangled_varlen
    )
    flash_attention_bias_mod.flash_attention_with_bias = _fake_flash_attention_with_bias
    flash_pkg.ops = ops_pkg  # type: ignore[attr-defined]
    ops_pkg.flash_attention = flash_attention_mod  # type: ignore[attr-defined]
    ops_pkg.flash_attention_varlen = flash_attention_varlen_mod  # type: ignore[attr-defined]
    ops_pkg.flash_attention_bias = flash_attention_bias_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "flashdeberta", flash_pkg)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops.flash_attention", flash_attention_mod)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops.flash_attention_varlen", flash_attention_varlen_mod)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops.flash_attention_bias", flash_attention_bias_mod)

    real_version = importlib_metadata.version

    def _fake_distribution_version(name: str) -> str:
        if str(name) == "flashdeberta":
            return "0.0.7"
        return real_version(name)

    monkeypatch.setattr(importlib_metadata, "version", _fake_distribution_version)
    return calls


def _reload_flash_modules() -> tuple[types.ModuleType, types.ModuleType]:
    """Reload FlashDeBERTa adapter modules to pick up test-time fake imports.

    :return tuple[types.ModuleType, types.ModuleType]: Reloaded attention and patch modules.
    """

    sys.modules.pop("deberta.modeling.flashdeberta_attention", None)
    sys.modules.pop("deberta.modeling.flashdeberta_patch", None)
    attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
    patch_mod = importlib.import_module("deberta.modeling.flashdeberta_patch")
    return importlib.reload(attention_mod), importlib.reload(patch_mod)


def test_flashdeberta_version_guard_accepts_pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)

    import deberta.modeling.flashdeberta_version as version_mod

    version_mod = importlib.reload(version_mod)

    assert version_mod.flashdeberta_version_error() is None
    assert version_mod.flashdeberta_runtime_version() == "0.0.7"


def test_flashdeberta_kernel_tuning_table_resolves_default_policy() -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import (
        FlashKernelContext,
        configure_flashdeberta_kernel_overrides,
        flash_route_choice,
        flash_seq_bucket,
        resolve_flash_kernel_config,
    )

    configure_flashdeberta_kernel_overrides(None)

    bucket = flash_seq_bucket(seq_len=2048, total_tokens=3000, batch_size=2)
    assert bucket == "2048_medium"
    assert flash_route_choice(policy="padding", seq_bucket=bucket) == "varlen"
    assert flash_route_choice(policy="docblock", seq_bucket=flash_seq_bucket(seq_len=1024)) == "docblock_bias"
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="docblock",
            kind="bwd_kv",
            seq_len=1024,
            total_tokens=4096,
            batch_size=8,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            disentangled=True,
            att_span=256,
        )
    ) == (16, 16, 1, 2)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="docblock",
            kind="bwd_q",
            seq_len=1024,
            total_tokens=4096,
            batch_size=8,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            disentangled=True,
            att_span=256,
        )
    ) == (32, 32, 2, 4)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="docblock",
            kind="bwd_kv",
            seq_len=2048,
            total_tokens=4096,
            batch_size=2,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            disentangled=True,
            att_span=256,
        )
    ) == (64, 32, 2, 4)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="docblock",
            kind="bwd_q",
            seq_len=4096,
            total_tokens=4096,
            batch_size=1,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            disentangled=True,
            att_span=256,
        )
    ) == (64, 64, 3, 8)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="varlen",
            kind="bwd_kv",
            seq_len=2048,
            total_tokens=3000,
            batch_size=2,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            disentangled=True,
            att_span=256,
        )
    ) == (64, 32, 2, 4)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="dense_bias",
            kind="fwd",
            seq_len=4096,
            batch_size=1,
            num_heads=12,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
        )
    ) == (64, 128, 2, 4)
    assert resolve_flash_kernel_config(
        FlashKernelContext(
            compute_capability=(12, 0),
            route="bias_docblock_specialized",
            kind="bwd_kv",
            seq_len=4096,
            batch_size=1,
            query_len=4096,
            key_len=4096,
            num_heads=12,
            head_dim=64,
            dtype="bfloat16",
            causal=False,
            has_mask=True,
        )
    ) == (16, 32, 1, 4)


def test_flashdeberta_kernel_tuning_override_path_wins(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import (
        FlashKernelContext,
        configure_flashdeberta_kernel_overrides,
        resolve_flash_kernel_config,
    )

    override_path = tmp_path / "flash_tuning.json"
    override_path.write_text(
        """
{
  "kernels": [
    {
      "compute_capability": "sm_120",
      "route": "varlen",
      "kind": "bwd_kv",
      "seq_bucket": "2048_medium",
      "head_dim": 64,
      "block_m": 16,
      "block_n": 32,
      "num_stages": 1,
      "num_warps": 2
    }
  ]
}
""",
        encoding="utf-8",
    )

    try:
        configure_flashdeberta_kernel_overrides(str(override_path))
        assert resolve_flash_kernel_config(
            FlashKernelContext(
                compute_capability=(12, 0),
                route="varlen",
                kind="bwd_kv",
                seq_len=2048,
                total_tokens=3000,
                batch_size=2,
                head_dim=64,
            )
        ) == (16, 32, 1, 2)
    finally:
        configure_flashdeberta_kernel_overrides(None)


def test_flashdeberta_route_policy_override_path_changes_routing(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import configure_flashdeberta_kernel_overrides
    from deberta.training.compile import (
        _flash_route_hint_for_docblock_batch,
        _flash_route_hint_for_padding_batch,
    )

    override_path = tmp_path / "flash_routes.json"
    override_path.write_text(
        """
{
  "route_policies": {
    "padding": [
      {
        "seq_bucket": "2048_medium",
        "choice": "fixed"
      }
    ],
    "docblock": [
      {
        "seq_bucket": "1024_exact",
        "choice": "docblock"
      }
    ]
  }
}
""",
        encoding="utf-8",
    )

    try:
        configure_flashdeberta_kernel_overrides(str(override_path))
        assert _flash_route_hint_for_padding_batch(seq_len=2048, active_tokens=4096, batch_size=2) == "fixed"
        assert _flash_route_hint_for_docblock_batch(seq_len=1024) == "docblock"
    finally:
        configure_flashdeberta_kernel_overrides(None)


def test_docblock_bias_route_uses_table_with_ragged_override() -> None:
    from deberta.training.compile import _flash_route_hint_for_docblock_batch

    assert _flash_route_hint_for_docblock_batch(seq_len=1024) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=2048) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=4096) == "docblock_bias"
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=1024,
            flash_cfg={"docblock_bias_seq_len": 1024},
        )
        == "docblock_bias"
    )
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=1024,
            flash_cfg={"docblock_bias_seq_len": 0},
        )
        == "docblock"
    )


def _small_deberta_config():
    """Build a small config for native DeBERTa patch tests."""

    pytest.importorskip("transformers")
    from deberta.modeling.deberta_v2_native import DebertaV2Config

    return DebertaV2Config(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        position_biased_input=False,
    )


def _docblock_attention_config(*, seq_len: int):
    """Build a tiny native DeBERTa config for doc-block attention tests."""

    pytest.importorskip("transformers")
    from deberta.modeling.deberta_v2_native import DebertaV2Config

    return DebertaV2Config(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=int(seq_len),
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=4,
        max_relative_positions=8,
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        position_biased_input=False,
    )


def test_enable_flashdeberta_attention_validates_without_global_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, patch_mod = _reload_flash_modules()

    from deberta.modeling import deberta_v2_native as dv2
    from deberta.modeling import rtd

    patch_mod.disable_flashdeberta_attention()

    orig_attention = dv2.DisentangledSelfAttention
    orig_get_rel_pos = dv2.DebertaV2Encoder.get_rel_pos
    orig_emd_mask = rtd._ensure_emd_pairwise_attention_mask

    patch_mod.enable_flashdeberta_attention(strict=True)

    assert dv2.DisentangledSelfAttention is orig_attention
    assert dv2.DebertaV2Encoder.get_rel_pos is orig_get_rel_pos
    assert rtd._ensure_emd_pairwise_attention_mask is orig_emd_mask

    cfg = _small_deberta_config()
    cfg.hf_attention_impl = "flash"
    attention = dv2.DebertaV2Attention(cfg)
    assert isinstance(attention.self, attention_mod.FlashDisentangledSelfAttention)

    encoder = dv2.DebertaV2Encoder(cfg)
    hidden_states = torch.zeros((1, 4, cfg.hidden_size))
    relative_pos = torch.ones((4, 4), dtype=torch.long)

    assert encoder.get_rel_pos(hidden_states) is None
    assert encoder.get_rel_pos(hidden_states, relative_pos=relative_pos) is relative_pos

    flash_broadcast_mask = rtd._ensure_emd_flash_attention_mask(torch.tensor([[1, 1, 0]], dtype=torch.long))
    assert flash_broadcast_mask.dtype == torch.bool
    assert tuple(flash_broadcast_mask.shape) == (1, 1, 1, 3)
    assert torch.equal(flash_broadcast_mask[0, 0, 0], torch.tensor([True, True, False]))

    pairwise = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)
    pairwise_mask = rtd._ensure_emd_flash_attention_mask(pairwise)
    assert tuple(pairwise_mask.shape) == (1, 1, 2, 2)
    assert torch.equal(pairwise_mask[:, 0], pairwise)

    patch_mod.disable_flashdeberta_attention()

    assert dv2.DisentangledSelfAttention is orig_attention
    assert dv2.DebertaV2Encoder.get_rel_pos is orig_get_rel_pos
    assert rtd._ensure_emd_pairwise_attention_mask is orig_emd_mask


def test_native_attention_selects_flash_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    from deberta.modeling.deberta_v2_native import DebertaV2Attention

    cfg = _small_deberta_config()
    cfg.hf_attention_impl = "flash"

    attention = DebertaV2Attention(cfg)

    assert isinstance(attention.self, attention_mod.FlashDisentangledSelfAttention)


def test_enable_flashdeberta_attention_strict_false_is_noop_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "flashdeberta", raising=False)
    monkeypatch.delitem(sys.modules, "flashdeberta.ops", raising=False)
    monkeypatch.delitem(sys.modules, "flashdeberta.ops.flash_attention", raising=False)
    _, patch_mod = _reload_flash_modules()

    from deberta.modeling import deberta_v2_native as dv2

    original_import_module = patch_mod.importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "flashdeberta":
            raise ImportError("simulated missing flashdeberta")
        return original_import_module(name, package)

    monkeypatch.setattr(patch_mod.importlib, "import_module", _fake_import_module)

    patch_mod.disable_flashdeberta_attention()
    orig_attention = dv2.DisentangledSelfAttention
    patch_mod.enable_flashdeberta_attention(strict=False)
    assert dv2.DisentangledSelfAttention is orig_attention


def test_flash_attention_pairwise_mask_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    cfg.hf_flash = {"force_varlen": True, "varlen_min_seq_len": 2048, "eager_dense_max_seq_len": 0}
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))
    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    pairwise_mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ]
        ]
    )

    seen: dict[str, object] = {}

    def _fake_eager_forward(self, *args, **kwargs):
        """Record eager fallback calls for the pairwise-mask test."""

        del self
        seen["args"] = args
        seen["kwargs"] = kwargs
        return torch.full((1, 4, cfg.hidden_size), 7.0), None

    monkeypatch.setattr(attention_mod._EagerDisentangledSelfAttention, "forward", _fake_eager_forward)

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=pairwise_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert "kwargs" in seen
    assert seen["kwargs"]["attention_mask"] is pairwise_mask


def test_flash_attention_projected_qkv_dtype_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    bf16_qkv = torch.zeros((1, cfg.num_attention_heads, 4, head_dim), dtype=torch.bfloat16)
    fp32_qkv = bf16_qkv.float()

    assert (
        attention._projected_qkv_fallback_reason(
            query_layer=bf16_qkv,
            key_layer=bf16_qkv,
            value_layer=bf16_qkv,
        )
        is None
    )
    reason = attention._projected_qkv_fallback_reason(
        query_layer=fp32_qkv,
        key_layer=fp32_qkv,
        value_layer=fp32_qkv,
    )
    assert reason is not None
    assert reason[0] == "dtype"


def test_flash_attention_varlen_path_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    cfg.hf_flash = {"force_varlen": True, "varlen_min_seq_len": 2048, "eager_dense_max_seq_len": 0}
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    seen: dict[str, torch.Tensor] = {}

    def _fake_varlen_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask_2d: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> torch.Tensor:
        """Return zero output while recording padded-varlen wrapper inputs."""

        del (
            key_layer,
            value_layer,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        seen["mask"] = attention_mask_2d
        return torch.zeros_like(query_layer)

    monkeypatch.setattr(attention_mod, "flashdeberta_varlen_padded", _fake_varlen_wrapper)

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    attention_mod.reset_flashdeberta_stats()
    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert tuple(seen["mask"].shape) == (1, 4)
    assert seen["mask"].dtype == torch.bool
    assert torch.equal(seen["mask"], attention_mask)

    stats = attention_mod.flashdeberta_stats_snapshot()
    assert stats["forward_calls"] == 1
    assert stats["flash_eligible_calls"] == 1
    assert stats["flash_varlen_calls"] == 1
    assert stats.get("flash_fixed_calls", 0) == 0
    assert stats.get("fallback_calls", 0) == 0


def test_flash_attention_fixed_path_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    seen: dict[str, object] = {}

    def _fake_fixed_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        seq_lengths: torch.Tensor | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        causal: bool,
    ) -> torch.Tensor:
        """Return zero output while recording fixed wrapper inputs."""

        del (
            key_layer,
            value_layer,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        seen["seq_lengths"] = seq_lengths
        return torch.zeros_like(query_layer)

    monkeypatch.setattr(attention_mod, "flashdeberta_fixed", _fake_fixed_wrapper)

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    attention_mod.reset_flashdeberta_stats()
    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert seen["seq_lengths"] is None

    stats = attention_mod.flashdeberta_stats_snapshot()
    assert stats["forward_calls"] == 1
    assert stats["flash_eligible_calls"] == 1
    assert stats["flash_fixed_calls"] == 1
    assert stats.get("flash_varlen_calls", 0) == 0
    assert stats.get("fallback_calls", 0) == 0


def test_flash_attention_dense_local_bias_path_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    cfg.max_position_embeddings = 1024
    cfg.max_relative_positions = 1024
    cfg.position_buckets = 256
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)
    attention.train()

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    seen: dict[str, object] = {}

    def _fake_bias_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor | None,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Return zero output while recording compact local-bias inputs."""

        del key_layer, value_layer, pos_key, pos_query, bias_scale, sm_scale, causal
        seen["bucket_shape"] = tuple(bucket_index.shape)
        seen["keep_mask"] = keep_mask
        seen["query_shape"] = tuple(query_layer.shape)
        return torch.zeros_like(query_layer)

    monkeypatch.setattr(attention_mod, "flashdeberta_bias_from_positions", _fake_bias_wrapper)

    hidden_states = torch.randn((1, 1024, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    attention_mod.reset_flashdeberta_stats()
    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 1024, cfg.hidden_size)
    assert seen["query_shape"] == (
        1,
        cfg.num_attention_heads,
        1024,
        cfg.hidden_size // cfg.num_attention_heads,
    )
    assert seen["bucket_shape"] == (1024, 1024)
    assert seen["keep_mask"] is None

    stats = attention_mod.flashdeberta_stats_snapshot()
    assert stats["forward_calls"] == 1
    assert stats["flash_eligible_calls"] == 1
    assert stats["flash_bias_calls"] == 1
    assert stats.get("flash_fixed_calls", 0) == 0
    assert stats.get("flash_varlen_calls", 0) == 0
    assert stats.get("fallback_calls", 0) == 0


def test_prefix_pack_pair_and_triple_cpu_roundtrip() -> None:
    from deberta.modeling.flashdeberta_prefix_pack import (
        prefix_pack_padded_rows_pair,
        prefix_pack_padded_rows_triple,
        prefix_unpack_padded_rows_pair,
        prefix_unpack_padded_rows_triple,
    )

    seqlens = torch.tensor([3, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 3, 4], dtype=torch.int32)
    a = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    b = a + 100.0
    c = a + 200.0

    packed_a, packed_b = prefix_pack_padded_rows_pair(
        a,
        b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        total_tokens=4,
    )
    packed_c1, packed_c2, packed_c3 = prefix_pack_padded_rows_triple(
        a,
        b,
        c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=4,
        total_tokens=4,
    )

    expected_a = torch.cat([a[0, :3], a[1, :1]], dim=0)
    expected_b = torch.cat([b[0, :3], b[1, :1]], dim=0)
    expected_c = torch.cat([c[0, :3], c[1, :1]], dim=0)
    assert torch.equal(packed_a, expected_a)
    assert torch.equal(packed_b, expected_b)
    assert torch.equal(packed_c1, expected_a)
    assert torch.equal(packed_c2, expected_b)
    assert torch.equal(packed_c3, expected_c)

    unpacked_a, unpacked_b = prefix_unpack_padded_rows_pair(
        packed_a,
        packed_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )
    unpacked_c1, unpacked_c2, unpacked_c3 = prefix_unpack_padded_rows_triple(
        packed_c1,
        packed_c2,
        packed_c3,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )

    expected_unpacked_a = torch.zeros_like(a)
    expected_unpacked_b = torch.zeros_like(b)
    expected_unpacked_c = torch.zeros_like(c)
    expected_unpacked_a[0, :3] = a[0, :3]
    expected_unpacked_a[1, :1] = a[1, :1]
    expected_unpacked_b[0, :3] = b[0, :3]
    expected_unpacked_b[1, :1] = b[1, :1]
    expected_unpacked_c[0, :3] = c[0, :3]
    expected_unpacked_c[1, :1] = c[1, :1]
    assert torch.equal(unpacked_a, expected_unpacked_a)
    assert torch.equal(unpacked_b, expected_unpacked_b)
    assert torch.equal(unpacked_c1, expected_unpacked_a)
    assert torch.equal(unpacked_c2, expected_unpacked_b)
    assert torch.equal(unpacked_c3, expected_unpacked_c)


def test_segment_pack_pair_and_triple_cpu_roundtrip() -> None:
    from deberta.modeling.flashdeberta_segment_pack import (
        segment_pack_grad_and_delta_from_padded,
        segment_pack_padded_rows_pair,
        segment_pack_padded_rows_triple,
        segment_unpack_padded_rows_pair,
        segment_unpack_padded_rows_triple,
    )

    segment_offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    segment_lengths = torch.tensor([2, 2, 1, 2], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 4, 5, 7], dtype=torch.int32)
    base = torch.arange(2 * 5 * 3 * 3, dtype=torch.float32).view(2, 5, 3, 3)
    a = base[..., 0]
    b = base[..., 1]
    c = base[..., 2]

    assert not a.is_contiguous()
    assert not b.is_contiguous()
    assert not c.is_contiguous()

    packed_a, packed_b = segment_pack_padded_rows_pair(
        a,
        b,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=7,
    )
    packed_c1, packed_c2, packed_c3 = segment_pack_padded_rows_triple(
        a,
        b,
        c,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=7,
    )

    expected_a = torch.cat((a[0, :2], a[0, 2:4], a[1, :1], a[1, 1:3]), dim=0)
    expected_b = torch.cat((b[0, :2], b[0, 2:4], b[1, :1], b[1, 1:3]), dim=0)
    expected_c = torch.cat((c[0, :2], c[0, 2:4], c[1, :1], c[1, 1:3]), dim=0)
    expected_base = torch.cat((base[0, :2], base[0, 2:4], base[1, :1], base[1, 1:3]), dim=0)
    assert torch.equal(packed_a, expected_a)
    assert torch.equal(packed_b, expected_b)
    assert torch.equal(packed_c1, expected_a)
    assert torch.equal(packed_c2, expected_b)
    assert torch.equal(packed_c3, expected_c)

    grad_unpad, delta = segment_pack_grad_and_delta_from_padded(
        grad_output=base,
        out_unpad=expected_base + 1.0,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=7,
    )
    assert torch.equal(grad_unpad, expected_base)
    assert torch.equal(delta, ((expected_base + 1.0) * expected_base).sum(dim=-1))

    unpacked_a, unpacked_b = segment_unpack_padded_rows_pair(
        packed_a,
        packed_b,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=5,
    )
    unpacked_c1, unpacked_c2, unpacked_c3 = segment_unpack_padded_rows_triple(
        packed_c1,
        packed_c2,
        packed_c3,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=5,
    )

    expected_unpacked_a = torch.zeros_like(a)
    expected_unpacked_b = torch.zeros_like(b)
    expected_unpacked_c = torch.zeros_like(c)
    expected_unpacked_a[0, :2] = a[0, :2]
    expected_unpacked_a[0, 2:4] = a[0, 2:4]
    expected_unpacked_a[1, :1] = a[1, :1]
    expected_unpacked_a[1, 1:3] = a[1, 1:3]
    expected_unpacked_b[0, :2] = b[0, :2]
    expected_unpacked_b[0, 2:4] = b[0, 2:4]
    expected_unpacked_b[1, :1] = b[1, :1]
    expected_unpacked_b[1, 1:3] = b[1, 1:3]
    expected_unpacked_c[0, :2] = c[0, :2]
    expected_unpacked_c[0, 2:4] = c[0, 2:4]
    expected_unpacked_c[1, :1] = c[1, :1]
    expected_unpacked_c[1, 1:3] = c[1, 1:3]
    assert torch.equal(unpacked_a, expected_unpacked_a)
    assert torch.equal(unpacked_b, expected_unpacked_b)
    assert torch.equal(unpacked_c1, expected_unpacked_a)
    assert torch.equal(unpacked_c2, expected_unpacked_b)
    assert torch.equal(unpacked_c3, expected_unpacked_c)


def test_docblock_forward_pads_saved_aux_without_expanding_kernel_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deberta.modeling.flashdeberta_docblock_op as docblock_mod

    seen: dict[str, int] = {}

    def _fake_fwd(
        q,
        k,
        v,
        pos_key,
        pos_query,
        cu_q,
        cu_k,
        max_q,
        max_k,
        causal,
        sm_scale,
        block_m,
        block_n,
        position_buckets,
        max_relative_distance,
        num_warps,
        num_stages,
        att_span,
    ):
        del (
            k,
            v,
            pos_key,
            pos_query,
            cu_q,
            cu_k,
            max_q,
            max_k,
            causal,
            sm_scale,
            block_m,
            block_n,
            position_buckets,
            max_relative_distance,
            num_warps,
            num_stages,
            att_span,
        )
        seen["tokens"] = int(q.shape[0])
        return q + 1.0, torch.zeros((q.shape[0], q.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(docblock_mod._varlen_mod, "_flash_attn_v2_fwd_dise_lowlevel", _fake_fwd)
    monkeypatch.setattr(
        docblock_mod._varlen_mod,
        "_get_fwd_config_lowlevel",
        lambda **kwargs: (16, 16, 1, 1),
    )

    q = torch.arange(1 * 5 * 2 * 3, dtype=torch.float32).view(1, 5, 2, 3)
    pos = torch.arange(1 * 5 * 2 * 4, dtype=torch.float32).view(1, 5, 2, 4)
    segment_offsets = torch.tensor([0, 3], dtype=torch.int32)
    segment_lengths = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)

    output, lse, q_aux, k_aux, v_aux, out_aux, lse_aux, pos_key_aux, pos_query_aux = (
        docblock_mod._docblock_forward_impl(
            query_layer=q,
            key_layer=q + 10,
            value_layer=q + 20,
            segment_offsets=segment_offsets,
            segment_lengths=segment_lengths,
            cu_seqlens=cu_seqlens,
            pos_key=pos,
            pos_query=pos + 10,
            sm_scale=1.0,
            position_buckets=4,
            max_relative_distance=4,
            causal=False,
            num_segments=2,
            max_seqlen=2,
            total_tokens=3,
            require_lse=True,
            aux_capacity=5,
        )
    )

    assert seen["tokens"] == 3
    assert q_aux.shape[0] == 5
    assert k_aux.shape[0] == 5
    assert v_aux.shape[0] == 5
    assert out_aux.shape[0] == 5
    assert lse_aux.shape[0] == 5
    assert pos_key_aux is not None and pos_key_aux.shape[0] == 5
    assert pos_query_aux is not None and pos_query_aux.shape[0] == 5
    assert lse is not None and lse.shape == (1, 5, 2)
    expected = torch.zeros_like(q)
    expected[0, :2] = q[0, :2] + 1.0
    expected[0, 3:4] = q[0, 3:4] + 1.0
    assert torch.equal(output, expected)


def test_docblock_backward_narrows_fixed_capacity_saved_aux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deberta.modeling.flashdeberta_docblock_op as docblock_mod

    seen: dict[str, int] = {}

    def _fake_backward_raw(
        *,
        q_unpad,
        k_unpad,
        v_unpad,
        out_unpad,
        grad_unpad,
        lse_unpad,
        delta,
        pos_key_unpad,
        pos_query_unpad,
        cu_seqlens,
        batch_size,
        seq_bound,
        token_capacity,
        sm_scale,
        position_buckets,
        max_relative_distance,
        causal,
        dense_mid_tensors,
        route,
    ):
        del (
            k_unpad,
            v_unpad,
            out_unpad,
            grad_unpad,
            lse_unpad,
            delta,
            cu_seqlens,
            batch_size,
            seq_bound,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
            dense_mid_tensors,
        )
        seen["route"] = str(route)
        seen["q_tokens"] = int(q_unpad.shape[0])
        seen["pos_tokens"] = int(pos_key_unpad.shape[0]) if pos_key_unpad is not None else -1
        seen["capacity"] = int(token_capacity)
        dpos_key = torch.full_like(pos_key_unpad, 4.0) if pos_key_unpad is not None else None
        dpos_query = torch.full_like(pos_query_unpad, 5.0) if pos_query_unpad is not None else None
        return (
            torch.ones_like(q_unpad),
            torch.full_like(q_unpad, 2.0),
            torch.full_like(q_unpad, 3.0),
            dpos_key,
            dpos_query,
        )

    monkeypatch.setattr(docblock_mod._varlen_mod, "_varlen_backward_raw_impl", _fake_backward_raw)

    q = torch.zeros((1, 5, 2, 3), dtype=torch.float32)
    pos = torch.zeros((1, 5, 2, 4), dtype=torch.float32)
    segment_offsets = torch.tensor([0, 3], dtype=torch.int32)
    segment_lengths = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    q_aux = torch.randn((5, 2, 3))
    out_aux = torch.randn((5, 2, 3))
    lse_aux = torch.randn((5, 2))
    pos_aux = torch.randn((5, 2, 4))

    dq, dk, dv, dpos_key, dpos_query = docblock_mod._docblock_backward_impl(
        grad_output=torch.ones_like(q),
        query_layer=q,
        key_layer=q,
        value_layer=q,
        output_padded=q,
        lse_padded=torch.zeros((1, 5, 2), dtype=torch.float32),
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        pos_key=pos,
        pos_query=pos,
        sm_scale=1.0,
        position_buckets=4,
        max_relative_distance=4,
        causal=False,
        num_segments=2,
        max_seqlen=2,
        total_tokens=3,
        q_unpad=q_aux,
        k_unpad=q_aux,
        v_unpad=q_aux,
        out_unpad=out_aux,
        lse_unpad=lse_aux,
        pos_key_unpad=pos_aux,
        pos_query_unpad=pos_aux,
    )

    assert seen == {"route": "docblock", "q_tokens": 3, "pos_tokens": 3, "capacity": 3}
    assert torch.equal(dq[0, :2], torch.ones_like(dq[0, :2]))
    assert torch.equal(dq[0, 3:4], torch.ones_like(dq[0, 3:4]))
    assert torch.equal(dk[0, :2], torch.full_like(dk[0, :2], 2.0))
    assert torch.equal(dv[0, 3:4], torch.full_like(dv[0, 3:4], 3.0))
    assert dpos_key is not None and torch.equal(dpos_key[0, 3:4], torch.full_like(dpos_key[0, 3:4], 4.0))
    assert dpos_query is not None and torch.equal(
        dpos_query[0, :2],
        torch.full_like(dpos_query[0, :2], 5.0),
    )


def test_flash_attention_debug_stats_skip_during_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    attention_mod.reset_flashdeberta_stats()

    attention_mod._record_stat("forward_calls")
    assert attention_mod.flashdeberta_stats_snapshot()["forward_calls"] == 1

    monkeypatch.setattr(attention_mod, "_is_torch_compiling", lambda: True)
    attention_mod._record_stat("forward_calls")

    assert attention_mod.flashdeberta_stats_snapshot()["forward_calls"] == 1

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        attention_mod.FlashDisentangledSelfAttention._warn_once(
            reason="compile_skip",
            message="should not warn while compiling",
        )

    assert len(captured) == 0


def test_varlen_remains_enabled_while_compiling_when_custom_op_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()

    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    monkeypatch.setattr(attention_mod, "_is_torch_compiling", lambda: True)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: True)
    attention_mod.refresh_flashdeberta_runtime_config_from_env()

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=1024) is False
    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=2048) is True

    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: False)

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=2048) is False


def test_varlen_min_seq_len_config_override_restores_1024_varlen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()

    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    monkeypatch.setattr(attention_mod, "_is_torch_compiling", lambda: False)

    assert (
        attention_mod._should_use_varlen(
            attention_mask=mask,
            seq_len=1024,
            runtime_config=attention_mod.FlashDebertaRuntimeConfig(varlen_min_seq_len=1024),
        )
        is True
    )


def test_varlen_wrapper_prefers_triton_op_while_compiling(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    calls = {"triton": 0, "custom": 0}

    def _fake_triton_op(*args):
        calls["triton"] += 1
        return torch.ones_like(args[0]), torch.zeros(
            (args[0].shape[0], args[0].shape[1], args[0].shape[2]),
            dtype=torch.float32,
            device=args[0].device,
        )

    def _fake_custom_op(*args):
        calls["custom"] += 1
        return torch.zeros_like(args[0]), torch.zeros(
            (args[0].shape[0], args[0].shape[1], args[0].shape[2]),
            dtype=torch.float32,
            device=args[0].device,
        )

    monkeypatch.setattr(varlen_mod, "_is_torch_compiling", lambda: True)
    monkeypatch.setattr(varlen_mod, "_FLASHDEBERTA_VARLEN_TRITON_OP", _fake_triton_op)
    monkeypatch.setattr(varlen_mod, "_FLASHDEBERTA_VARLEN_CUSTOM_OP", _fake_custom_op)

    with fake_tensor_mod.FakeTensorMode():
        q = torch.empty((1, 4, 2, 8), device="cuda", dtype=torch.bfloat16)
        k = torch.empty((1, 4, 2, 8), device="cuda", dtype=torch.bfloat16)
        v = torch.empty((1, 4, 2, 8), device="cuda", dtype=torch.bfloat16)
        mask = torch.ones((1, 4), device="cuda", dtype=torch.bool)

        output = varlen_mod.flashdeberta_varlen_padded(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            attention_mask_2d=mask,
            pos_key=None,
            pos_query=None,
            sm_scale=1.0,
            position_buckets=32,
            max_relative_distance=128,
            causal=False,
        )

    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert output.device.type == q.device.type
    assert calls == {"triton": 1, "custom": 0}


def test_varlen_metadata_cache_reuses_repeated_mask_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    varlen_mod._clear_unpad_metadata_cache()
    calls = {"count": 0}
    orig_build = varlen_mod._build_unpad_metadata

    def _counting_build(mask_2d: torch.Tensor):
        calls["count"] += 1
        return orig_build(mask_2d)

    monkeypatch.setattr(varlen_mod, "_build_unpad_metadata", _counting_build)

    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
    first_seqlens, first_cu, first_max = varlen_mod._get_unpad_metadata_cached(mask)
    second_seqlens, second_cu, second_max = varlen_mod._get_unpad_metadata_cached(mask)
    clone_seqlens, clone_cu, clone_max = varlen_mod._get_unpad_metadata_cached(mask.clone())

    assert calls["count"] == 2
    assert first_max == second_max == clone_max == 2
    assert first_seqlens.data_ptr() == second_seqlens.data_ptr()
    assert first_cu.data_ptr() == second_cu.data_ptr()
    assert clone_seqlens.data_ptr() != first_seqlens.data_ptr()
    assert clone_cu.data_ptr() != first_cu.data_ptr()

    varlen_mod._clear_unpad_metadata_cache()


def test_varlen_mid_tensor_cache_reuses_registered_cu_seqlens() -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    varlen_mod._clear_unpad_metadata_cache()
    varlen_mod._clear_mid_tensor_cache()

    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, False, False],
        ],
        dtype=torch.bool,
    )
    entry = varlen_mod._get_unpad_metadata_entry(mask)

    first_batch, first_start, first_mn = varlen_mod._get_mid_tensors_cached(
        cu_seqlens=entry.cu_seqlens,
        block_m=2,
        device=entry.cu_seqlens.device,
    )
    second_batch, second_start, second_mn = varlen_mod._get_mid_tensors_cached(
        cu_seqlens=entry.cu_seqlens,
        block_m=2,
        device=entry.cu_seqlens.device,
    )

    assert first_mn == second_mn == 2
    assert torch.equal(first_batch.cpu(), torch.tensor([0, 1], dtype=torch.long))
    assert torch.equal(first_start.cpu(), torch.tensor([0, 2], dtype=torch.long))
    assert first_batch.data_ptr() == second_batch.data_ptr()
    assert first_start.data_ptr() == second_start.data_ptr()

    varlen_mod._clear_mid_tensor_cache()
    varlen_mod._clear_unpad_metadata_cache()


def test_prefix_pack_round_trips_with_prefix_padding_contract() -> None:
    import deberta.modeling.flashdeberta_prefix_pack as prefix_mod

    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3).contiguous()
    seqlens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    packed = prefix_mod.prefix_pack_padded_rows(
        tensor,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )
    unpacked = prefix_mod.prefix_unpack_padded_rows(
        packed,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )

    expected_packed = torch.cat((tensor[0, :2], tensor[1, :3]), dim=0)
    expected_unpacked = torch.zeros_like(tensor)
    expected_unpacked[0, :2] = tensor[0, :2]
    expected_unpacked[1, :3] = tensor[1, :3]

    assert torch.equal(packed, expected_packed)
    assert torch.equal(unpacked, expected_unpacked)


def test_prefix_pack_explicit_total_tokens_avoids_tensor_item(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_prefix_pack as prefix_mod

    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3).contiguous()
    seqlens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    original_item = torch.Tensor.item

    def _forbid_item(self, *args, **kwargs):
        raise AssertionError("Tensor.item should not be used when total_tokens is provided")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    try:
        packed = prefix_mod.prefix_pack_padded_rows(
            tensor,
            seqlens=seqlens,
            cu_seqlens=cu_seqlens,
            max_seqlen=3,
            total_tokens=5,
        )
    finally:
        monkeypatch.setattr(torch.Tensor, "item", original_item)

    expected = torch.cat((tensor[0, :2], tensor[1, :3]), dim=0)
    assert torch.equal(packed, expected)


def test_prefix_unpack_pair_and_triple_match_single_tensor_behavior() -> None:
    import deberta.modeling.flashdeberta_prefix_pack as prefix_mod

    tensor_a = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3).contiguous()
    tensor_b = (tensor_a + 100.0).contiguous()
    tensor_c = (tensor_a + 200.0).contiguous()
    seqlens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    packed_a = prefix_mod.prefix_pack_padded_rows(
        tensor_a,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )
    packed_b = prefix_mod.prefix_pack_padded_rows(
        tensor_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )
    packed_c = prefix_mod.prefix_pack_padded_rows(
        tensor_c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )

    unpacked_pair_a, unpacked_pair_b = prefix_mod.prefix_unpack_padded_rows_pair(
        packed_a,
        packed_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )
    unpacked_triple_a, unpacked_triple_b, unpacked_triple_c = prefix_mod.prefix_unpack_padded_rows_triple(
        packed_a,
        packed_b,
        packed_c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )

    expected_a = prefix_mod.prefix_unpack_padded_rows(
        packed_a,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )
    expected_b = prefix_mod.prefix_unpack_padded_rows(
        packed_b,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )
    expected_c = prefix_mod.prefix_unpack_padded_rows(
        packed_c,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        batch_size=2,
        seq_len=4,
    )

    assert torch.equal(unpacked_pair_a, expected_a)
    assert torch.equal(unpacked_pair_b, expected_b)
    assert torch.equal(unpacked_triple_a, expected_a)
    assert torch.equal(unpacked_triple_b, expected_b)
    assert torch.equal(unpacked_triple_c, expected_c)


def test_pack_grad_and_delta_from_padded_matches_reference() -> None:
    import deberta.modeling.flashdeberta_prefix_pack as prefix_mod
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    output = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).view(2, 4, 2, 3).contiguous()
    grad = (output + 10.0).contiguous()
    seqlens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    expected_out = prefix_mod.prefix_pack_padded_rows(
        output,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )
    expected_grad = prefix_mod.prefix_pack_padded_rows(
        grad,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
    )
    expected_delta = (expected_out.to(dtype=torch.float32) * expected_grad.to(dtype=torch.float32)).sum(
        dim=-1
    )

    out_unpad, grad_unpad, delta = varlen_mod._pack_grad_and_delta_from_padded(
        grad_output=grad,
        output_padded=output,
        out_unpad=None,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        total_tokens=5,
    )
    assert torch.equal(out_unpad, expected_out)
    assert torch.equal(grad_unpad, expected_grad)
    assert torch.equal(delta, expected_delta)

    cached_out, cached_grad, cached_delta = varlen_mod._pack_grad_and_delta_from_padded(
        grad_output=grad,
        output_padded=output,
        out_unpad=expected_out,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=3,
        total_tokens=5,
    )
    assert torch.equal(cached_out, expected_out)
    assert torch.equal(cached_grad, expected_grad)
    assert torch.equal(cached_delta, expected_delta)


def test_flashdeberta_pack_and_varlen_modules_import_without_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-importing these modules chain-imports the rest of the flash tree
    # (fixed_op, upstream flashdeberta) under the poisoned triton. Snapshot the
    # whole affected namespace and restore the original healthy module objects
    # afterwards so later tests never see import state cached under triton=None.
    affected_prefixes = ("deberta.modeling.flashdeberta_", "flashdeberta")
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith(affected_prefixes)}
    for name in saved:
        sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)

    try:
        prefix_mod = importlib.import_module("deberta.modeling.flashdeberta_prefix_pack")
        segment_mod = importlib.import_module("deberta.modeling.flashdeberta_segment_pack")
        varlen_mod = importlib.import_module("deberta.modeling.flashdeberta_varlen_op")
        docblock_mod = importlib.import_module("deberta.modeling.flashdeberta_docblock_op")
        bias_mod = importlib.import_module("deberta.modeling.flashdeberta_bias_op")
        dense_bias_mod = importlib.import_module("deberta.modeling.flashdeberta_dense_bias_op")

        assert prefix_mod.flashdeberta_prefix_pack_available() is False
        assert segment_mod.flashdeberta_segment_pack_available() is False
        assert varlen_mod.flashdeberta_compiled_varlen_available() is False
        # The custom-op modules recover previously registered ops from the
        # process-global torch.library registry, so availability can be True
        # here when an earlier healthy import registered them. The contract
        # under test is import safety: the calls must not raise.
        assert isinstance(docblock_mod.flashdeberta_compiled_docblock_available(), bool)
        assert isinstance(bias_mod.flashdeberta_compiled_bias_available(), bool)
        assert isinstance(bias_mod.flashdeberta_compiled_position_bias_available(), bool)
        assert isinstance(dense_bias_mod.flashdeberta_compiled_dense_bias_available(), bool)
    finally:
        for name in [n for n in sys.modules if n.startswith(affected_prefixes)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def test_prepare_flash_attention_batch_metadata_routes_dense_pairwise_and_padded() -> None:
    import deberta.training.compile as compile_mod

    dense_batch = {"input_ids": torch.zeros((2, 1024), dtype=torch.long)}
    prepared_dense, dense_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=dense_batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert prepared_dense is dense_batch
    assert dense_meta is not None
    assert dense_meta.normalized_route_hint() == "dense"
    assert "flash_seq_lengths" not in prepared_dense
    assert "flash_active_tokens" not in prepared_dense

    pairwise_batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4, 4), dtype=torch.bool),
    }
    prepared_pairwise, pairwise_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=pairwise_batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert prepared_pairwise is pairwise_batch
    assert pairwise_meta is None
    assert "flash_seq_lengths" not in prepared_pairwise
    assert "flash_active_tokens" not in prepared_pairwise

    padded_1024 = {
        "input_ids": torch.zeros((2, 1024), dtype=torch.long),
        "attention_mask": torch.cat(
            (
                torch.ones((1, 1024), dtype=torch.bool),
                torch.cat(
                    (
                        torch.ones((1, 768), dtype=torch.bool),
                        torch.zeros((1, 256), dtype=torch.bool),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        ),
    }
    prepared_fixed, fixed_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=padded_1024,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert fixed_meta is not None
    assert fixed_meta.normalized_route_hint() == "fixed"
    assert torch.equal(prepared_fixed["flash_seq_lengths"], torch.tensor([1024, 768], dtype=torch.int32))
    assert prepared_fixed["flash_active_tokens"] == 1792
    assert fixed_meta.active_tokens_host == 1792

    padded_2048 = {
        "input_ids": torch.zeros((2, 2048), dtype=torch.long),
        "attention_mask": torch.cat(
            (
                torch.cat(
                    (
                        torch.ones((1, 1800), dtype=torch.bool),
                        torch.zeros((1, 248), dtype=torch.bool),
                    ),
                    dim=1,
                ),
                torch.cat(
                    (
                        torch.ones((1, 1700), dtype=torch.bool),
                        torch.zeros((1, 348), dtype=torch.bool),
                    ),
                    dim=1,
                ),
            ),
            dim=0,
        ),
    }
    prepared_varlen, varlen_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=padded_2048,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert varlen_meta is not None
    assert varlen_meta.normalized_route_hint() == "varlen"
    assert torch.equal(prepared_varlen["flash_seq_lengths"], torch.tensor([1800, 1700], dtype=torch.int32))
    assert prepared_varlen["flash_active_tokens"] == 3500
    assert varlen_meta.active_tokens_host == 3500


def test_prepare_flash_attention_batch_metadata_routes_docblock() -> None:
    import deberta.training.compile as compile_mod

    batch = {
        "input_ids": torch.zeros((2, 5), dtype=torch.long),
        "doc_ids": torch.tensor(
            [
                [1, 1, 2, 2, 0],
                [1, 2, 2, 0, 0],
            ],
            dtype=torch.long,
        ),
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )

    assert meta is not None
    assert meta.normalized_route_hint() == "docblock"
    assert "doc_ids" not in prepared
    assert torch.equal(
        prepared["attention_mask"],
        torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, True, False, False],
            ],
            dtype=torch.bool,
        ),
    )
    assert torch.equal(prepared["flash_seq_lengths"], torch.tensor([4, 3], dtype=torch.int32))
    assert prepared["flash_active_tokens"] == 7
    assert meta.active_tokens_host == 7
    assert tuple(prepared["flash_doc_segment_offsets"].shape) == (10,)
    assert tuple(prepared["flash_doc_segment_lengths"].shape) == (10,)
    assert tuple(prepared["flash_doc_cu_seqlens"].shape) == (11,)
    assert torch.equal(
        prepared["flash_doc_segment_offsets"][:4],
        torch.tensor([0, 2, 5, 6], dtype=torch.int32),
    )
    assert torch.equal(
        prepared["flash_doc_segment_lengths"][:4],
        torch.tensor([2, 2, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        prepared["flash_doc_cu_seqlens"][:5],
        torch.tensor([0, 2, 4, 5, 7], dtype=torch.int32),
    )
    assert torch.count_nonzero(prepared["flash_doc_segment_lengths"][4:]).item() == 0


def test_prepare_flash_attention_batch_metadata_docblock_eager_gets_pairwise_mask() -> None:
    import deberta.training.compile as compile_mod

    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.long)
    batch = {"input_ids": torch.zeros((1, 4), dtype=torch.long), "doc_ids": doc_ids}

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=False,
    )

    assert meta is None
    assert "flash_seq_lengths" not in prepared
    assert "flash_active_tokens" not in prepared
    assert tuple(prepared["attention_mask"].shape) == (1, 4, 4)
    assert torch.equal(prepared["attention_mask"], compile_mod._build_doc_block_mask(doc_ids))


@pytest.mark.parametrize("seq_len", [2048, 4096])
def test_prepare_flash_attention_batch_metadata_docblock_eager_gets_large_pairwise_mask(seq_len: int) -> None:
    import deberta.training.compile as compile_mod

    doc_ids = torch.cat(
        (
            torch.ones((1, seq_len // 2), dtype=torch.long),
            torch.full((1, seq_len - (seq_len // 2)), 2, dtype=torch.long),
        ),
        dim=1,
    )
    batch = {"input_ids": torch.zeros((1, seq_len), dtype=torch.long), "doc_ids": doc_ids}

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=False,
    )

    assert meta is None
    assert "flash_seq_lengths" not in prepared
    assert tuple(prepared["attention_mask"].shape) == (1, seq_len, seq_len)
    assert prepared["attention_mask"].ndim == 3
    assert not bool(prepared["attention_mask"][0, 0, seq_len // 2])
    assert not bool(prepared["attention_mask"][0, seq_len // 2, 0])


@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_docblock_pairwise_attention_blocks_cross_document_probs_on_cpu(seq_len: int) -> None:
    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention
    from deberta.modeling.mask_utils import build_doc_block_mask

    cfg = _docblock_attention_config(seq_len=seq_len)
    attention = DisentangledSelfAttention(cfg).eval()
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    doc_ids = torch.cat(
        (
            torch.ones((1, seq_len // 2), dtype=torch.long),
            torch.full((1, seq_len // 2), 2, dtype=torch.long),
        ),
        dim=1,
    )
    pairwise_mask = build_doc_block_mask(doc_ids).unsqueeze(1)

    _, probs = attention(
        hidden_states=hidden_states,
        attention_mask=pairwise_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert probs is not None
    assert float(probs[0, 0, 0, seq_len // 2 :].detach().abs().max()) == pytest.approx(0.0)
    assert float(probs[0, 0, seq_len // 2, : seq_len // 2].detach().abs().max()) == pytest.approx(0.0)


@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_docblock_forced_flash_eager_fallback_rebuilds_pairwise_mask_probs_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    seq_len: int,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    from deberta.modeling.mask_utils import build_doc_segment_metadata

    cfg = _docblock_attention_config(seq_len=seq_len)
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    doc_ids = torch.cat(
        (
            torch.ones((1, seq_len // 2), dtype=torch.long),
            torch.full((1, seq_len // 2), 2, dtype=torch.long),
        ),
        dim=1,
    )
    segment_offsets, segment_lengths, cu_seqlens, _ = build_doc_segment_metadata(doc_ids)

    _, probs = attention(
        hidden_states=hidden_states,
        attention_mask=doc_ids.ne(0),
        output_attentions=True,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(
            doc_segment_offsets=segment_offsets,
            doc_segment_lengths=segment_lengths,
            doc_cu_seqlens=cu_seqlens,
            active_tokens_host=seq_len,
            doc_num_segments_host=2,
            doc_max_segment_length_host=seq_len // 2,
            route_hint="docblock",
        ),
    )

    assert probs is not None
    assert float(probs[0, 0, 0, seq_len // 2 :].detach().abs().max()) == pytest.approx(0.0)
    assert float(probs[0, 0, seq_len // 2, : seq_len // 2].detach().abs().max()) == pytest.approx(0.0)


def test_prepare_flash_attention_batch_metadata_routes_docblock_bias() -> None:
    import deberta.training.compile as compile_mod

    batch = {
        "input_ids": torch.zeros((2, 5), dtype=torch.long),
        "doc_ids": torch.tensor(
            [
                [1, 1, 2, 2, 0],
                [1, 2, 2, 0, 0],
            ],
            dtype=torch.long,
        ),
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
        flash_cfg={"docblock_bias_seq_len": 5},
    )

    assert meta is not None
    assert meta.normalized_route_hint() == "docblock_bias"
    assert "doc_ids" not in prepared
    assert "flash_doc_segment_offsets" not in prepared
    assert "flash_doc_segment_lengths" not in prepared
    assert "flash_doc_cu_seqlens" not in prepared
    assert tuple(prepared["attention_mask"].shape) == (2, 5, 5)
    assert prepared["attention_mask"].dtype == torch.bool
    assert torch.equal(prepared["flash_seq_lengths"], torch.tensor([4, 3], dtype=torch.int32))
    assert prepared["flash_active_tokens"] == 7
    assert meta.active_tokens_host == 7


def test_prepare_flash_attention_batch_metadata_docblock_host_stats() -> None:
    import deberta.training.compile as compile_mod

    batch = {
        "input_ids": torch.zeros((2, 5), dtype=torch.long),
        "doc_ids": torch.tensor(
            [
                [1, 1, 2, 2, 0],
                [1, 2, 2, 0, 0],
            ],
            dtype=torch.long,
        ),
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )

    assert prepared["flash_doc_num_segments"] == 4
    assert prepared["flash_doc_max_seqlen"] == 2
    assert meta is not None
    assert meta.doc_num_segments_host == 4
    assert meta.doc_max_segment_length_host == 2
    assert meta.active_tokens_scalar is not None
    assert meta.active_tokens_scalar.device.type == "cpu"
    assert meta.active_tokens_scalar.ndim == 0
    assert int(meta.active_tokens_scalar) == 7
    assert meta.doc_num_segments_scalar is not None
    assert meta.doc_num_segments_scalar.device.type == "cpu"
    assert meta.doc_num_segments_scalar.ndim == 0
    assert int(meta.doc_num_segments_scalar) == 4
    assert meta.doc_max_segment_length_scalar is not None
    assert meta.doc_max_segment_length_scalar.device.type == "cpu"
    assert meta.doc_max_segment_length_scalar.ndim == 0
    assert int(meta.doc_max_segment_length_scalar) == 2


def test_prepare_flash_attention_batch_metadata_respects_force_varlen() -> None:
    import deberta.training.compile as compile_mod

    batch = {
        "input_ids": torch.zeros((2, 1024), dtype=torch.long),
        "attention_mask": torch.tensor(
            [
                [True, True, False, False],
                [True, True, True, False],
            ],
            dtype=torch.bool,
        ),
    }
    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
        flash_cfg={"force_varlen": True},
    )
    assert meta is not None
    assert meta.normalized_route_hint() == "varlen"
    assert torch.equal(prepared["flash_seq_lengths"], torch.tensor([2, 3], dtype=torch.int32))
    assert prepared["flash_active_tokens"] == 5
    assert meta.active_tokens_host == 5
    assert meta.active_tokens_scalar is not None
    assert meta.active_tokens_scalar.device.type == "cpu"
    assert meta.active_tokens_scalar.ndim == 0
    assert int(meta.active_tokens_scalar) == 5


def test_flash_attention_docblock_path_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(
        attention,
        "_eager_fallback_attention_mask",
        lambda **kwargs: pytest.fail("docblock flash fast path should not build eager fallback masks"),
    )
    monkeypatch.setattr(attention_mod, "flashdeberta_docblock_import_error", lambda: None)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_docblock_available", lambda: True)
    seen: dict[str, torch.Tensor] = {}

    def _fake_docblock_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_lengths: torch.Tensor,
        cu_seqlens: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
        position_buckets: int,
        max_relative_distance: int,
        num_segments: int | torch.Tensor,
        max_seqlen: int | torch.Tensor,
        total_tokens: int | torch.Tensor,
        causal: bool,
    ) -> torch.Tensor:
        del (
            key_layer,
            value_layer,
            pos_key,
            pos_query,
            sm_scale,
            position_buckets,
            max_relative_distance,
            causal,
        )
        seen["segment_offsets"] = segment_offsets
        seen["segment_lengths"] = segment_lengths
        seen["cu_seqlens"] = cu_seqlens
        seen["num_segments"] = torch.as_tensor(num_segments).cpu()
        seen["max_seqlen"] = torch.as_tensor(max_seqlen).cpu()
        seen["total_tokens"] = torch.as_tensor(total_tokens).cpu()
        return torch.zeros_like(query_layer)

    monkeypatch.setattr(attention_mod, "flashdeberta_docblock", _fake_docblock_wrapper)

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    attention_mod.reset_flashdeberta_stats()
    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(
            doc_segment_offsets=torch.tensor([0, 2], dtype=torch.int32),
            doc_segment_lengths=torch.tensor([2, 1], dtype=torch.int32),
            doc_cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int32),
            active_tokens_host=3,
            doc_num_segments_host=2,
            doc_max_segment_length_host=2,
            active_tokens_scalar=torch.tensor(3, dtype=torch.int32),
            doc_num_segments_scalar=torch.tensor(2, dtype=torch.int32),
            doc_max_segment_length_scalar=torch.tensor(2, dtype=torch.int32),
            route_hint="docblock",
        ),
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert torch.equal(seen["segment_offsets"], torch.tensor([0, 2], dtype=torch.int32))
    assert torch.equal(seen["segment_lengths"], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(seen["cu_seqlens"], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert seen["num_segments"].item() == 2
    assert seen["max_seqlen"].item() == 2
    assert seen["total_tokens"].item() == 3

    stats = attention_mod.flashdeberta_stats_snapshot()
    assert stats["forward_calls"] == 1
    assert stats["flash_eligible_calls"] == 1
    assert stats["flash_docblock_calls"] == 1
    assert stats.get("flash_varlen_calls", 0) == 0
    assert stats.get("flash_fixed_calls", 0) == 0
    assert stats.get("fallback_calls", 0) == 0


def test_flash_attention_docblock_bias_path_records_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, _ = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention_mod, "flashdeberta_bias_import_error", lambda: None)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_bias_available", lambda: True)
    seen: dict[str, torch.Tensor] = {}

    def _fake_bias_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        bucket_index: torch.Tensor,
        keep_mask: torch.Tensor | None,
        bias_scale: float,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        del key_layer, value_layer, pos_key, pos_query, bucket_index, bias_scale, sm_scale, causal
        seen["keep_mask"] = keep_mask
        out = torch.empty_like(query_layer)
        for head_idx in range(int(query_layer.shape[1])):
            for seq_idx in range(int(query_layer.shape[2])):
                out[:, head_idx, seq_idx, :] = float(head_idx * 100 + seq_idx)
        return out

    monkeypatch.setattr(attention_mod, "flashdeberta_bias_from_positions", _fake_bias_wrapper)

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, False],
                [False, False, False, False],
            ]
        ],
        dtype=torch.bool,
    )
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    attention_mod.reset_flashdeberta_stats()
    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(route_hint="docblock_bias"),
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    for seq_idx in range(4):
        for head_idx in range(cfg.num_attention_heads):
            start = head_idx * head_dim
            end = start + head_dim
            assert torch.all(output[0, seq_idx, start:end].eq(float(head_idx * 100 + seq_idx)))
    assert seen["keep_mask"] is not None
    assert tuple(seen["keep_mask"].shape) == (1, 1, 4, 4)
    stats = attention_mod.flashdeberta_stats_snapshot()
    assert stats["forward_calls"] == 1
    assert stats["flash_eligible_calls"] == 1
    assert stats["flash_docblock_bias_calls"] == 1
    assert stats.get("flash_docblock_calls", 0) == 0
    assert stats.get("fallback_calls", 0) == 0


def test_build_dense_flash_bias_matches_reference() -> None:
    import deberta.modeling.flashdeberta_attention as attention_mod

    batch_size = 2
    num_heads = 3
    seq_len = 4
    buckets = 7

    pos_key = torch.randn((batch_size, num_heads, seq_len, buckets), dtype=torch.float32)
    pos_query = torch.randn((batch_size, num_heads, seq_len, buckets), dtype=torch.float32)
    bucket_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
            [2, 3, 0, 5],
            [3, 4, 5, 0],
        ],
        dtype=torch.int64,
    )
    keep_mask = torch.tensor(
        [
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ],
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ],
        ],
        dtype=torch.bool,
    )

    index = bucket_index.view(1, 1, seq_len, seq_len).expand(batch_size, num_heads, seq_len, seq_len)
    reference = torch.gather(pos_key, dim=-1, index=index)
    reverse = (
        bucket_index.t()
        .contiguous()
        .view(1, 1, seq_len, seq_len)
        .expand(batch_size, num_heads, seq_len, seq_len)
    )
    reference = reference + torch.gather(pos_query, dim=-1, index=reverse).transpose(-1, -2)
    reference = reference.masked_fill(~keep_mask, -1.0e4)

    actual = attention_mod._build_dense_flash_bias(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
    )

    assert torch.equal(actual, reference)


def test_flashdeberta_dense_bias_wrapper_matches_scaled_reference() -> None:
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    batch_size = 2
    num_heads = 3
    seq_len = 4
    buckets = 7
    scale = 0.125

    pos_key = torch.randn((batch_size, num_heads, seq_len, buckets), dtype=torch.float32)
    pos_query = torch.randn((batch_size, num_heads, seq_len, buckets), dtype=torch.float32)
    bucket_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 4],
            [2, 3, 0, 5],
            [3, 4, 5, 0],
        ],
        dtype=torch.int64,
    )
    keep_mask = torch.tensor(
        [
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ],
            [
                [
                    [True, False, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ],
        ],
        dtype=torch.bool,
    )

    expected = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    actual = dense_bias_mod.flashdeberta_dense_bias(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )

    assert torch.equal(actual, expected)


def test_dense_bias_bucket_reduce_matches_scatter_reference() -> None:
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    torch.manual_seed(0)
    bucket_index = torch.tensor(
        [
            [1, 1, 2, 3],
            [0, 1, 1, 2],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=torch.int64,
    )
    grad = torch.randn((2, 3, 4, 4), dtype=torch.float32)

    actual = dense_bias_mod._dense_bucket_reduce(
        grad=grad,
        bucket_index=bucket_index,
        num_buckets=4,
        output_dtype=torch.float32,
    )
    gather_index = bucket_index.view(1, 1, 4, 4).expand(grad.shape[0], grad.shape[1], -1, -1)
    expected = torch.zeros((2, 3, 4, 4), dtype=torch.float32).scatter_add_(-1, gather_index, grad)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("use_pos_key", "use_pos_query", "use_mask"),
    [(True, True, True), (True, False, False), (False, True, True)],
)
def test_position_bias_dense_grad_reduction_matches_autograd(
    use_pos_key: bool,
    use_pos_query: bool,
    use_mask: bool,
) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    torch.manual_seed(0)
    batch_size, num_heads, seq_len, num_buckets = 2, 3, 4, 6
    scale = 0.7
    bucket_index = torch.tensor(
        [
            [2, 2, 3, 4],
            [1, 2, 2, 3],
            [1, 1, 2, 2],
            [0, 1, 1, 2],
        ],
        dtype=torch.int64,
    )
    keep_mask = None
    if use_mask:
        keep_mask = torch.tensor(
            [
                [
                    [
                        [True, True, False, False],
                        [True, True, False, False],
                        [False, False, True, True],
                        [False, False, True, True],
                    ]
                ],
                [
                    [
                        [True, False, False, False],
                        [False, True, False, False],
                        [False, False, True, False],
                        [False, False, False, True],
                    ]
                ],
            ],
            dtype=torch.bool,
        )
    pos_key = (
        torch.randn((batch_size, num_heads, seq_len, num_buckets), requires_grad=True)
        if use_pos_key
        else None
    )
    pos_query = (
        torch.randn((batch_size, num_heads, seq_len, num_buckets), requires_grad=True)
        if use_pos_query
        else None
    )
    d_bias = torch.randn((batch_size, num_heads, seq_len, seq_len), dtype=torch.float32)

    bias = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    (bias * d_bias).sum().backward()

    actual_key, actual_query = bias_mod._position_bias_backward_from_dense_grad(
        d_bias=d_bias,
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )

    if use_pos_key:
        assert actual_key is not None
        assert torch.allclose(actual_key, pos_key.grad, atol=1e-6, rtol=1e-6)
    else:
        assert actual_key is None
    if use_pos_query:
        assert actual_query is not None
        assert torch.allclose(actual_query, pos_query.grad, atol=1e-6, rtol=1e-6)
    else:
        assert actual_query is None


def test_position_bias_backward_fake_outputs_use_input_shapes() -> None:
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")

    import deberta.modeling.flashdeberta_bias_op as bias_mod

    if bias_mod._FLASHDEBERTA_POSITION_BIAS_BWD_CUSTOM_OP is None:
        pytest.skip("Compiled position-bias custom op is unavailable in this environment.")

    with fake_tensor_mod.FakeTensorMode():
        q = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        k = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        v = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        grad_out = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        lse = torch.empty((2, 4, 3), device="cuda", dtype=torch.float32)
        bias = torch.empty((2, 4, 3, 3), device="cuda", dtype=torch.bfloat16)
        pos_key = torch.empty((2, 3, 4, 7), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
        pos_query = torch.empty((2, 3, 4, 7), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
        bucket_index = torch.empty((3, 3), device="cuda", dtype=torch.int64)
        keep_mask = torch.empty((2, 1, 3, 3), device="cuda", dtype=torch.bool)

        dq, dk, dv, dpos_key, dpos_query = bias_mod._FLASHDEBERTA_POSITION_BIAS_BWD_CUSTOM_OP(
            grad_out,
            q,
            k,
            v,
            pos_key,
            pos_query,
            bucket_index,
            keep_mask,
            bias,
            out,
            lse,
            0.5,
            0.5,
            False,
            True,
            True,
            True,
        )

    assert tuple(dq.shape) == tuple(q.shape)
    assert tuple(dk.shape) == tuple(k.shape)
    assert tuple(dv.shape) == tuple(v.shape)
    assert tuple(dpos_key.shape) == tuple(pos_key.shape)
    assert tuple(dpos_query.shape) == tuple(pos_query.shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for fused position-bias parity.")
@pytest.mark.parametrize("use_mask", [False, True])
def test_position_bias_attention_cuda_matches_dense_composition(use_mask: bool) -> None:
    import deberta.modeling.flashdeberta_attention as attention_mod
    import deberta.modeling.flashdeberta_bias_op as bias_mod
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    if bias_mod._FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP is None:
        pytest.skip("Compiled position-bias custom op is unavailable in this environment.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, num_heads, seq_len, head_dim, num_buckets = 1, 2, 32, 64, 16
    dtype = torch.bfloat16
    scale = float(head_dim) ** -0.5
    bucket_index = attention_mod._dense_bucket_index_tensor(
        seq_len=seq_len,
        position_buckets=num_buckets // 2,
        max_relative_distance=seq_len,
        device=device,
    )
    keep_mask = None
    if use_mask:
        doc_ids = torch.tensor([[1] * 12 + [2] * 12 + [0] * 8], device=device)
        keep = doc_ids.ne(0)
        keep_mask = (
            keep[:, None, :, None]
            & keep[:, None, None, :]
            & doc_ids[:, None, :, None].eq(doc_ids[:, None, None, :])
        )

    def _leaf(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype).requires_grad_()

    q = _leaf((batch_size, num_heads, seq_len, head_dim))
    k = _leaf((batch_size, num_heads, seq_len, head_dim))
    v = _leaf((batch_size, num_heads, seq_len, head_dim))
    pos_key = _leaf((batch_size, num_heads, seq_len, num_buckets))
    pos_query = _leaf((batch_size, num_heads, seq_len, num_buckets))

    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    v_ref = v.detach().clone().requires_grad_()
    pos_key_ref = pos_key.detach().clone().requires_grad_()
    pos_query_ref = pos_query.detach().clone().requires_grad_()

    ref_bias = dense_bias_mod.flashdeberta_dense_bias(
        pos_key=pos_key_ref,
        pos_query=pos_query_ref,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    ref_out = bias_mod.flashdeberta_bias(
        query_layer=q_ref,
        key_layer=k_ref,
        value_layer=v_ref,
        bias=ref_bias,
        sm_scale=scale,
        causal=False,
    )
    fused_out = bias_mod.flashdeberta_bias_from_positions(
        query_layer=q,
        key_layer=k,
        value_layer=v,
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        bias_scale=scale,
        sm_scale=scale,
        causal=False,
    )

    grad = torch.randn_like(ref_out)
    (ref_out.float() * grad.float()).sum().backward()
    (fused_out.float() * grad.float()).sum().backward()

    torch.testing.assert_close(fused_out, ref_out, atol=2e-2, rtol=2e-2)
    for actual, expected in (
        (q.grad, q_ref.grad),
        (k.grad, k_ref.grad),
        (v.grad, v_ref.grad),
        (pos_key.grad, pos_key_ref.grad),
        (pos_query.grad, pos_query_ref.grad),
    ):
        torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for saved-tensor hook coverage.")
def test_position_bias_attention_cuda_saves_dense_bias_aux_tensor() -> None:
    import deberta.modeling.flashdeberta_attention as attention_mod
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    if bias_mod._FLASHDEBERTA_POSITION_BIAS_CUSTOM_OP is None:
        pytest.skip("Compiled position-bias custom op is unavailable in this environment.")

    torch.manual_seed(1)
    device = torch.device("cuda")
    batch_size, num_heads, seq_len, head_dim, num_buckets = 1, 2, 32, 64, 16
    dtype = torch.bfloat16
    scale = float(head_dim) ** -0.5
    q = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype).requires_grad_()
    k = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype).requires_grad_()
    v = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype).requires_grad_()
    pos_key = torch.randn(
        (batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype
    ).requires_grad_()
    pos_query = torch.randn(
        (batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype
    ).requires_grad_()
    bucket_index = attention_mod._dense_bucket_index_tensor(
        seq_len=seq_len,
        position_buckets=num_buckets // 2,
        max_relative_distance=seq_len,
        device=device,
    )
    saved_shapes: list[tuple[int, ...]] = []

    def _pack(tensor: torch.Tensor) -> torch.Tensor:
        saved_shapes.append(tuple(int(dim) for dim in tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(_pack, lambda tensor: tensor):
        out = bias_mod.flashdeberta_bias_from_positions(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=None,
            bias_scale=scale,
            sm_scale=scale,
            causal=False,
        )
        out.float().sum().backward()

    assert (batch_size, num_heads, seq_len, seq_len) in saved_shapes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real-kernel leakage checks.")
@pytest.mark.parametrize("route", ["docblock", "docblock_bias"])
def test_docblock_real_kernel_blocks_cross_document_gradients_on_cuda(route: str) -> None:
    """Gradient-isolation leakage check on the actual Triton doc-block routes.

    Doc-1 query outputs must carry exactly zero gradient back to doc-2 hidden
    states; any nonzero gradient means cross-document attention leaked. The
    route counters prove the flash kernel actually ran instead of a silent
    eager fallback (which would also block correctly and mask a kernel bug).

    Other tests in this file reload the flash module tree against fake
    flashdeberta packages and leave those reloaded twins cached, so this test
    re-imports a clean real-kernel tree and restores the prior modules after.
    """

    import dataclasses

    affected_prefixes = ("deberta.modeling.flashdeberta_", "flashdeberta")
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith(affected_prefixes)}
    for name in saved:
        sys.modules.pop(name, None)
    try:
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        if attention_mod.flashdeberta_fixed_import_error() is not None:
            pytest.skip("FlashDeBERTa kernels are unavailable in this environment.")
        # The fresh module instance is discarded in the finally block, so
        # mutating its runtime config does not need monkeypatch cleanup.
        attention_mod._RUNTIME_CONFIG = dataclasses.replace(
            attention_mod._RUNTIME_CONFIG, enable_debug_stats=True
        )
        _run_docblock_real_kernel_leak_check(attention_mod=attention_mod, route=route)
    finally:
        for name in [n for n in sys.modules if n.startswith(affected_prefixes)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


def _run_docblock_real_kernel_leak_check(*, attention_mod, route: str) -> None:
    from deberta.modeling.deberta_v2_native import DebertaV2Config
    from deberta.modeling.mask_utils import (
        build_doc_block_mask,
        build_doc_segment_metadata,
        doc_segment_metadata_host_stats,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 1024
    boundary = seq_len // 2
    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=seq_len,
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=32,
        max_relative_positions=seq_len,
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        position_biased_input=False,
    )
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).to(device=device, dtype=dtype).eval()

    doc_ids = torch.cat(
        (
            torch.ones((1, boundary), dtype=torch.long),
            torch.full((1, seq_len - boundary), 2, dtype=torch.long),
        ),
        dim=1,
    )
    if route == "docblock_bias":
        attention_mask: torch.Tensor = build_doc_block_mask(doc_ids.to(device=device))
        flash_meta = FlashBatchMeta(
            seq_lengths=doc_ids.ne(0).sum(-1, dtype=torch.int32).to(device=device),
            active_tokens_host=seq_len,
            route_hint="docblock_bias",
        )
    else:
        segment_offsets, segment_lengths, cu_seqlens, active_tokens = build_doc_segment_metadata(doc_ids)
        num_segments, max_seqlen, _ = doc_segment_metadata_host_stats(
            segment_lengths,
            active_tokens=active_tokens,
        )
        attention_mask = doc_ids.ne(0).to(device=device)
        flash_meta = FlashBatchMeta(
            seq_lengths=doc_ids.ne(0).sum(-1, dtype=torch.int32).to(device=device),
            doc_segment_offsets=segment_offsets.to(device=device),
            doc_segment_lengths=segment_lengths.to(device=device),
            doc_cu_seqlens=cu_seqlens.to(device=device),
            active_tokens_host=active_tokens,
            doc_num_segments_host=num_segments,
            doc_max_segment_length_host=max_seqlen,
            route_hint="docblock",
        )

    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), device=device, dtype=dtype).requires_grad_()
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), device=device, dtype=dtype)

    attention_mod.reset_flashdeberta_stats()
    output, _ = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
        flash_meta=flash_meta,
    )
    stats = attention_mod.flashdeberta_stats_snapshot()
    expected_counter = "flash_docblock_bias_calls" if route == "docblock_bias" else "flash_docblock_calls"
    assert stats.get(expected_counter, 0) >= 1, f"flash {route} route did not run: stats={stats}"
    assert stats.get("fallback_calls", 0) == 0, f"unexpected eager fallback: stats={stats}"

    output[0, :boundary].float().square().sum().backward()

    assert hidden_states.grad is not None
    doc2_grad = hidden_states.grad[0, boundary:]
    assert torch.all(doc2_grad == 0), (
        f"cross-document gradient leak on {route}: max abs doc-2 grad {float(doc2_grad.abs().max()):.3e}"
    )
    doc1_grad = hidden_states.grad[0, :boundary]
    assert float(doc1_grad.abs().max()) > 0.0


def test_varlen_bwd_config_resolution_falls_back_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    monkeypatch.setattr(varlen_mod, "_varlen_repo_tuned_bwd_config", lambda **kwargs: None)
    monkeypatch.setattr(varlen_mod, "_get_bwd_config_varlen_lowlevel", lambda **kwargs: (16, 16, 1, 2))

    kv_config = varlen_mod._resolve_varlen_bwd_kernel_config(
        kind="kv",
        total_tokens_q=3500,
        total_tokens_k=3500,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        batch_size=2,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    q_config = varlen_mod._resolve_varlen_bwd_kernel_config(
        kind="q",
        total_tokens_q=3500,
        total_tokens_k=3500,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        batch_size=2,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert kv_config == (16, 16, 1, 2)
    assert q_config == (16, 16, 1, 2)


def test_varlen_repo_tuned_bwd_config_uses_density_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    monkeypatch.setattr(varlen_mod, "_varlen_device_capability", lambda device: (12, 0))

    sparse_cfg = varlen_mod._varlen_repo_tuned_bwd_config(
        kind="kv",
        seq_len=2048,
        total_tokens=1800,
        batch_size=2,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    long_cfg = varlen_mod._varlen_repo_tuned_bwd_config(
        kind="q",
        seq_len=4096,
        total_tokens=3500,
        batch_size=1,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    assert sparse_cfg == (64, 32, 2, 4)
    assert long_cfg == (64, 64, 3, 8)


def test_varlen_backward_fake_outputs_use_contiguous_padded_layout() -> None:
    fake_tensor_mod = pytest.importorskip("torch._subclasses.fake_tensor")

    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    if varlen_mod._FLASHDEBERTA_VARLEN_BWD_CUSTOM_OP is None:
        pytest.skip("Compiled varlen custom op is unavailable in this environment.")

    with fake_tensor_mod.FakeTensorMode():
        q = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        k = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        v = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        grad_out = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((2, 4, 3, 5), device="cuda", dtype=torch.bfloat16)
        lse = torch.empty((2, 4, 3), device="cuda", dtype=torch.float32)
        mask = torch.ones((2, 4), device="cuda", dtype=torch.bool)
        pos_key = torch.empty((2, 3, 4, 7), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)
        pos_query = torch.empty((2, 3, 4, 7), device="cuda", dtype=torch.bfloat16).permute(0, 2, 1, 3)

        _, _, _, dpos_key, dpos_query = varlen_mod._FLASHDEBERTA_VARLEN_BWD_CUSTOM_OP(
            grad_out,
            q,
            k,
            v,
            mask,
            out,
            lse,
            pos_key,
            pos_query,
            1.0,
            32,
            128,
            False,
        )

    assert dpos_key is not None
    assert dpos_query is not None
    assert dpos_key.shape == pos_key.shape
    assert dpos_query.shape == pos_query.shape
    assert dpos_key.stride() == (84, 21, 7, 1)
    assert dpos_query.stride() == (84, 21, 7, 1)


def test_fixed_repo_tuned_config_matches_sm120_dense_1024(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_fixed_op as fixed_mod

    monkeypatch.setattr(fixed_mod.torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (12, 0))

    assert fixed_mod._fixed_repo_tuned_config(
        kind="fwd",
        query_len=1024,
        key_len=1024,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ) == (64, 64, 2, 4)

    assert fixed_mod._fixed_repo_tuned_config(
        kind="bwd",
        query_len=1024,
        key_len=1024,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ) == (16, 16, 1, 2)

    assert (
        fixed_mod._fixed_repo_tuned_config(
            kind="bwd",
            query_len=2048,
            key_len=2048,
            head_dim=64,
            causal=False,
            disentangled=True,
            att_span=256,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )
        is None
    )


def test_bias_bwd_config_resolution_falls_back_to_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    monkeypatch.setattr(bias_mod, "_bias_repo_tuned_config", lambda **kwargs: None)
    monkeypatch.setattr(bias_mod, "_get_bwd_config_bias_lowlevel", lambda *args, **kwargs: (16, 16, 1, 2))

    kv_config = bias_mod._resolve_bias_bwd_kernel_config(
        kind="kv",
        batch_size=4,
        num_heads=12,
        query_len=1024,
        key_len=1024,
        head_dim=64,
        causal=False,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    q_config = bias_mod._resolve_bias_bwd_kernel_config(
        kind="q",
        batch_size=4,
        num_heads=12,
        query_len=1024,
        key_len=1024,
        head_dim=64,
        causal=False,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert kv_config == (16, 16, 1, 2)
    assert q_config == (16, 16, 1, 2)


def test_bias_repo_tuned_config_is_table_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    monkeypatch.setattr(bias_mod.torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (12, 0))

    base = dict(
        batch_size=2,
        num_heads=12,
        query_len=1024,
        key_len=1024,
        head_dim=64,
        causal=False,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    assert bias_mod._bias_repo_tuned_config(kind="fwd", **base) == (128, 64, 3, 4)
    assert bias_mod._bias_repo_tuned_config(kind="bwd", **base) == (64, 64, 2, 4)
    assert bias_mod._bias_repo_tuned_config(kind="bwd_kv", **base) == (64, 64, 2, 4)
    assert bias_mod._bias_repo_tuned_config(kind="bwd_q", **base) == (64, 64, 2, 4)

    assert (
        bias_mod._bias_repo_tuned_config(
            kind="fwd",
            **{
                **base,
                "query_len": 2048,
                "key_len": 2048,
            },
        )
        is None
    )


def test_bias_backward_dispatches_to_specialized_docblock_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    monkeypatch.setattr(bias_mod, "_flash_attn_v2_bwd_bias_lowlevel", object())
    monkeypatch.setattr(
        bias_mod,
        "_should_use_specialized_docblock_bias_backward",
        lambda **_kwargs: True,
    )
    seen: dict[str, torch.Tensor] = {}

    def _fake_specialized(**kwargs: torch.Tensor | float) -> tuple[torch.Tensor, ...]:
        seen["q"] = kwargs["q"]  # type: ignore[index]
        q = kwargs["q"]  # type: ignore[index]
        k = kwargs["k"]  # type: ignore[index]
        v = kwargs["v"]  # type: ignore[index]
        bias = kwargs["bias"]  # type: ignore[index]
        return (
            torch.ones_like(q),
            torch.ones_like(k),
            torch.ones_like(v),
            torch.ones_like(bias),
        )

    monkeypatch.setattr(bias_mod, "_bias_specialized_docblock_backward_impl", _fake_specialized)

    q = torch.randn((1, 1, 4, 2), dtype=torch.float32)
    k = torch.randn((1, 1, 4, 2), dtype=torch.float32)
    v = torch.randn((1, 1, 4, 2), dtype=torch.float32)
    bias = torch.randn((1, 1, 4, 4), dtype=torch.float32)
    out = torch.randn_like(q)
    lse = torch.randn((1, 1, 4), dtype=torch.float32)
    grad = torch.randn_like(q)

    dq, dk, dv, d_bias = bias_mod._bias_eager_backward_impl(
        grad_out=grad,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        sm_scale=0.5,
        causal=False,
    )

    assert seen["q"] is q
    assert torch.equal(dq, torch.ones_like(q))
    assert torch.equal(dk, torch.ones_like(k))
    assert torch.equal(dv, torch.ones_like(v))
    assert torch.equal(d_bias, torch.ones_like(bias))


def test_specialized_docblock_bias_policy_is_table_gated() -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import FlashKernelContext, resolve_flash_kernel_config

    base = dict(
        compute_capability=(12, 0),
        route="bias_docblock_specialized",
        kind="bwd",
        seq_len=1024,
        query_len=1024,
        key_len=1024,
        num_heads=12,
        head_dim=64,
        dtype="bfloat16",
        causal=False,
        has_mask=True,
    )

    assert resolve_flash_kernel_config(FlashKernelContext(batch_size=4, **base)) == (16, 16, 1, 2)
    assert resolve_flash_kernel_config(
        FlashKernelContext(batch_size=4, kind="bwd_kv", **{k: v for k, v in base.items() if k != "kind"})
    ) == (16, 32, 1, 4)
    assert resolve_flash_kernel_config(
        FlashKernelContext(batch_size=4, kind="bwd_q", **{k: v for k, v in base.items() if k != "kind"})
    ) == (64, 64, 2, 4)
    assert resolve_flash_kernel_config(FlashKernelContext(batch_size=5, **base)) is None


def test_dense_bias_repo_tuned_config_matches_sm120_docblock_1024(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    monkeypatch.setattr(dense_bias_mod.torch.cuda, "get_device_capability", lambda *_args, **_kwargs: (12, 0))

    assert dense_bias_mod._dense_bias_repo_tuned_config(
        batch_size=4,
        num_heads=12,
        seq_len=1024,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
        has_mask=False,
    ) == (64, 128, 2, 4)

    assert (
        dense_bias_mod._dense_bias_repo_tuned_config(
            batch_size=2,
            num_heads=12,
            seq_len=1024,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            has_mask=True,
        )
        is None
    )


def test_encoder_compile_hidden_state_snapshots_clone_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from deberta.modeling import deberta_v2_native as dv2
    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = _small_deberta_config()
    model = DebertaV2Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    monkeypatch.setattr(dv2, "_is_torch_compiling", lambda: True)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == int(cfg.num_hidden_layers) + 1
    assert outputs.hidden_states[0].data_ptr() != outputs.hidden_states[1].data_ptr()
    assert outputs.hidden_states[-1].data_ptr() != outputs.last_hidden_state.data_ptr()
    torch.testing.assert_close(outputs.hidden_states[-1], outputs.last_hidden_state)


def test_native_model_forward_remains_valid_after_flash_patch_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    _, patch_mod = _reload_flash_modules()
    pytest.importorskip("transformers")

    from deberta.modeling.deberta_v2_native import DebertaV2Model

    patch_mod.disable_flashdeberta_attention()
    patch_mod.enable_flashdeberta_attention(strict=True)

    cfg = _small_deberta_config()
    model = DebertaV2Model(cfg)
    input_ids = torch.tensor([[1, 7, 8, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert tuple(output.last_hidden_state.shape) == (1, 4, cfg.hidden_size)

    patch_mod.disable_flashdeberta_attention()


def test_docblock_varlen_backward_uses_docblock_tuning_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    seen: dict[str, str] = {}

    def _fake_resolve(context):
        seen["route"] = context.route
        seen["kind"] = context.kind
        return (16, 32, 1, 4)

    monkeypatch.setattr(varlen_mod, "resolve_flash_kernel_config", _fake_resolve)
    monkeypatch.setattr(varlen_mod, "_varlen_device_capability", lambda _device: (12, 0))
    monkeypatch.setattr(
        varlen_mod,
        "_get_bwd_config_varlen_lowlevel",
        lambda **_kwargs: pytest.fail("docblock route should resolve through the tuning table"),
    )

    assert varlen_mod._resolve_varlen_bwd_kernel_config(
        route="docblock",
        kind="kv",
        total_tokens_q=4096,
        total_tokens_k=4096,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        batch_size=8,
        head_dim=64,
        causal=False,
        disentangled=True,
        att_span=256,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    ) == (16, 32, 1, 4)
    assert seen == {"route": "docblock", "kind": "bwd_kv"}
