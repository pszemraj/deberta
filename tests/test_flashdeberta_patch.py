"""Tests for the optional FlashDeBERTa runtime patch integration."""

from __future__ import annotations

import importlib
import sys
import types
import warnings

import pytest
import torch


def _install_fake_flashdeberta(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Install a minimal in-memory FlashDeBERTa module tree for tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    :return dict[str, int]: Mutable call counters for fake flash operators.
    """

    flash_pkg = types.ModuleType("flashdeberta")
    flash_pkg.__path__ = []  # type: ignore[attr-defined]
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


def test_enable_flashdeberta_attention_patches_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod, patch_mod = _reload_flash_modules()

    from deberta.modeling import deberta_v2_native as dv2
    from deberta.modeling import rtd

    patch_mod.disable_flashdeberta_attention()

    orig_attention = dv2.DisentangledSelfAttention
    orig_get_rel_pos = dv2.DebertaV2Encoder.get_rel_pos
    orig_emd_mask = rtd._ensure_emd_pairwise_attention_mask

    patch_mod.enable_flashdeberta_attention(strict=True)

    assert dv2.DisentangledSelfAttention is attention_mod.FlashDisentangledSelfAttention
    assert dv2.DebertaV2Encoder.get_rel_pos is not orig_get_rel_pos
    assert rtd._ensure_emd_pairwise_attention_mask is not orig_emd_mask

    cfg = _small_deberta_config()
    attention = dv2.DebertaV2Attention(cfg)
    assert isinstance(attention.self, attention_mod.FlashDisentangledSelfAttention)

    encoder = dv2.DebertaV2Encoder(cfg)
    hidden_states = torch.zeros((1, 4, cfg.hidden_size))
    relative_pos = torch.ones((4, 4), dtype=torch.long)

    assert encoder.get_rel_pos(hidden_states) is None
    assert encoder.get_rel_pos(hidden_states, relative_pos=relative_pos) is relative_pos

    broadcast_mask = rtd._ensure_emd_pairwise_attention_mask(torch.tensor([[1, 1, 0]], dtype=torch.long))
    assert broadcast_mask.dtype == torch.bool
    assert tuple(broadcast_mask.shape) == (1, 1, 1, 3)
    assert torch.equal(broadcast_mask[0, 0, 0], torch.tensor([True, True, False]))

    pairwise = torch.tensor([[[True, False], [False, True]]], dtype=torch.bool)
    pairwise_mask = rtd._ensure_emd_pairwise_attention_mask(pairwise)
    assert tuple(pairwise_mask.shape) == (1, 1, 2, 2)
    assert torch.equal(pairwise_mask[:, 0], pairwise)

    patch_mod.disable_flashdeberta_attention()

    assert dv2.DisentangledSelfAttention is orig_attention
    assert dv2.DebertaV2Encoder.get_rel_pos is orig_get_rel_pos
    assert rtd._ensure_emd_pairwise_attention_mask is orig_emd_mask


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
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    monkeypatch.setenv("FLASHDEBERTA_DEBUG_STATS", "1")
    monkeypatch.setenv("FLASHDEBERTA_FORCE_VARLEN", "1")
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
    monkeypatch.delenv("FLASHDEBERTA_FORCE_VARLEN", raising=False)
    monkeypatch.delenv("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", raising=False)
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
    monkeypatch.delenv("FLASHDEBERTA_FORCE_VARLEN", raising=False)
    monkeypatch.delenv("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", raising=False)
    attention_mod.refresh_flashdeberta_runtime_config_from_env()
    monkeypatch.setattr(attention, "_fallback_reason", lambda **kwargs: None)
    monkeypatch.setattr(attention, "_projected_qkv_fallback_reason", lambda **kwargs: None)
    seen: dict[str, object] = {}

    def _fake_bias_wrapper(
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        bias: torch.Tensor,
        sm_scale: float,
        causal: bool,
    ) -> torch.Tensor:
        """Return zero output while recording dense local-bias inputs."""

        del key_layer, value_layer, sm_scale, causal
        seen["bias_shape"] = tuple(bias.shape)
        seen["query_shape"] = tuple(query_layer.shape)
        return torch.zeros_like(query_layer)

    monkeypatch.setattr(attention_mod, "flashdeberta_bias", _fake_bias_wrapper)

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
    assert seen["bias_shape"] == (1, cfg.num_attention_heads, 1024, 1024)

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
    monkeypatch.delenv("FLASHDEBERTA_FORCE_VARLEN", raising=False)
    monkeypatch.delenv("FLASHDEBERTA_VARLEN_MIN_SEQ_LEN", raising=False)
    attention_mod.refresh_flashdeberta_runtime_config_from_env()

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=1024) is True

    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: False)

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=1024) is False


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


def test_varlen_forward_aux_cache_round_trips() -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    varlen_mod._clear_forward_aux_cache()

    output = torch.randn((1, 4, 2, 8), dtype=torch.float32)
    seqlens = torch.tensor([2, 1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    q_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    k_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    v_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    out_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    lse_unpad = torch.randn((3, 2), dtype=torch.float32)
    pos_key_unpad = torch.randn((3, 2, 4), dtype=torch.float32)
    pos_query_unpad = torch.randn((3, 2, 4), dtype=torch.float32)

    varlen_mod._store_forward_aux_cache(
        output_padded=output,
        seqlens=seqlens,
        cu_seqlens=cu_seqlens,
        max_seqlen=2,
        total_tokens=3,
        q_unpad=q_unpad,
        k_unpad=k_unpad,
        v_unpad=v_unpad,
        out_unpad=out_unpad,
        lse_unpad=lse_unpad,
        pos_key_unpad=pos_key_unpad,
        pos_query_unpad=pos_query_unpad,
    )

    cached = varlen_mod._pop_forward_aux_cache(output)
    assert cached is not None
    assert cached.max_seqlen == 2
    assert cached.total_tokens == 3
    assert cached.seqlens is seqlens
    assert cached.cu_seqlens is cu_seqlens
    assert cached.q_unpad is q_unpad
    assert cached.k_unpad is k_unpad
    assert cached.v_unpad is v_unpad
    assert cached.out_unpad is out_unpad
    assert cached.lse_unpad is lse_unpad
    assert cached.pos_key_unpad is pos_key_unpad
    assert cached.pos_query_unpad is pos_query_unpad
    assert varlen_mod._pop_forward_aux_cache(output) is None

    varlen_mod._clear_forward_aux_cache()


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


def test_varlen_kernel_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    for name in (
        "FLASHDEBERTA_VARLEN_BWD_BLOCK_M",
        "FLASHDEBERTA_VARLEN_BWD_BLOCK_N",
        "FLASHDEBERTA_VARLEN_BWD_NUM_STAGES",
        "FLASHDEBERTA_VARLEN_BWD_NUM_WARPS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert varlen_mod._varlen_kernel_override_from_env(kind="bwd") is None

    monkeypatch.setenv("FLASHDEBERTA_VARLEN_BWD_BLOCK_M", "64")
    monkeypatch.setenv("FLASHDEBERTA_VARLEN_BWD_BLOCK_N", "64")
    monkeypatch.setenv("FLASHDEBERTA_VARLEN_BWD_NUM_STAGES", "3")
    monkeypatch.setenv("FLASHDEBERTA_VARLEN_BWD_NUM_WARPS", "8")

    assert varlen_mod._varlen_kernel_override_from_env(kind="bwd") == (64, 64, 3, 8)


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


def test_fixed_kernel_override_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_fixed_op as fixed_mod

    for name in (
        "FLASHDEBERTA_FIXED_BWD_BLOCK_M",
        "FLASHDEBERTA_FIXED_BWD_BLOCK_N",
        "FLASHDEBERTA_FIXED_BWD_NUM_STAGES",
        "FLASHDEBERTA_FIXED_BWD_NUM_WARPS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert fixed_mod._fixed_kernel_override_from_env(kind="bwd") is None

    monkeypatch.setenv("FLASHDEBERTA_FIXED_BWD_BLOCK_M", "64")
    monkeypatch.setenv("FLASHDEBERTA_FIXED_BWD_BLOCK_N", "64")
    monkeypatch.setenv("FLASHDEBERTA_FIXED_BWD_NUM_STAGES", "3")
    monkeypatch.setenv("FLASHDEBERTA_FIXED_BWD_NUM_WARPS", "8")

    assert fixed_mod._fixed_kernel_override_from_env(kind="bwd") == (64, 64, 3, 8)


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
