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
    calls = {"fixed": 0, "varlen": 0}

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

    flash_attention_mod.flash_attention_with_disentangled = _fake_flash_attention_with_disentangled
    flash_attention_varlen_mod.flash_attention_with_disentangled_varlen = (
        _fake_flash_attention_with_disentangled_varlen
    )
    flash_pkg.ops = ops_pkg  # type: ignore[attr-defined]
    ops_pkg.flash_attention = flash_attention_mod  # type: ignore[attr-defined]
    ops_pkg.flash_attention_varlen = flash_attention_varlen_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "flashdeberta", flash_pkg)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops.flash_attention", flash_attention_mod)
    monkeypatch.setitem(sys.modules, "flashdeberta.ops.flash_attention_varlen", flash_attention_varlen_mod)
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
    first_indices, first_cu, first_max = varlen_mod._get_unpad_metadata_cached(mask)
    second_indices, second_cu, second_max = varlen_mod._get_unpad_metadata_cached(mask)
    clone_indices, clone_cu, clone_max = varlen_mod._get_unpad_metadata_cached(mask.clone())

    assert calls["count"] == 2
    assert first_max == second_max == clone_max == 2
    assert first_indices.data_ptr() == second_indices.data_ptr()
    assert first_cu.data_ptr() == second_cu.data_ptr()
    assert clone_indices.data_ptr() != first_indices.data_ptr()
    assert clone_cu.data_ptr() != first_cu.data_ptr()

    varlen_mod._clear_unpad_metadata_cache()


def test_varlen_forward_aux_cache_round_trips() -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    varlen_mod._clear_forward_aux_cache()

    output = torch.randn((1, 2, 4, 8), dtype=torch.float32)
    indices = torch.tensor([0, 1, 4], dtype=torch.long)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    batch_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    seq_indices = torch.tensor([0, 1, 0], dtype=torch.long)
    q_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    k_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    v_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    out_unpad = torch.randn((3, 2, 8), dtype=torch.float32)
    lse_unpad = torch.randn((3, 2), dtype=torch.float32)
    pos_key_unpad = torch.randn((3, 2, 4), dtype=torch.float32)
    pos_query_unpad = torch.randn((3, 2, 4), dtype=torch.float32)

    varlen_mod._store_forward_aux_cache(
        output_padded=output,
        indices=indices,
        cu_seqlens=cu_seqlens,
        max_seqlen=2,
        batch_indices=batch_indices,
        seq_indices=seq_indices,
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
    assert cached.indices is indices
    assert cached.cu_seqlens is cu_seqlens
    assert cached.batch_indices is batch_indices
    assert cached.seq_indices is seq_indices
    assert cached.q_unpad is q_unpad
    assert cached.k_unpad is k_unpad
    assert cached.v_unpad is v_unpad
    assert cached.out_unpad is out_unpad
    assert cached.lse_unpad is lse_unpad
    assert cached.pos_key_unpad is pos_key_unpad
    assert cached.pos_query_unpad is pos_query_unpad
    assert varlen_mod._pop_forward_aux_cache(output) is None

    varlen_mod._clear_forward_aux_cache()


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
