"""Tests for optional FlashDeBERTa runtime integration."""

from __future__ import annotations

import importlib
import json
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch

from deberta.modeling.mask_utils import FlashBatchMeta, build_doc_block_mask, build_doc_segment_metadata


def _install_fake_flashdeberta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal in-memory FlashDeBERTa module tree for tests.

    :param pytest.MonkeyPatch monkeypatch: Pytest monkeypatch fixture.
    :return None: The fake modules are installed through ``monkeypatch``.
    """

    flash_pkg = types.ModuleType("flashdeberta")
    flash_pkg.__path__ = []  # type: ignore[attr-defined]
    ops_pkg = types.ModuleType("flashdeberta.ops")
    ops_pkg.__path__ = []  # type: ignore[attr-defined]
    flash_attention_mod = types.ModuleType("flashdeberta.ops.flash_attention")
    flash_attention_varlen_mod = types.ModuleType("flashdeberta.ops.flash_attention_varlen")
    flash_attention_bias_mod = types.ModuleType("flashdeberta.ops.flash_attention_bias")

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


def _reload_flash_modules() -> types.ModuleType:
    """Reload the FlashDeBERTa adapter to pick up test-time fake imports."""

    sys.modules.pop("deberta.modeling.flashdeberta_attention", None)
    return importlib.import_module("deberta.modeling.flashdeberta_attention")


def _restore_saved_flash_modules(
    saved: dict[str, types.ModuleType], affected_prefixes: tuple[str, ...]
) -> None:
    """Restore a snapshotted flash module tree in sys.modules and parent packages.

    Re-importing the flash tree rebinds each parent-package attribute (e.g.
    ``deberta.modeling.flashdeberta_bias_op``) to the fresh twin. Restoring
    sys.modules alone leaves ``from X.Y import name`` (parent-attribute lookup)
    and ``import X.Y`` (sys.modules lookup) resolving to different module
    objects in later tests, so configure-style module globals silently diverge.

    :param dict[str, types.ModuleType] saved: Snapshot taken before the swap.
    :param tuple[str, ...] affected_prefixes: Module-name prefixes that were swapped.
    """

    for name in [n for n in sys.modules if n.startswith(affected_prefixes)]:
        fresh = sys.modules.pop(name)
        if name not in saved:
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name) if parent_name else None
            if parent is not None and getattr(parent, child, None) is fresh:
                delattr(parent, child)
    sys.modules.update(saved)
    for name, mod in saved.items():
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child, mod)


@contextmanager
def _isolated_flash_modules() -> Iterator[None]:
    """Temporarily remove and then fully restore the imported flash module tree."""

    affected_prefixes = ("deberta.modeling.flashdeberta_", "flashdeberta")
    saved = {name: mod for name, mod in sys.modules.items() if name.startswith(affected_prefixes)}
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        _restore_saved_flash_modules(saved, affected_prefixes)


@contextmanager
def _empty_varlen_caches(varlen_mod: types.ModuleType) -> Iterator[None]:
    """Run one cache test with fresh varlen module caches."""

    caches = (
        varlen_mod._MASK_METADATA_CACHE,
        varlen_mod._CU_SEQLENS_HOST_CACHE,
        varlen_mod._MID_TENSOR_CACHE,
    )
    for cache in caches:
        cache.clear()
    try:
        yield
    finally:
        for cache in caches:
            cache.clear()


@contextmanager
def _kernel_tuning_overrides(
    tmp_path: Path,
    payload: dict[str, Any],
    *,
    filename: str = "flash_overrides.json",
) -> Iterator[Path]:
    """Write and activate one temporary kernel-tuning override table.

    :param Path tmp_path: Pytest temporary directory.
    :param dict[str, Any] payload: Override-table JSON payload.
    :param str filename: Temporary JSON filename.
    :return Iterator[Path]: Active override path, restored on context exit.
    """

    from deberta.modeling.flashdeberta_kernel_tuning import configure_flashdeberta_kernel_overrides

    override_path = tmp_path / filename
    override_path.write_text(json.dumps(payload), encoding="utf-8")
    configure_flashdeberta_kernel_overrides(str(override_path))
    try:
        yield override_path
    finally:
        configure_flashdeberta_kernel_overrides(None)


def test_flash_runtime_config_reads_supported_options() -> None:
    from deberta.modeling.flashdeberta_attention import _runtime_config_from_deberta_config

    config = types.SimpleNamespace(
        hf_flash={"docblock_bias_seq_len": "1024", "kernel_overrides_path": "custom.json"}
    )

    runtime = _runtime_config_from_deberta_config(config)

    assert runtime.docblock_bias_seq_len == 1024
    assert runtime.kernel_overrides_path == "custom.json"


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
    assert bucket == "2048_plus"
    assert flash_route_choice(policy="padding", seq_bucket=bucket) == "varlen"
    # Dense doc-block is the measured sm_120 default; other hardware and
    # capability-blind callers get the conservative ragged route.
    docblock_bucket = flash_seq_bucket(seq_len=1024)
    assert (
        flash_route_choice(policy="docblock", seq_bucket=docblock_bucket, compute_capability=(12, 0))
        == "docblock_bias"
    )
    assert (
        flash_route_choice(policy="docblock", seq_bucket=docblock_bucket, compute_capability=(9, 0)) is None
    )
    assert flash_route_choice(policy="docblock", seq_bucket=docblock_bucket) is None
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


def test_flashdeberta_kernel_overrides_same_path_reconfigure_keeps_cache(monkeypatch) -> None:
    from deberta.modeling import flashdeberta_kernel_tuning as tuning

    tuning.configure_flashdeberta_kernel_overrides(None)
    reads = {"count": 0}
    real_read = tuning._read_json

    def _counting_read(path):
        reads["count"] += 1
        return real_read(path)

    monkeypatch.setattr(tuning, "_read_json", _counting_read)
    tuning._load_tuning_payload.cache_clear()
    tuning.flash_seq_bucket.cache_clear()
    tuning.resolve_flash_kernel_config.cache_clear()

    baseline_bucket = tuning.flash_seq_bucket(seq_len=1024)
    first_reads = reads["count"]
    assert first_reads >= 1

    for _ in range(5):
        tuning.configure_flashdeberta_kernel_overrides(None)
        assert tuning.flash_seq_bucket(seq_len=1024) == baseline_bucket
    assert reads["count"] == first_reads


def test_flashdeberta_shape_keyed_tuning_caches_are_bounded() -> None:
    from deberta.modeling import flashdeberta_kernel_tuning as tuning

    tuning.configure_flashdeberta_kernel_overrides(None)
    tuning.flash_seq_bucket.cache_clear()
    tuning.resolve_flash_kernel_config.cache_clear()

    # Cache keys carry raw per-batch token counts, so both lookups must stay
    # bounded LRUs; a plain functools.cache grows monotonically over a long
    # variably-packed training run.
    maxsize = tuning.flash_seq_bucket.cache_info().maxsize
    assert maxsize is not None
    assert tuning.resolve_flash_kernel_config.cache_info().maxsize is not None

    for total_tokens in range(1, maxsize + 129):
        tuning.flash_seq_bucket(seq_len=2048, total_tokens=total_tokens, batch_size=2)
    assert tuning.flash_seq_bucket.cache_info().currsize <= maxsize


def test_repo_tuned_config_guard_precedes_device_capability_lookup() -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import resolve_repo_tuned_config

    def _unexpected_capability_lookup() -> tuple[int, int]:
        raise AssertionError("guarded tuning lookup touched the CUDA device")

    assert (
        resolve_repo_tuned_config(
            guard=lambda: False,
            compute_capability=_unexpected_capability_lookup,
            route="dense_bias",
            kind="fwd",
            seq_len=128,
            head_dim=0,
        )
        is None
    )


def test_flashdeberta_kernel_tuning_override_path_wins(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import (
        FlashKernelContext,
        resolve_flash_kernel_config,
    )

    context = FlashKernelContext(
        compute_capability=(12, 0),
        route="varlen",
        kind="bwd_kv",
        seq_len=2048,
        total_tokens=3000,
        batch_size=2,
        head_dim=64,
    )
    shipped = resolve_flash_kernel_config(context)
    assert shipped == (64, 32, 2, 4)
    with _kernel_tuning_overrides(
        tmp_path,
        {
            "kernels": [
                {
                    "compute_capability": "sm_120",
                    "route": "varlen",
                    "kind": "bwd_kv",
                    "seq_bucket": "2048_plus",
                    "head_dim": 64,
                    "block_m": 16,
                    "block_n": 32,
                    "num_stages": 1,
                    "num_warps": 2,
                }
            ]
        },
    ):
        assert resolve_flash_kernel_config(context) == (16, 32, 1, 2)
    assert resolve_flash_kernel_config(context) == shipped


def test_flash_kernel_config_capability_precedence_and_override_append(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import (
        FlashKernelContext,
        resolve_flash_kernel_config,
    )

    # An appended hardware-agnostic kernels row (the documented override
    # workflow for giving several untuned GPUs a conservative tile) must not
    # outrank the shipped exact sm_120 row on sm_120 itself.
    def _context(capability: tuple[int, int]) -> FlashKernelContext:
        return FlashKernelContext(
            compute_capability=capability,
            route="varlen",
            kind="bwd_kv",
            seq_len=2048,
            total_tokens=3000,
            batch_size=2,
            head_dim=64,
        )

    with _kernel_tuning_overrides(
        tmp_path,
        {
            "kernels": [
                {
                    "route": "varlen",
                    "kind": "bwd_kv",
                    "seq_bucket": "2048_plus",
                    "head_dim": "*",
                    "block_m": 16,
                    "block_n": 16,
                    "num_stages": 1,
                    "num_warps": 2,
                }
            ]
        },
    ):
        assert resolve_flash_kernel_config(_context((12, 0))) == (64, 32, 2, 4)
        # Untuned hardware picks up the appended wildcard row.
        assert resolve_flash_kernel_config(_context((9, 0))) == (16, 16, 1, 2)


def test_flashdeberta_route_policy_override_path_changes_routing(tmp_path) -> None:
    from deberta.training.compile import (
        _flash_route_hint_for_docblock_batch,
        _flash_route_hint_for_padding_batch,
    )

    with _kernel_tuning_overrides(
        tmp_path,
        {
            "route_policies": {
                "padding": [{"seq_bucket": "2048_plus", "choice": "fixed"}],
                "docblock": [{"seq_bucket": "1024_exact", "choice": "docblock"}],
            }
        },
    ):
        assert _flash_route_hint_for_padding_batch(seq_len=2048, active_tokens=4096, batch_size=2) == "fixed"
        assert _flash_route_hint_for_docblock_batch(seq_len=1024) == "docblock"


def test_flash_padding_route_honors_policy_row_seq_len_bounds(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import flash_padding_route

    with _kernel_tuning_overrides(
        tmp_path,
        {"route_policies": {"padding": [{"seq_bucket": "default", "choice": "varlen", "max_seq_len": 1024}]}},
    ):
        # Both lengths resolve to the default bucket; only the in-bound length
        # may take the row.
        assert flash_padding_route(seq_len=512, total_tokens=512, batch_size=1) == "varlen"
        # Beyond the row's own bound the row must not apply; the resolver
        # falls back to the conservative default instead.
        assert flash_padding_route(seq_len=1500, total_tokens=1500, batch_size=1) == "fixed"


def test_flash_padding_route_honors_policy_row_batch_size_bounds(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import flash_padding_route

    with _kernel_tuning_overrides(
        tmp_path,
        {"route_policies": {"padding": [{"seq_bucket": "default", "choice": "varlen", "max_batch_size": 2}]}},
    ):
        assert flash_padding_route(seq_len=512, total_tokens=1024, batch_size=2) == "varlen"
        # Past the row's batch bound the row must not apply, mirroring the
        # seq_len bound semantics above.
        assert flash_padding_route(seq_len=512, total_tokens=2048, batch_size=4) == "fixed"


def test_flashdeberta_seq_bucket_override_rows_are_reachable(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import flash_seq_bucket

    with _kernel_tuning_overrides(
        tmp_path,
        {"seq_buckets": [{"name": "exact_3000", "min_seq_len": 3000, "max_seq_len": 3000}]},
    ):
        # The shipped 2048 bucket also covers this length, so the override is
        # only reachable if override rows are consulted first.
        assert flash_seq_bucket(seq_len=3000) == "exact_3000"
        # Shipped resolution order is untouched for lengths the override does not claim.
        assert flash_seq_bucket(seq_len=1024) == "1024_exact"
        assert flash_seq_bucket(seq_len=4096) == "4096_plus"


def test_flashdeberta_same_name_seq_bucket_override_replaces_shipped_row(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import flash_seq_bucket

    with _kernel_tuning_overrides(
        tmp_path,
        {"seq_buckets": [{"name": "4096_plus", "min_seq_len": 4096, "max_seq_len": 8192}]},
    ):
        # In-range lengths keep resolving to the (narrowed) bucket.
        assert flash_seq_bucket(seq_len=4096) == "4096_plus"
        assert flash_seq_bucket(seq_len=8192) == "4096_plus"
        # A narrowing override must actually narrow: lengths past the new
        # max_seq_len must not fall through to the shipped unbounded row.
        assert flash_seq_bucket(seq_len=10000) == "default"


def test_docblock_bias_route_uses_table_with_ragged_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deberta.training.compile as compile_mod
    from deberta.training.compile import _flash_route_hint_for_docblock_batch

    device = torch.device("cpu")
    monkeypatch.setattr(compile_mod, "device_compute_capability", lambda _device: (12, 0))
    assert _flash_route_hint_for_docblock_batch(seq_len=1024, device=device) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=2048, device=device) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=4096, device=device) == "docblock_bias"
    # The dense row is bounded at the largest measured length: the 4096_plus
    # bucket is open-ended, and dense docblock_bias saves a quadratic
    # (B,H,S,S) bias, so unmeasured longer contexts fall back to the ragged
    # route. The docblock_bias_seq_len knob stays the explicit opt-in.
    assert _flash_route_hint_for_docblock_batch(seq_len=8192, device=device) == "docblock"
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=8192,
            flash_cfg={"docblock_bias_seq_len": 8192},
            device=device,
        )
        == "docblock_bias"
    )
    # The dense rows also carry max_batch_size bounds: the saved bias scales
    # linearly in batch size (about 4.8 GiB per batch element at 4096), so
    # growing per-device batch size past the bound falls back to ragged
    # instead of a step-1 CUDA OOM. The knob bypasses bounds explicitly.
    assert _flash_route_hint_for_docblock_batch(seq_len=4096, batch_size=2, device=device) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=4096, batch_size=3, device=device) == "docblock"
    assert _flash_route_hint_for_docblock_batch(seq_len=1024, batch_size=8, device=device) == "docblock_bias"
    assert _flash_route_hint_for_docblock_batch(seq_len=1024, batch_size=9, device=device) == "docblock"
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=4096,
            batch_size=16,
            flash_cfg={"docblock_bias_seq_len": 4096},
            device=device,
        )
        == "docblock_bias"
    )
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=1024,
            flash_cfg={"docblock_bias_seq_len": 1024},
            device=device,
        )
        == "docblock_bias"
    )
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=1024,
            flash_cfg={"docblock_bias_seq_len": 0},
            device=device,
        )
        == "docblock"
    )

    # Hardware without capability-scoped table rows defaults to the ragged
    # route, and the explicit dense override still works there.
    monkeypatch.setattr(compile_mod, "device_compute_capability", lambda _device: (9, 0))
    assert _flash_route_hint_for_docblock_batch(seq_len=1024, device=device) == "docblock"
    assert _flash_route_hint_for_docblock_batch(seq_len=4096, device=device) == "docblock"
    assert (
        _flash_route_hint_for_docblock_batch(
            seq_len=1024,
            flash_cfg={"docblock_bias_seq_len": 1024},
            device=device,
        )
        == "docblock_bias"
    )
    # Capability-blind callers (no device) also resolve conservatively.
    assert _flash_route_hint_for_docblock_batch(seq_len=1024) == "docblock"


def test_flash_route_policy_capability_precedence_and_override_append(tmp_path) -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import flash_route_choice

    # A user override table can promote the dense route for new hardware by
    # appending a capability-scoped row.
    with _kernel_tuning_overrides(
        tmp_path,
        {
            "route_policies": {
                "docblock": [
                    {
                        "seq_bucket": "1024_exact",
                        "choice": "docblock_bias",
                        "compute_capability": "sm_90",
                    }
                ]
            }
        },
        filename="flash_routes_sm90.json",
    ):
        assert (
            flash_route_choice(policy="docblock", seq_bucket="1024_exact", compute_capability=(9, 0))
            == "docblock_bias"
        )
        assert (
            flash_route_choice(policy="docblock", seq_bucket="1024_exact", compute_capability=(12, 0))
            == "docblock_bias"
        )
        # Hardware without an exact row leaves routing to the consumer default.
        assert (
            flash_route_choice(policy="docblock", seq_bucket="1024_exact", compute_capability=(8, 0)) is None
        )

    # An appended wildcard row must not outrank the shipped exact sm_120 row.
    with _kernel_tuning_overrides(
        tmp_path,
        {"route_policies": {"docblock": [{"seq_bucket": "1024_exact", "choice": "docblock"}]}},
        filename="flash_routes_wildcard.json",
    ):
        assert (
            flash_route_choice(policy="docblock", seq_bucket="1024_exact", compute_capability=(12, 0))
            == "docblock_bias"
        )
        assert flash_route_choice(policy="docblock", seq_bucket="1024_exact") == "docblock"


def test_dense_bucket_index_reuses_native_log_bucket_math(monkeypatch: pytest.MonkeyPatch) -> None:
    import math as _math

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    _dense_bucket_index_tensor = attention_mod._dense_bucket_index_tensor

    def _prior_formula(seq_len: int, buckets: int, max_rel: int) -> torch.Tensor:
        """Re-derive the pre-refactor standalone bucket math as a reference."""

        positions = torch.arange(seq_len, dtype=torch.int32)
        rel = positions[:, None] - positions[None, :]
        mid = max(1, buckets // 2)
        sign = torch.sign(rel.to(torch.float32))
        abs_rel = rel.abs()
        near = (rel < mid) & (rel > -mid)
        abs_pos = torch.where(near, torch.full_like(abs_rel, mid - 1), abs_rel)
        log_denom = _math.log((max_rel - 1) / mid)
        log_scaled = torch.log(abs_pos.to(torch.float32).clamp_min(float(mid)) / mid) / log_denom * (mid - 1)
        log_pos = torch.ceil(log_scaled) + mid
        bucket_pos = torch.where(abs_pos <= mid, rel.to(torch.float32), log_pos * sign)
        return (bucket_pos + buckets).clamp_(0.0, float(2 * buckets - 1)).to(torch.int64)

    # In the shipped regime (max_relative_positions >= seq_len) the shared
    # native math must agree exactly with the prior standalone formula.
    for seq_len, buckets, max_rel in ((16, 8, 16), (64, 32, 64), (128, 32, 512), (32, 256, 512)):
        got = _dense_bucket_index_tensor(
            seq_len=seq_len,
            position_buckets=buckets,
            max_relative_distance=max_rel,
            device=torch.device("cpu"),
        )
        assert torch.equal(got, _prior_formula(seq_len, buckets, max_rel))

    # Beyond max_relative_positions the native math saturates like the eager
    # backbone: distant-negative offsets bucket to embedding row 1, not 0.
    saturated = _dense_bucket_index_tensor(
        seq_len=64,
        position_buckets=8,
        max_relative_distance=32,
        device=torch.device("cpu"),
    )
    assert saturated[63, 0].item() == 15
    assert saturated[0, 63].item() == 1


def test_flash_padding_route_shared_resolver() -> None:
    from deberta.modeling.flashdeberta_kernel_tuning import (
        configure_flashdeberta_kernel_overrides,
        flash_padding_route,
    )

    configure_flashdeberta_kernel_overrides(None)
    assert flash_padding_route(seq_len=1024) == "fixed"
    assert flash_padding_route(seq_len=2048, total_tokens=3000, batch_size=2) == "varlen"
    assert flash_padding_route(seq_len=2048) == flash_padding_route(
        seq_len=2048, total_tokens=3000, batch_size=2
    )


def _small_deberta_config(
    *,
    hidden_size: int = 32,
    num_attention_heads: int = 4,
    intermediate_size: int = 64,
    max_position_embeddings: int = 16,
    position_buckets: int = 8,
    max_relative_positions: int = 16,
):
    """Build a small config for native DeBERTa patch tests."""

    from deberta.modeling.deberta_v2_native import DebertaV2Config

    cfg = DebertaV2Config(
        vocab_size=64,
        hidden_size=int(hidden_size),
        num_hidden_layers=1,
        num_attention_heads=int(num_attention_heads),
        intermediate_size=int(intermediate_size),
        max_position_embeddings=int(max_position_embeddings),
        type_vocab_size=0,
        relative_attention=True,
        position_buckets=int(position_buckets),
        max_relative_positions=int(max_relative_positions),
        pos_att_type=["c2p", "p2c"],
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        position_biased_input=False,
    )
    cfg.hf_flash = {}
    return cfg


def _docblock_attention_config(*, seq_len: int):
    """Build a tiny native DeBERTa config for doc-block attention tests."""

    return _small_deberta_config(
        hidden_size=8,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=int(seq_len),
        position_buckets=4,
        max_relative_positions=8,
    )


def _flash_attention_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cfg: Any | None = None,
) -> tuple[types.ModuleType, Any, Any]:
    """Build a FlashDisentangledSelfAttention with fallbacks stubbed.

    Installs the fake flashdeberta package, reloads the flash modules, and
    stubs both fallback probes so the route under test is taken. Returns
    ``(attention_mod, attention, cfg)``.
    """

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    cfg = cfg if cfg is not None else _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)
    monkeypatch.setattr(attention, "_requires_eager_fallback", lambda **kwargs: False)
    monkeypatch.setattr(attention, "_projected_qkv_requires_eager_fallback", lambda **kwargs: False)
    return attention_mod, attention, cfg


def test_native_flash_helper_preserves_relative_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    _reload_flash_modules()

    from deberta.modeling import deberta_v2_native as dv2

    cfg = _small_deberta_config()
    cfg.hf_attention_impl = "flash"
    encoder = dv2.DebertaV2Encoder(cfg)
    hidden_states = torch.zeros((1, 4, cfg.hidden_size))
    relative_pos = torch.ones((4, 4), dtype=torch.long)

    assert encoder.get_rel_pos(hidden_states) is None
    assert encoder.get_rel_pos(hidden_states, relative_pos=relative_pos) is relative_pos


def test_native_model_flash_output_attentions_returns_prob_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    _reload_flash_modules()
    from deberta.modeling.deberta_v2_native import DebertaV2Model

    cfg = _small_deberta_config()
    cfg.hf_attention_impl = "flash"
    model = DebertaV2Model(cfg).eval()
    input_ids = torch.tensor([[1, 7, 8, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )

    assert outputs.attentions is not None
    assert len(outputs.attentions) == int(cfg.num_hidden_layers)
    assert tuple(outputs.attentions[0].shape) == (
        1,
        cfg.num_attention_heads,
        4,
        4,
    )
    assert outputs.attentions[0].numel() > 0


def test_flash_attention_pairwise_mask_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
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
        output_attentions=False,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert "kwargs" in seen
    assert seen["kwargs"]["attention_mask"] is pairwise_mask


def test_flash_attention_output_attentions_falls_back_to_eager_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()

    monkeypatch.setattr(
        attention,
        "_requires_eager_fallback",
        lambda **kwargs: pytest.fail("output_attentions=True should bypass flash eligibility checks"),
    )
    monkeypatch.setattr(
        attention_mod,
        "flashdeberta_fixed",
        lambda **kwargs: pytest.fail("output_attentions=True should use eager attention"),
    )

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
    )

    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert probs is not None
    assert tuple(probs.shape) == (1, cfg.num_attention_heads, 4, 4)


def test_flash_attention_explicit_relative_pos_falls_back_and_preserves_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit relative_pos must route to eager with the exact tensor preserved."""

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    from deberta.modeling.deberta_v2_native import build_relative_position

    cfg = _small_deberta_config()
    torch.manual_seed(0)
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    reference = attention_mod._EagerDisentangledSelfAttention(cfg).eval()
    reference.load_state_dict(attention.state_dict())

    monkeypatch.setattr(
        attention,
        "_requires_eager_fallback",
        lambda **kwargs: pytest.fail("explicit relative_pos should bypass flash eligibility checks"),
    )

    seq_len = 4
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    default_pos = build_relative_position(
        seq_len,
        seq_len,
        bucket_size=cfg.position_buckets,
        max_position=cfg.max_relative_positions,
        device=hidden_states.device,
    )
    # Deliberately non-default map so dropping the tensor would change outputs.
    shifted_pos = default_pos.roll(shifts=1, dims=-1)

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=False,
        relative_pos=shifted_pos,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None

    with torch.no_grad():
        expected, _ = reference(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=False,
            relative_pos=shifted_pos,
            rel_embeddings=rel_embeddings,
        )
        default_out, _ = reference(
            hidden_states=hidden_states,
            attention_mask=None,
            output_attentions=False,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
        )
    torch.testing.assert_close(output, expected)
    # Guard against silently reverting to the default relative-position map.
    assert not torch.allclose(output, default_out)


def test_non_prefix_padding_mask_falls_back_to_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-prefix padding masks must use eager, not collapse to prefix lengths.

    The fixed/varlen routes compress padding masks into per-example prefix
    lengths, which reinterprets masks with holes or left padding as right
    padding. Such masks are legal eager inputs and must fall back.
    """

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    cfg = _small_deberta_config()
    torch.manual_seed(0)
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    reference = attention_mod._EagerDisentangledSelfAttention(cfg).eval()
    reference.load_state_dict(attention.state_dict())

    seq_len = 4
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    # Canonical encoder-expanded (B,1,1,S) broadcast layout with a hole.
    holey_mask = torch.tensor([True, False, True, False]).view(1, 1, 1, seq_len)

    output, _ = attention(
        hidden_states=hidden_states,
        attention_mask=holey_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
    )

    with torch.no_grad():
        expected, _ = reference(
            hidden_states=hidden_states,
            attention_mask=holey_mask,
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )
    torch.testing.assert_close(output, expected)

    # A length-mismatched mask is a shape error, not a reinterpretable input:
    # Unattested masks take the static eager fallback; eager then rejects the
    # non-broadcastable shape itself.
    with pytest.raises((RuntimeError, ValueError)):
        attention(
            hidden_states=hidden_states,
            attention_mask=torch.ones((1, 1, 1, seq_len + 1), dtype=torch.bool),
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )


def test_mask_to_2d_keep_mask_rejects_length_mismatch() -> None:
    """Length-mismatched padding masks must raise instead of silently slicing."""

    from deberta.modeling.mask_utils import mask_to_2d_keep_mask

    too_long = torch.ones((1, 5), dtype=torch.bool)
    too_short = torch.ones((1, 3), dtype=torch.bool)
    broadcast_too_long = torch.ones((1, 1, 1, 5), dtype=torch.bool)

    with pytest.raises(ValueError, match="exactly match seq_len=4"):
        mask_to_2d_keep_mask(too_long, seq_len=4)
    with pytest.raises(ValueError, match="exactly match seq_len=4"):
        mask_to_2d_keep_mask(too_short, seq_len=4)
    with pytest.raises(ValueError, match="exactly match seq_len=4"):
        mask_to_2d_keep_mask(broadcast_too_long, seq_len=4)


def test_flash_attention_projected_qkv_dtype_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    bf16_qkv = torch.zeros((1, cfg.num_attention_heads, 4, head_dim), dtype=torch.bfloat16)
    fp32_qkv = bf16_qkv.float()

    assert not attention._projected_qkv_requires_eager_fallback(
        query_layer=bf16_qkv,
        key_layer=bf16_qkv,
        value_layer=bf16_qkv,
    )
    assert attention._projected_qkv_requires_eager_fallback(
        query_layer=fp32_qkv,
        key_layer=fp32_qkv,
        value_layer=fp32_qkv,
    )


def test_flash_attention_varlen_path_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _small_deberta_config()
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch, cfg=cfg)
    monkeypatch.setattr(attention_mod, "_should_use_varlen", lambda **kwargs: True)
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

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert tuple(seen["mask"].shape) == (1, 4)
    assert seen["mask"].dtype == torch.bool
    assert torch.equal(seen["mask"], attention_mask)


@pytest.mark.parametrize(
    ("import_error", "compiling"),
    [
        (ImportError("varlen-only Triton API missing"), False),
        (None, True),
    ],
)
def test_flash_attention_varlen_unavailability_falls_back_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    import_error: Exception | None,
    compiling: bool,
) -> None:
    """A healthy fixed import must not hide an unavailable selected varlen route."""

    cfg = _small_deberta_config()
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch, cfg=cfg)
    fallback_calls = 0
    eager_fallback = attention._eager_forward_fallback

    def _record_fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return eager_fallback(**kwargs)

    monkeypatch.setattr(attention, "_eager_forward_fallback", _record_fallback)
    monkeypatch.setattr(attention_mod, "flashdeberta_varlen_import_error", lambda: import_error)
    monkeypatch.setattr(attention_mod, "is_torch_compiling", lambda: compiling)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: False)
    monkeypatch.setattr(
        attention_mod,
        "flashdeberta_varlen_padded",
        lambda **kwargs: pytest.fail("unavailable varlen route must fall back before dispatch"),
    )

    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([True, True, False, False]).view(1, 1, 1, 4)
    flash_meta = FlashBatchMeta(
        seq_lengths=torch.tensor([2], dtype=torch.int32),
        route_hint="varlen",
    )
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size))

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
        flash_meta=flash_meta,
    )

    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert probs is None
    assert fallback_calls == 1


def test_flash_attention_fixed_path_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch)
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

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    assert seen["seq_lengths"] is None


def test_flash_attention_dense_local_bias_path_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _small_deberta_config()
    cfg.max_position_embeddings = 1024
    cfg.max_relative_positions = 1024
    cfg.position_buckets = 256
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch, cfg=cfg)
    # The shipped local-bias policy row is scoped to sm_120.
    monkeypatch.setattr(attention_mod, "device_compute_capability", lambda _device: (12, 0))
    attention.train()
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

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=None,
        output_attentions=False,
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


def test_local_bias_gate_uses_capability_table_independent_of_docblock_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    def _gate(hf_flash: dict[str, object], *, capability: tuple[int, int] = (12, 0)) -> bool:
        cfg = _small_deberta_config()
        cfg.hf_flash = hf_flash
        attention = attention_mod.FlashDisentangledSelfAttention(cfg)
        attention.train()
        monkeypatch.setattr(attention_mod, "device_compute_capability", lambda _device: capability)
        pos_term = torch.zeros((1, 1, 1, 1))
        return attention._should_use_local_bias(
            attention_mask=None,
            batch_size=1,
            seq_len=1024,
            pos_key=pos_term,
            pos_query=pos_term,
            device=torch.device("cpu"),
        )

    assert _gate({}) is True
    # The packed doc-block override must not leak into the plain-batch local-bias route.
    assert _gate({"docblock_bias_seq_len": 0}) is True
    assert _gate({"docblock_bias_seq_len": 2048}) is True
    # The shipped policy row is sm_120-scoped: other hardware keeps local-bias off.
    assert _gate({}, capability=(9, 0)) is False


@pytest.mark.parametrize(
    ("other", "message"),
    [
        (torch.empty((2, 4, 3)), "shape mismatch"),
        (torch.empty((2, 3), device="meta"), "device mismatch"),
    ],
)
def test_pack_layout_validation_rejects_shape_and_device_drift(other: torch.Tensor, message: str) -> None:
    from deberta.modeling.flashdeberta_op_utils import require_matching_tensor_layout

    with pytest.raises(ValueError, match=message):
        require_matching_tensor_layout(torch.empty((2, 3)), other, context="test")


def test_bounded_lru_cache_refreshes_hot_entries_before_eviction() -> None:
    from deberta.modeling.flashdeberta_op_utils import BoundedLRUCache

    cache = BoundedLRUCache[str, int](max_entries=3)
    cache["hot"] = 1
    cache["cold-1"] = 2
    cache["cold-2"] = 3

    assert cache.get("hot") == 1
    cache["new"] = 4

    assert list(cache) == ["cold-2", "hot", "new"]
    assert "cold-1" not in cache


@pytest.mark.parametrize(
    ("value", "expected"),
    [("flash", True), (" FLASH ", True), ("eager", False), (None, False)],
)
def test_flash_attention_impl_predicate_is_shared_and_normalized(value: object, expected: bool) -> None:
    from deberta.modeling.flashdeberta_op_utils import is_flash_attention_impl

    assert is_flash_attention_impl(value) is expected


def test_prefix_pack_pair_and_triple_cpu_roundtrip() -> None:
    from deberta.modeling.flashdeberta_prefix_pack import (
        prefix_pack_padded_rows_pair,
        prefix_pack_padded_rows_triple,
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


def test_segment_pack_rank4_strided_avoids_contiguous_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_segment_pack as segment_mod

    seen: dict[str, object] = {}

    def _record_launch(inputs, outputs, **kwargs):
        seen["inputs"] = inputs
        seen["outputs"] = outputs
        seen["address_mode"] = kwargs["address_mode"]

    monkeypatch.setattr(segment_mod, "_can_use_triton_segment_rank4_pack", lambda *args, **kwargs: True)
    monkeypatch.setattr(segment_mod, "launch_triton_row_copy", _record_launch)

    base = torch.empty((2, 5, 3, 9))
    tensors = (base[..., 0::3], base[..., 1::3], base[..., 2::3])
    segment_offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    segment_lengths = torch.tensor([2, 2, 1, 2], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 4, 5, 7], dtype=torch.int32)

    outputs = segment_mod.segment_pack_padded_rows_triple(
        *tensors,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=cu_seqlens,
        total_tokens=7,
    )

    assert all(not tensor.is_contiguous() for tensor in tensors)
    assert all(actual is expected for actual, expected in zip(seen["inputs"], tensors, strict=True))
    assert seen["address_mode"] == segment_mod.TRITON_ROW_COPY_SEGMENT_PACK_STRIDED
    assert all(tuple(output.shape) == (7, 3, 3) for output in outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton row-copy parity.")
def test_shared_triton_row_copy_cuda_roundtrips_prefix_and_segments() -> None:
    pytest.importorskip("triton")
    import deberta.modeling.flashdeberta_prefix_pack as prefix_mod
    import deberta.modeling.flashdeberta_segment_pack as segment_mod

    base = torch.arange(2 * 5 * 3 * 9, dtype=torch.float32, device="cuda").view(2, 5, 3, 9)
    tensors = (base[..., 0::3], base[..., 1::3], base[..., 2::3])
    assert all(not tensor.is_contiguous() for tensor in tensors)

    prefix_lengths = torch.tensor([3, 2], dtype=torch.int32, device="cuda")
    prefix_cu = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    prefix_packed = prefix_mod.prefix_pack_padded_rows_triple(
        *tensors,
        seqlens=prefix_lengths,
        cu_seqlens=prefix_cu,
        max_seqlen=3,
        total_tokens=5,
    )
    prefix_unpacked = prefix_mod.prefix_unpack_padded_rows_triple(
        *prefix_packed,
        seqlens=prefix_lengths,
        cu_seqlens=prefix_cu,
        batch_size=2,
        seq_len=5,
    )
    for source, packed, unpacked in zip(tensors, prefix_packed, prefix_unpacked, strict=True):
        expected_packed = torch.cat((source[0, :3], source[1, :2]))
        expected_unpacked = torch.zeros_like(source)
        expected_unpacked[0, :3] = source[0, :3]
        expected_unpacked[1, :2] = source[1, :2]
        assert torch.equal(packed, expected_packed)
        assert torch.equal(unpacked, expected_unpacked)

    prefix_source = tensors[0].contiguous()
    prefix_single = prefix_mod.prefix_pack_padded_rows(
        prefix_source,
        seqlens=prefix_lengths,
        cu_seqlens=prefix_cu,
        max_seqlen=3,
        total_tokens=5,
    )
    prefix_single_unpacked = prefix_mod.prefix_unpack_padded_rows(
        prefix_single,
        seqlens=prefix_lengths,
        cu_seqlens=prefix_cu,
        batch_size=2,
        seq_len=5,
    )
    assert torch.equal(prefix_single, torch.cat((prefix_source[0, :3], prefix_source[1, :2])))
    assert torch.equal(prefix_single_unpacked, prefix_unpacked[0])

    segment_offsets = torch.tensor([0, 2, 5, 6], dtype=torch.int32, device="cuda")
    segment_lengths = torch.tensor([2, 2, 1, 3], dtype=torch.int32, device="cuda")
    segment_cu = torch.tensor([0, 2, 4, 5, 8], dtype=torch.int32, device="cuda")
    segment_packed = segment_mod.segment_pack_padded_rows_triple(
        *tensors,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=segment_cu,
        total_tokens=8,
        max_segment_length=3,
    )
    segment_unpacked = segment_mod.segment_unpack_padded_rows_triple(
        *segment_packed,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=segment_cu,
        batch_size=2,
        seq_len=5,
        max_segment_length=3,
    )
    for source, packed, unpacked in zip(tensors, segment_packed, segment_unpacked, strict=True):
        expected_packed = torch.cat((source[0, :2], source[0, 2:4], source[1, :1], source[1, 1:4]))
        expected_unpacked = torch.zeros_like(source)
        expected_unpacked[0, :4] = source[0, :4]
        expected_unpacked[1, :4] = source[1, :4]
        assert torch.equal(packed, expected_packed)
        assert torch.equal(unpacked, expected_unpacked)

    segment_sources = (tensors[0].contiguous(), tensors[1].contiguous())
    segment_pair = segment_mod.segment_pack_padded_rows_pair(
        *segment_sources,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=segment_cu,
        total_tokens=8,
        max_segment_length=3,
    )
    segment_pair_unpacked = segment_mod.segment_unpack_padded_rows_pair(
        *segment_pair,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=segment_cu,
        batch_size=2,
        seq_len=5,
        max_segment_length=3,
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(segment_pair, segment_packed[:2], strict=True)
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(segment_pair_unpacked, segment_unpacked[:2], strict=True)
    )


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


def test_flash_dispatch_is_fullgraph_without_mask_scalar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real attention dispatcher must compile without reading mask scalars."""

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    cfg = _small_deberta_config()
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    hidden_states = torch.randn((1, 4, cfg.hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([True, True, False, False]).view(1, 1, 1, 4)
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    flash_meta = FlashBatchMeta(
        seq_lengths=torch.tensor([2], dtype=torch.int32),
        route_hint="fixed",
    )

    compiled = torch.compile(attention, backend="eager", fullgraph=True, dynamic=False)
    output, probs = compiled(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        rel_embeddings=rel_embeddings,
        flash_meta=flash_meta,
    )

    assert output.shape == hidden_states.shape
    assert probs is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for dispatch profiling.")
def test_flash_dispatch_profiler_has_no_local_scalar_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validated CUDA dispatch must not synchronize mask data into Python."""

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    # Use a real-kernel-compatible head dimension when an earlier CUDA test has
    # already registered the process-global fixed custom op.
    cfg = _small_deberta_config(
        hidden_size=64,
        intermediate_size=128,
    )
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention.eval()
    hidden_states = torch.randn(
        (1, 4, cfg.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    attention_mask = torch.tensor(
        [[[[True, True, False, False]]]],
        device="cuda",
    )
    rel_embeddings = torch.randn(
        (cfg.position_buckets * 2, cfg.hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
    )
    flash_meta = FlashBatchMeta(
        seq_lengths=torch.tensor([2], device="cuda", dtype=torch.int32),
        route_hint="fixed",
    )

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        output, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )
        assert output.shape == hidden_states.shape
        torch.cuda.synchronize()

    event_names = {event.key for event in profiler.key_averages()}
    assert "aten::_local_scalar_dense" not in event_names


def test_varlen_remains_enabled_while_compiling_when_custom_op_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)

    monkeypatch.setattr(attention_mod, "is_torch_compiling", lambda: True)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: True)

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=1024) is False
    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=2048) is True

    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_varlen_available", lambda: False)

    assert attention_mod._should_use_varlen(attention_mask=mask, seq_len=2048) is False


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

    monkeypatch.setattr(varlen_mod, "is_torch_compiling", lambda: True)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Inductor varlen coverage.")
def test_varlen_triton_op_compiles_with_inductor_and_backpropagates() -> None:
    pytest.importorskip("triton")
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    if not varlen_mod.flashdeberta_compiled_varlen_available():
        pytest.skip("Compile-visible FlashDeBERTa varlen kernels are unavailable.")

    def _forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the real compile-visible padded-varlen wrapper."""

        return varlen_mod.flashdeberta_varlen_padded(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            attention_mask_2d=mask,
            pos_key=None,
            pos_query=None,
            sm_scale=0.25,
            position_buckets=32,
            max_relative_distance=128,
            causal=False,
        )

    torch.manual_seed(0)
    shape = (2, 32, 2, 16)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones((2, 32), device="cuda", dtype=torch.bool)
    mask[1, 24:] = False

    compiled = torch.compile(_forward, backend="inductor", fullgraph=True, dynamic=False)
    output = compiled(q, k, v, mask)
    output.float().square().mean().backward()
    torch.cuda.synchronize()

    assert output.shape == q.shape
    assert torch.isfinite(output).all()
    for tensor in (q, k, v):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_varlen_metadata_cache_reuses_repeated_mask_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    with _empty_varlen_caches(varlen_mod):
        calls = {"count": 0}
        orig_build = varlen_mod._build_unpad_metadata

        def _counting_build(mask_2d: torch.Tensor):
            calls["count"] += 1
            return orig_build(mask_2d)

        monkeypatch.setattr(varlen_mod, "_build_unpad_metadata", _counting_build)

        mask = torch.tensor([[True, True, False, False]], dtype=torch.bool)
        first_entry = varlen_mod._get_unpad_metadata_entry(mask)
        second_entry = varlen_mod._get_unpad_metadata_entry(mask)
        clone_entry = varlen_mod._get_unpad_metadata_entry(mask.clone())

        assert calls["count"] == 2
        assert first_entry.max_seqlen == second_entry.max_seqlen == clone_entry.max_seqlen == 2
        assert first_entry.seqlens.data_ptr() == second_entry.seqlens.data_ptr()
        assert first_entry.cu_seqlens.data_ptr() == second_entry.cu_seqlens.data_ptr()
        assert clone_entry.seqlens.data_ptr() != first_entry.seqlens.data_ptr()
        assert clone_entry.cu_seqlens.data_ptr() != first_entry.cu_seqlens.data_ptr()


def test_varlen_metadata_cache_verifies_storage_identity_on_key_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    with _empty_varlen_caches(varlen_mod):
        first_mask = torch.tensor([[True, False, False, False]], dtype=torch.bool)
        second_mask = torch.tensor([[True, True, True, False]], dtype=torch.bool)
        shared_key = varlen_mod._mask_metadata_cache_key(first_mask)
        monkeypatch.setattr(varlen_mod, "_mask_metadata_cache_key", lambda _mask: shared_key)

        first_entry = varlen_mod._get_unpad_metadata_entry(first_mask)
        second_entry = varlen_mod._get_unpad_metadata_entry(second_mask)
        repeated_entry = varlen_mod._get_unpad_metadata_entry(second_mask)

        assert first_entry.max_seqlen == first_entry.total_tokens == 1
        assert second_entry.max_seqlen == second_entry.total_tokens == 3
        assert torch.equal(second_entry.seqlens, torch.tensor([3], dtype=torch.int32))
        assert repeated_entry is second_entry


def test_varlen_mid_tensor_cache_reuses_registered_cu_seqlens() -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    with _empty_varlen_caches(varlen_mod):
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


def _prefix_pack_reference(tensor: torch.Tensor, seqlens: torch.Tensor) -> torch.Tensor:
    """Pack active prefixes with simple host-side indexing."""

    pieces = [tensor[index, :length] for index, length in enumerate(seqlens.tolist()) if length > 0]
    return torch.cat(pieces, dim=0) if pieces else tensor.new_empty((0, *tensor.shape[2:]))


def _prefix_unpack_reference(
    packed: torch.Tensor,
    seqlens: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    """Scatter packed prefixes with simple host-side indexing."""

    output = packed.new_zeros((int(seqlens.numel()), int(seq_len), *packed.shape[1:]))
    cursor = 0
    for index, length in enumerate(seqlens.tolist()):
        output[index, :length] = packed[cursor : cursor + length]
        cursor += length
    return output


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

    expected_packed = _prefix_pack_reference(tensor, seqlens)
    expected_unpacked = _prefix_unpack_reference(expected_packed, seqlens, seq_len=4)

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

    expected = _prefix_pack_reference(tensor, seqlens)
    assert torch.equal(packed, expected)


def test_pack_grad_and_delta_from_padded_matches_reference() -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    output = torch.arange(2 * 4 * 2 * 3, dtype=torch.float32).view(2, 4, 2, 3).contiguous()
    grad = (output + 10.0).contiguous()
    seqlens = torch.tensor([2, 3], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    expected_out = _prefix_pack_reference(output, seqlens)
    expected_grad = _prefix_pack_reference(grad, seqlens)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for wide-head fallback coverage.")
def test_wide_head_grad_delta_pack_falls_back_without_truncation() -> None:
    import deberta.modeling.flashdeberta_segment_pack as segment_mod
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    device = torch.device("cuda")
    head_dim = 264

    prefix_grad = torch.randn((2, 4, 2, head_dim), device=device)
    prefix_lengths = torch.tensor([3, 2], dtype=torch.int32, device=device)
    prefix_cu = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    prefix_out = torch.randn((5, 2, head_dim), device=device)
    _, prefix_packed_grad, prefix_delta = varlen_mod._pack_grad_and_delta_from_padded(
        grad_output=prefix_grad,
        output_padded=torch.empty_like(prefix_grad),
        out_unpad=prefix_out,
        seqlens=prefix_lengths,
        cu_seqlens=prefix_cu,
        max_seqlen=3,
        total_tokens=5,
    )
    expected_prefix_grad = torch.cat((prefix_grad[0, :3], prefix_grad[1, :2]), dim=0)
    torch.testing.assert_close(prefix_packed_grad, expected_prefix_grad)
    torch.testing.assert_close(
        prefix_delta,
        (prefix_out.float() * expected_prefix_grad.float()).sum(dim=-1),
    )

    segment_grad = torch.randn((1, 6, 2, head_dim), device=device)
    segment_offsets = torch.tensor([0, 3], dtype=torch.int32, device=device)
    segment_lengths = torch.tensor([2, 3], dtype=torch.int32, device=device)
    segment_cu = torch.tensor([0, 2, 5], dtype=torch.int32, device=device)
    segment_out = torch.randn((5, 2, head_dim), device=device)
    segment_packed_grad, segment_delta = segment_mod.segment_pack_grad_and_delta_from_padded(
        grad_output=segment_grad,
        out_unpad=segment_out,
        segment_offsets=segment_offsets,
        segment_lengths=segment_lengths,
        cu_seqlens=segment_cu,
        total_tokens=5,
        max_segment_length=3,
    )
    expected_segment_grad = torch.cat((segment_grad[0, :2], segment_grad[0, 3:6]), dim=0)
    torch.testing.assert_close(segment_packed_grad, expected_segment_grad)
    torch.testing.assert_close(
        segment_delta,
        (segment_out.float() * expected_segment_grad.float()).sum(dim=-1),
    )


def test_flashdeberta_pack_and_varlen_modules_import_without_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    # Re-importing these modules chain-imports the rest of the flash tree
    # (fixed_op, upstream flashdeberta) under the poisoned triton. Snapshot the
    # whole affected namespace and restore the original healthy module objects
    # afterwards so later tests never see import state cached under triton=None.
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)

    with _isolated_flash_modules():
        importlib.import_module("deberta.modeling.flashdeberta_prefix_pack")
        segment_mod = importlib.import_module("deberta.modeling.flashdeberta_segment_pack")
        varlen_mod = importlib.import_module("deberta.modeling.flashdeberta_varlen_op")
        docblock_mod = importlib.import_module("deberta.modeling.flashdeberta_docblock_op")
        bias_mod = importlib.import_module("deberta.modeling.flashdeberta_bias_op")
        importlib.import_module("deberta.modeling.flashdeberta_dense_bias_op")

        assert segment_mod.flashdeberta_segment_pack_available() is False
        assert varlen_mod.flashdeberta_compiled_varlen_available() is False
        # The custom-op modules recover previously registered ops from the
        # process-global torch.library registry, so availability can be True
        # here when an earlier healthy import registered them. The contract
        # under test is import safety: the calls must not raise.
        assert isinstance(docblock_mod.flashdeberta_compiled_docblock_available(), bool)
        assert isinstance(bias_mod.flashdeberta_compiled_position_bias_available(), bool)


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
    assert dense_meta.route_hint == "dense"

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
        "_flash_meta": FlashBatchMeta(
            seq_lengths=torch.tensor([1024, 768], dtype=torch.int32),
            active_tokens_scalar=torch.tensor(1792, dtype=torch.int32),
        ),
    }
    prepared_fixed, fixed_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=padded_1024,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert fixed_meta is not None
    assert fixed_meta.route_hint == "fixed"
    assert "_flash_meta" not in prepared_fixed
    assert torch.equal(fixed_meta.seq_lengths, torch.tensor([1024, 768], dtype=torch.int32))
    assert int(fixed_meta.active_tokens_scalar) == 1792

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
        "_flash_meta": FlashBatchMeta(
            seq_lengths=torch.tensor([1800, 1700], dtype=torch.int32),
            active_tokens_scalar=torch.tensor(3500, dtype=torch.int32),
        ),
    }
    prepared_varlen, varlen_meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=padded_2048,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )
    assert varlen_meta is not None
    assert varlen_meta.route_hint == "varlen"
    assert "_flash_meta" not in prepared_varlen
    assert torch.equal(varlen_meta.seq_lengths, torch.tensor([1800, 1700], dtype=torch.int32))
    assert int(varlen_meta.active_tokens_scalar) == 3500


def test_prepare_flash_attention_batch_metadata_builds_doc_mask_for_other_backbones() -> None:
    import deberta.training.compile as compile_mod

    # Cross-document blocking must not depend on caller-side preambles: a
    # non-hf_deberta_v2 batch carrying doc_ids gets the dense pairwise mask
    # built here, or packed documents silently attend across boundaries.
    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.long)
    batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": doc_ids.ne(0),
        "doc_ids": doc_ids,
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="rope",
        flash_enabled=False,
    )

    assert meta is None
    assert "doc_ids" not in prepared
    assert tuple(prepared["attention_mask"].shape) == (1, 4, 4)
    assert torch.equal(prepared["attention_mask"], build_doc_block_mask(doc_ids))


def test_prepare_eager_doc_mask_accepts_device_resident_doc_ids() -> None:
    import deberta.training.compile as compile_mod

    # The training loop transfers the batch before preparation. A meta tensor
    # exercises the non-CPU contract without requiring CUDA in the unit suite.
    doc_ids = torch.empty((1, 4), dtype=torch.long, device="meta")
    batch = {
        "input_ids": torch.empty((1, 4), dtype=torch.long, device="meta"),
        "attention_mask": torch.empty((1, 4), dtype=torch.bool, device="meta"),
        "doc_ids": doc_ids,
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="rope",
        flash_enabled=False,
    )

    assert meta is None
    assert "doc_ids" not in prepared
    assert prepared["attention_mask"].device.type == "meta"
    assert tuple(prepared["attention_mask"].shape) == (1, 4, 4)


def test_prepare_flash_attention_batch_metadata_routes_docblock() -> None:
    import deberta.training.compile as compile_mod

    doc_ids = torch.tensor(
        [
            [1, 1, 2, 2, 0],
            [1, 2, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    segment_offsets, segment_lengths, cu_seqlens, active_tokens = build_doc_segment_metadata(doc_ids)
    batch = {
        "input_ids": torch.zeros((2, 5), dtype=torch.long),
        "doc_ids": doc_ids,
        "_flash_meta": FlashBatchMeta(
            doc_segment_offsets=segment_offsets,
            doc_segment_lengths=segment_lengths,
            doc_cu_seqlens=cu_seqlens,
            active_tokens_scalar=torch.tensor(active_tokens, dtype=torch.int32),
            doc_num_segments_scalar=torch.tensor(4, dtype=torch.int32),
            doc_max_segment_length_scalar=torch.tensor(2, dtype=torch.int32),
        ),
    }
    batch["attention_mask"] = batch["doc_ids"].ne(0)

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )

    assert meta is not None
    assert meta.route_hint == "docblock"
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
    assert "_flash_meta" not in prepared
    assert meta.seq_lengths is None
    assert int(meta.active_tokens_scalar) == 7
    assert torch.equal(meta.doc_ids, doc_ids)
    assert meta.doc_segment_offsets is not None
    assert meta.doc_segment_lengths is not None
    assert meta.doc_cu_seqlens is not None
    assert tuple(meta.doc_segment_offsets.shape) == (10,)
    assert tuple(meta.doc_segment_lengths.shape) == (10,)
    assert tuple(meta.doc_cu_seqlens.shape) == (11,)
    assert torch.equal(
        meta.doc_segment_offsets[:4],
        torch.tensor([0, 2, 5, 6], dtype=torch.int32),
    )
    assert torch.equal(
        meta.doc_segment_lengths[:4],
        torch.tensor([2, 2, 1, 2], dtype=torch.int32),
    )
    assert torch.equal(
        meta.doc_cu_seqlens[:5],
        torch.tensor([0, 2, 4, 5, 7], dtype=torch.int32),
    )
    assert torch.count_nonzero(meta.doc_segment_lengths[4:]).item() == 0
    for scalar, expected in (
        (meta.active_tokens_scalar, 7),
        (meta.doc_num_segments_scalar, 4),
        (meta.doc_max_segment_length_scalar, 2),
    ):
        assert scalar is not None
        assert scalar.device.type == "cpu"
        assert scalar.ndim == 0
        assert int(scalar) == expected


def test_prepare_flash_attention_batch_metadata_docblock_eager_gets_pairwise_mask() -> None:
    import deberta.training.compile as compile_mod

    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.long)
    batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": doc_ids.ne(0),
        "doc_ids": doc_ids,
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=False,
    )

    assert meta is None
    assert tuple(prepared["attention_mask"].shape) == (1, 4, 4)
    assert torch.equal(prepared["attention_mask"], build_doc_block_mask(doc_ids))


def test_prepare_flash_attention_batch_metadata_docblock_eager_ignores_flash_overrides(
    tmp_path,
) -> None:
    import deberta.training.compile as compile_mod
    from deberta.modeling.flashdeberta_kernel_tuning import configure_flashdeberta_kernel_overrides

    configure_flashdeberta_kernel_overrides(None)
    doc_ids = torch.tensor([[1, 1, 2, 0]], dtype=torch.long)
    batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": doc_ids.ne(0),
        "doc_ids": doc_ids,
    }
    missing_override_path = tmp_path / "missing-flash-routes.json"

    try:
        prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
            batch=batch,
            backbone_type="hf_deberta_v2",
            flash_enabled=False,
            flash_cfg={"kernel_overrides_path": str(missing_override_path)},
        )

        assert meta is None
        assert torch.equal(prepared["attention_mask"], build_doc_block_mask(doc_ids))
        assert compile_mod._flash_route_hint_for_docblock_batch(seq_len=1024) == "docblock"
    finally:
        configure_flashdeberta_kernel_overrides(None)


def test_docblock_pairwise_attention_blocks_cross_document_probs_on_cpu() -> None:
    from deberta.modeling.deberta_v2_native import DisentangledSelfAttention
    from deberta.modeling.mask_utils import build_doc_block_mask

    seq_len = 16
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


def test_docblock_forced_flash_eager_fallback_rebuilds_pairwise_mask_probs_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    seq_len = 16
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
    _, probs = attention(
        hidden_states=hidden_states,
        attention_mask=doc_ids.ne(0),
        output_attentions=True,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(
            doc_ids=doc_ids,
            route_hint="docblock",
        ),
    )

    assert probs is not None
    assert float(probs[0, 0, 0, seq_len // 2 :].detach().abs().max()) == pytest.approx(0.0)
    assert float(probs[0, 0, seq_len // 2, : seq_len // 2].detach().abs().max()) == pytest.approx(0.0)


def test_docblock_missing_metadata_does_not_fallback_to_padding_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Doc-block eager fallback must fail closed without document ids.

    A compact 2D keep mask only preserves padding; running eager with it
    would silently allow cross-document attention. With neither a pairwise
    mask nor compact document ids, the fallback must raise.
    """

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()

    seq_len = 8
    cfg = _docblock_attention_config(seq_len=seq_len)
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    compact_keep_mask = torch.ones((1, seq_len), dtype=torch.bool)

    with pytest.raises(RuntimeError, match="requires doc_ids"):
        attention(
            hidden_states=hidden_states,
            attention_mask=compact_keep_mask,
            output_attentions=True,
            rel_embeddings=rel_embeddings,
            flash_meta=FlashBatchMeta(route_hint="docblock"),
        )


def test_docblock_missing_metadata_can_fallback_with_explicit_pairwise_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit pairwise mask keeps eager fallback usable without metadata."""

    _install_fake_flashdeberta(monkeypatch)
    attention_mod = _reload_flash_modules()
    from deberta.modeling.mask_utils import build_doc_block_mask

    seq_len = 8
    boundary = seq_len // 2
    cfg = _docblock_attention_config(seq_len=seq_len)
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).eval()
    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), dtype=torch.float32)
    rel_embeddings = torch.zeros((cfg.position_buckets * 2, cfg.hidden_size), dtype=torch.float32)
    doc_ids = torch.cat(
        (
            torch.ones((1, boundary), dtype=torch.long),
            torch.full((1, seq_len - boundary), 2, dtype=torch.long),
        ),
        dim=1,
    )
    pairwise_mask = build_doc_block_mask(doc_ids).unsqueeze(1)

    _, probs = attention(
        hidden_states=hidden_states,
        attention_mask=pairwise_mask,
        output_attentions=True,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(route_hint="docblock"),
    )

    assert probs is not None
    assert float(probs[0, 0, 0, boundary:].detach().abs().max()) == pytest.approx(0.0)
    assert float(probs[0, 0, boundary, :boundary].detach().abs().max()) == pytest.approx(0.0)


def test_prepare_flash_attention_batch_metadata_routes_docblock_bias() -> None:
    import deberta.training.compile as compile_mod

    doc_ids = torch.tensor(
        [
            [1, 1, 2, 2, 0],
            [1, 2, 2, 0, 0],
        ],
        dtype=torch.long,
    )
    batch = {
        "input_ids": torch.zeros((2, 5), dtype=torch.long),
        "doc_ids": doc_ids,
        "_flash_meta": FlashBatchMeta(
            active_tokens_scalar=torch.tensor(7, dtype=torch.int32),
        ),
    }
    batch["attention_mask"] = batch["doc_ids"].ne(0)

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
        flash_cfg={"docblock_bias_seq_len": 5},
    )

    assert meta is not None
    assert meta.route_hint == "docblock_bias"
    assert "doc_ids" not in prepared
    assert "_flash_meta" not in prepared
    assert tuple(prepared["attention_mask"].shape) == (2, 5, 5)
    assert prepared["attention_mask"].dtype == torch.bool
    assert meta.seq_lengths is None
    assert int(meta.active_tokens_scalar) == 7
    assert torch.equal(meta.doc_ids, doc_ids)


def test_prepare_flash_metadata_does_not_route_non_prefix_padding_mask_to_flash() -> None:
    """Metadata prep must not bake prefix seq_lengths from a mask with holes."""

    import deberta.training.compile as compile_mod

    batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.tensor([[True, False, True, False]]),
    }

    prepared, meta = compile_mod.prepare_flash_attention_batch_metadata(
        batch=batch,
        backbone_type="hf_deberta_v2",
        flash_enabled=True,
    )

    assert prepared is batch
    assert meta is None


def test_flash_attention_docblock_path_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch)
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

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(
            doc_segment_offsets=torch.tensor([0, 2], dtype=torch.int32),
            doc_segment_lengths=torch.tensor([2, 1], dtype=torch.int32),
            doc_cu_seqlens=torch.tensor([0, 2, 3], dtype=torch.int32),
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


def test_flash_attention_docblock_bias_path_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    attention_mod, attention, cfg = _flash_attention_harness(monkeypatch)
    monkeypatch.setattr(attention_mod, "flashdeberta_bias_import_error", lambda: None)
    monkeypatch.setattr(attention_mod, "flashdeberta_compiled_position_bias_available", lambda: True)
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

    output, probs = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
        flash_meta=FlashBatchMeta(route_hint="docblock_bias"),
    )

    assert probs is None
    assert tuple(output.shape) == (1, 4, cfg.hidden_size)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    # Rows 0-2 are active (diagonal True) and pass through; row 3 is an
    # inactive query (diagonal False) and must be zeroed to match eager.
    for seq_idx in range(3):
        for head_idx in range(cfg.num_attention_heads):
            start = head_idx * head_dim
            end = start + head_dim
            assert torch.all(output[0, seq_idx, start:end].eq(float(head_idx * 100 + seq_idx)))
    assert torch.all(output[0, 3].eq(0.0))
    assert seen["keep_mask"] is not None
    assert tuple(seen["keep_mask"].shape) == (1, 1, 4, 4)


def _canonical_dense_bias_reference(
    *,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    bucket_index: torch.Tensor,
    keep_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Build a test-only dense bias from the canonical scalar definition."""

    reference = pos_key if pos_key is not None else pos_query
    assert reference is not None
    batch_size, num_heads, seq_len, _ = reference.shape
    batches: list[torch.Tensor] = []
    for batch_idx in range(batch_size):
        heads: list[torch.Tensor] = []
        for head_idx in range(num_heads):
            rows: list[torch.Tensor] = []
            for query_idx in range(seq_len):
                cells: list[torch.Tensor] = []
                for key_idx in range(seq_len):
                    bucket = int(bucket_index[query_idx, key_idx])
                    terms: list[torch.Tensor] = []
                    if pos_key is not None:
                        terms.append(pos_key[batch_idx, head_idx, query_idx, bucket])
                    if pos_query is not None:
                        terms.append(pos_query[batch_idx, head_idx, key_idx, bucket])
                    cells.append(torch.stack(terms).sum() * float(scale))
                rows.append(torch.stack(cells))
            heads.append(torch.stack(rows))
        batches.append(torch.stack(heads))
    bias = torch.stack(batches)
    return bias.masked_fill(~keep_mask, -1.0e4 * float(scale)) if keep_mask is not None else bias


def test_dense_bias_fallback_matches_scaled_reference() -> None:
    """CPU dense-bias fallback must match the canonical DeBERTa definition.

    The reference is an explicit per-element loop:
    ``bias[m,n] = pos_key[m,bucket[m,n]] + pos_query[n,bucket[m,n]]``.
    The bucket map is deliberately asymmetric because signed relative-position
    buckets are not symmetric; reversing the P2C bucket fails this test.
    """

    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    torch.manual_seed(0)
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
            [4, 0, 1, 2],
            [5, 4, 0, 1],
            [6, 5, 4, 0],
        ],
        dtype=torch.int64,
    )
    assert not torch.equal(bucket_index, bucket_index.t())
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

    expected = _canonical_dense_bias_reference(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )

    actual = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    torch.testing.assert_close(actual, expected)


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

    bias = _canonical_dense_bias_reference(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    (bias * d_bias).sum().backward()

    actual_key, actual_query = bias_mod._position_bias_backward_from_dense_grad(
        d_bias=d_bias,
        pos_key_num_buckets=num_buckets if pos_key is not None else 0,
        pos_query_num_buckets=num_buckets if pos_query is not None else 0,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
        output_dtype=d_bias.dtype,
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
        bucket_index = torch.empty((3, 3), device="cuda", dtype=torch.int64)
        keep_mask = torch.empty((2, 1, 3, 3), device="cuda", dtype=torch.bool)

        dq, dk, dv, dpos_key, dpos_query = bias_mod._FLASHDEBERTA_POSITION_BIAS_BWD_CUSTOM_OP(
            grad_out,
            q,
            k,
            v,
            bucket_index,
            keep_mask,
            bias,
            out,
            lse,
            0.5,
            0.5,
            False,
            7,
            7,
            True,
        )

    assert tuple(dq.shape) == tuple(q.shape)
    assert tuple(dk.shape) == tuple(k.shape)
    assert tuple(dv.shape) == tuple(v.shape)
    assert tuple(dpos_key.shape) == (2, 4, 3, 7)
    assert tuple(dpos_query.shape) == (2, 4, 3, 7)


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

    ref_bias = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key_ref,
        pos_query=pos_query_ref,
        bucket_index=bucket_index,
        keep_mask=keep_mask,
        scale=scale,
    )
    with torch.no_grad():
        fallback_bias = dense_bias_mod._dense_bias_forward_fallback(
            pos_key=pos_key.detach(),
            pos_query=pos_query.detach(),
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            scale=scale,
        )
    torch.testing.assert_close(fallback_bias, ref_bias.detach(), atol=2e-2, rtol=2e-2)
    ref_scores = torch.matmul(q_ref.float(), k_ref.float().transpose(-1, -2)) * scale
    ref_probs = torch.softmax(ref_scores + ref_bias.float(), dim=-1)
    ref_out = torch.matmul(ref_probs, v_ref.float()).to(dtype=dtype)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for FlashDeBERTa parity.")
@pytest.mark.parametrize("route", ["fixed", "varlen"])
def test_fixed_and_varlen_kernels_match_canonical_unit_scale_p2c(route: str) -> None:
    """Pinned kernels must use the canonical signed P2C bucket in forward/backward."""

    with _isolated_flash_modules():
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        fixed_mod = importlib.import_module("deberta.modeling.flashdeberta_fixed_op")
        varlen_mod = importlib.import_module("deberta.modeling.flashdeberta_varlen_op")
        if fixed_mod.flashdeberta_fixed_import_error() is not None:
            pytest.skip("Fixed FlashDeBERTa kernels are unavailable.")
        if varlen_mod.flashdeberta_varlen_import_error() is not None:
            pytest.skip("Varlen FlashDeBERTa kernels are unavailable.")
        _run_fixed_or_varlen_canonical_p2c_check(
            route=route,
            attention_mod=attention_mod,
            fixed_mod=fixed_mod,
            varlen_mod=varlen_mod,
        )


def _run_fixed_or_varlen_canonical_p2c_check(*, route, attention_mod, fixed_mod, varlen_mod) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_heads, seq_len, head_dim, num_buckets = 2, 2, 32, 64, 16
    scale = float(head_dim) ** -0.5
    lengths = torch.tensor(
        [seq_len, seq_len if route == "fixed" else 19],
        dtype=torch.int32,
        device=device,
    )
    bucket_index = attention_mod._dense_bucket_index_tensor(
        seq_len=seq_len,
        position_buckets=num_buckets // 2,
        max_relative_distance=seq_len,
        device=device,
    )

    q = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # Unit-scale, asymmetric values prevent initialization scale from hiding a
    # signed-bucket reversal. Isolating P2C prevents C2P from compensating.
    pos_query = torch.randn(
        (batch_size, num_heads, seq_len, num_buckets),
        device=device,
        dtype=dtype,
    )
    q_kernel = q.clone().requires_grad_()
    k_kernel = k.clone().requires_grad_()
    v_kernel = v.clone().requires_grad_()
    pos_query_kernel = pos_query.clone().requires_grad_()

    if route == "fixed":
        output_kernel = fixed_mod.flashdeberta_fixed(
            query_layer=q_kernel,
            key_layer=k_kernel,
            value_layer=v_kernel,
            seq_lengths=None,
            pos_key=None,
            pos_query=pos_query_kernel,
            sm_scale=scale,
            position_buckets=num_buckets // 2,
            max_relative_distance=seq_len,
            causal=False,
        )
    else:
        mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        output_kernel = varlen_mod.flashdeberta_varlen_padded(
            query_layer=q_kernel.permute(0, 2, 1, 3).contiguous(),
            key_layer=k_kernel.permute(0, 2, 1, 3).contiguous(),
            value_layer=v_kernel.permute(0, 2, 1, 3).contiguous(),
            attention_mask_2d=mask,
            pos_key=None,
            pos_query=pos_query_kernel.permute(0, 2, 1, 3).contiguous(),
            sm_scale=scale,
            position_buckets=num_buckets // 2,
            max_relative_distance=seq_len,
            causal=False,
        ).permute(0, 2, 1, 3)

    q_ref = q.clone().float().requires_grad_()
    k_ref = k.clone().float().requires_grad_()
    v_ref = v.clone().float().requires_grad_()
    pos_query_ref = pos_query.clone().float().requires_grad_()
    reference_batches: list[torch.Tensor] = []
    for batch_idx, active_len in enumerate(lengths.tolist()):
        active_len = int(active_len)
        buckets = bucket_index[:active_len, :active_len]
        p2c_source = pos_query_ref[batch_idx : batch_idx + 1, :, None, :active_len, :]
        p2c_index = buckets.view(1, 1, active_len, active_len, 1).expand(
            1,
            num_heads,
            active_len,
            active_len,
            1,
        )
        p2c = torch.gather(
            p2c_source.expand(-1, -1, active_len, -1, -1),
            dim=-1,
            index=p2c_index,
        ).squeeze(-1)
        scores = (
            torch.matmul(
                q_ref[batch_idx : batch_idx + 1, :, :active_len],
                k_ref[batch_idx : batch_idx + 1, :, :active_len].transpose(-1, -2),
            )
            + p2c
        ) * scale
        active_output = torch.matmul(
            torch.softmax(scores, dim=-1),
            v_ref[batch_idx : batch_idx + 1, :, :active_len],
        )
        reference_batches.append(torch.nn.functional.pad(active_output, (0, 0, 0, seq_len - active_len)))
    output_ref = torch.cat(reference_batches, dim=0)

    grad_output = torch.randn_like(output_kernel)
    kernel_grads = torch.autograd.grad(
        (output_kernel * grad_output).sum(),
        (q_kernel, k_kernel, v_kernel, pos_query_kernel),
    )
    reference_grads = torch.autograd.grad(
        (output_ref * grad_output.float()).sum(),
        (q_ref, k_ref, v_ref, pos_query_ref),
    )

    torch.testing.assert_close(output_kernel.float(), output_ref, atol=3e-2, rtol=3e-2)
    for actual, expected in zip(kernel_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual.float(), expected, atol=7e-2, rtol=7e-2)


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
    assert (batch_size, num_heads, seq_len, num_buckets) not in saved_shapes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for dense-bias kernel checks.")
def test_dense_bias_triton_builder_honors_per_head_keep_mask() -> None:
    """The fused dense-bias builder must read each head's own keep-mask plane.

    A per-head ``(B,H,S,S)`` mask exercised head 0's plane for every head
    before the head stride was applied; the eager fallback (plain broadcast
    ops) is the shape-agnostic reference. Head dims other than 1 or H must be
    rejected instead of silently misread.
    """

    import deberta.modeling.flashdeberta_attention as attention_mod
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    if dense_bias_mod.triton is None:
        pytest.skip("Fused dense-bias builder is unavailable in this environment.")

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_heads, seq_len, num_buckets = 2, 3, 32, 16
    scale = 0.125
    bucket_index = attention_mod._dense_bucket_index_tensor(
        seq_len=seq_len,
        position_buckets=num_buckets // 2,
        max_relative_distance=seq_len,
        device=device,
    )
    pos_key = torch.randn((batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype)
    pos_query = torch.randn((batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype)
    per_head_mask = torch.rand((batch_size, num_heads, seq_len, seq_len), device=device) > 0.4
    # Make head planes provably different so a head-0 fallback cannot pass.
    per_head_mask[:, 1] = ~per_head_mask[:, 0]

    triton_bias = dense_bias_mod._dense_bias_forward_cuda(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=per_head_mask,
        scale=scale,
    )
    eager_bias = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=per_head_mask,
        scale=scale,
    )
    torch.testing.assert_close(triton_bias, eager_bias, atol=2e-2, rtol=2e-2)

    with pytest.raises(ValueError, match="head dimension"):
        dense_bias_mod._dense_bias_forward_cuda(
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=per_head_mask[:, :2],
            scale=scale,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for specialized backward checks.")
def test_specialized_docblock_backward_honors_per_head_keep_mask() -> None:
    """Specialized docblock backward must read each head's own keep-mask plane.

    The positional-gradient accumulation (dpos_key/dpos_query) gates its
    atomic adds on the keep mask; with a per-head ``(B,H,S,S)`` mask, head h
    must use plane h, not head 0's plane. A generic dense backward plus the
    independently validated canonical bucket reduction checks every gradient;
    one-head launches separately pin the mask-head stride.

    Other tests in this file reload the flash module tree against fake
    flashdeberta packages, so this test re-imports a clean real-kernel tree
    and restores the prior modules after.
    """

    with _isolated_flash_modules():
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        bias_mod = importlib.import_module("deberta.modeling.flashdeberta_bias_op")
        dense_bias_mod = importlib.import_module("deberta.modeling.flashdeberta_dense_bias_op")
        if bias_mod.triton is None or bias_mod._bwd_kv_kernel_docblock1024 is None:
            pytest.skip("Specialized docblock backward kernels are unavailable.")
        _run_specialized_docblock_backward_per_head_check(
            attention_mod=attention_mod,
            bias_mod=bias_mod,
            dense_bias_mod=dense_bias_mod,
        )


def _run_specialized_docblock_backward_per_head_check(*, attention_mod, bias_mod, dense_bias_mod) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, num_heads, seq_len, head_dim, num_buckets = 1, 2, 512, 64, 32
    sm_scale = float(head_dim) ** -0.5
    bias_scale = sm_scale

    configs = bias_mod._resolve_docblock1024_bwd_configs(
        batch_size=batch_size,
        num_heads=num_heads,
        query_len=seq_len,
        key_len=seq_len,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
    )
    assert configs is not None, "specialized kernels must be launchable for this shape"

    bucket_index = attention_mod._dense_bucket_index_tensor(
        seq_len=seq_len,
        position_buckets=num_buckets // 2,
        max_relative_distance=seq_len,
        device=device,
    )

    def _doc_pairwise(boundary: int) -> torch.Tensor:
        doc_ids = torch.cat(
            (
                torch.ones((1, boundary), dtype=torch.long, device=device),
                torch.full((1, seq_len - boundary), 2, dtype=torch.long, device=device),
            ),
            dim=1,
        )
        return doc_ids[:, :, None].eq(doc_ids[:, None, :]).unsqueeze(1)

    per_head_mask = torch.cat((_doc_pairwise(seq_len // 2), _doc_pairwise(seq_len // 4)), dim=1)
    assert not torch.equal(per_head_mask[:, 0], per_head_mask[:, 1])

    q = torch.randn((batch_size, num_heads, seq_len, head_dim), device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    pos_key = torch.randn((batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype)
    pos_query = torch.randn((batch_size, num_heads, seq_len, num_buckets), device=device, dtype=dtype)
    bias = dense_bias_mod._dense_bias_forward_fallback(
        pos_key=pos_key,
        pos_query=pos_query,
        bucket_index=bucket_index,
        keep_mask=per_head_mask,
        scale=bias_scale,
    )
    out, lse = bias_mod._bias_eager_forward_impl(
        q=q, k=k, v=v, bias=bias, sm_scale=sm_scale, causal=False, require_lse=True
    )
    if lse is None:
        pytest.skip("Low-level bias forward with LSE is unavailable.")
    grad_out = torch.randn_like(out)

    dq, dk, dv, dpos_key, dpos_query = bias_mod._position_bias_specialized_docblock_backward_impl(
        grad_out=grad_out,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        pos_key_num_buckets=num_buckets,
        pos_query_num_buckets=num_buckets,
        bucket_index=bucket_index,
        keep_mask=per_head_mask,
        bias_scale=bias_scale,
        sm_scale=sm_scale,
    )

    ref_dq, ref_dk, ref_dv, ref_d_bias = bias_mod._bias_generic_backward_impl(
        grad_out=grad_out,
        q=q,
        k=k,
        v=v,
        bias=bias,
        out=out,
        lse=lse,
        sm_scale=sm_scale,
        causal=False,
    )
    ref_dpos_key, ref_dpos_query = bias_mod._position_bias_backward_from_dense_grad(
        d_bias=ref_d_bias,
        pos_key_num_buckets=num_buckets,
        pos_query_num_buckets=num_buckets,
        bucket_index=bucket_index,
        keep_mask=per_head_mask,
        scale=bias_scale,
        output_dtype=dtype,
    )
    torch.testing.assert_close(dq, ref_dq, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dk, ref_dk, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dv, ref_dv, atol=5e-2, rtol=5e-2)
    assert ref_dpos_key is not None
    assert ref_dpos_query is not None
    torch.testing.assert_close(dpos_key, ref_dpos_key, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dpos_query, ref_dpos_query, atol=5e-2, rtol=5e-2)

    # Keep the per-head comparison as a separate stride regression. The
    # generic comparison above is the independent canonical P2C oracle.
    for head in range(num_heads):
        sl = slice(head, head + 1)
        ref = bias_mod._position_bias_specialized_docblock_backward_impl(
            grad_out=grad_out[:, sl].contiguous(),
            q=q[:, sl].contiguous(),
            k=k[:, sl].contiguous(),
            v=v[:, sl].contiguous(),
            bias=bias[:, sl].contiguous(),
            out=out[:, sl].contiguous(),
            lse=lse[:, sl].contiguous(),
            pos_key_num_buckets=num_buckets,
            pos_query_num_buckets=num_buckets,
            bucket_index=bucket_index,
            keep_mask=per_head_mask[:, sl].contiguous(),
            bias_scale=bias_scale,
            sm_scale=sm_scale,
        )
        dq_h, dk_h, dv_h, dpos_key_h, dpos_query_h = ref
        torch.testing.assert_close(dq[:, sl], dq_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dk[:, sl], dk_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dv[:, sl], dv_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dpos_key[:, sl], dpos_key_h, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(dpos_query[:, sl], dpos_query_h, atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real-kernel leakage checks.")
@pytest.mark.parametrize("route", ["docblock", "docblock_bias"])
def test_docblock_real_kernel_blocks_cross_document_gradients_on_cuda(route: str) -> None:
    """Gradient-isolation leakage check on the actual Triton doc-block routes.

    Doc-1 query outputs must carry exactly zero gradient back to doc-2 hidden
    states; any nonzero gradient means cross-document attention leaked. The
    test rejects eager fallback so only the selected flash route can pass.

    Other tests in this file reload the flash module tree against fake
    flashdeberta packages and leave those reloaded twins cached, so this test
    re-imports a clean real-kernel tree and restores the prior modules after.
    """

    with _isolated_flash_modules():
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        if attention_mod.flashdeberta_fixed_import_error() is not None:
            pytest.skip("FlashDeBERTa kernels are unavailable in this environment.")
        _run_docblock_real_kernel_leak_check(attention_mod=attention_mod, route=route)


def _run_docblock_real_kernel_leak_check(*, attention_mod, route: str) -> None:
    from deberta.modeling.deberta_v2_native import DebertaV2Config
    from deberta.modeling.mask_utils import build_doc_block_mask, build_doc_segment_metadata

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 1024
    # Cross the 64/128-token kernel tile boundaries to exercise ragged tails.
    boundary = 517
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
    cfg.hf_flash = {}
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).to(device=device, dtype=dtype).eval()

    def _reject_eager(**_kwargs):
        raise AssertionError(f"flash {route} route unexpectedly fell back to eager")

    attention._eager_forward_fallback = _reject_eager

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
            doc_ids=doc_ids.to(device=device),
            route_hint="docblock_bias",
        )
    else:
        segment_offsets, segment_lengths, cu_seqlens, active_tokens = build_doc_segment_metadata(doc_ids)
        active_segment_lengths = segment_lengths[segment_lengths.ne(0)]
        num_segments = int(active_segment_lengths.numel())
        max_seqlen = int(active_segment_lengths.max()) if num_segments else 0
        attention_mask = doc_ids.ne(0).to(device=device)
        flash_meta = FlashBatchMeta(
            doc_segment_offsets=segment_offsets.to(device=device),
            doc_segment_lengths=segment_lengths.to(device=device),
            doc_cu_seqlens=cu_seqlens.to(device=device),
            doc_ids=doc_ids.to(device=device),
            active_tokens_scalar=torch.tensor(active_tokens, dtype=torch.int32),
            doc_num_segments_scalar=torch.tensor(num_segments, dtype=torch.int32),
            doc_max_segment_length_scalar=torch.tensor(max_seqlen, dtype=torch.int32),
            route_hint="docblock",
        )

    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), device=device, dtype=dtype).requires_grad_()
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), device=device, dtype=dtype)

    output, _ = attention(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=False,
        rel_embeddings=rel_embeddings,
        flash_meta=flash_meta,
    )
    output[0, :boundary].float().square().sum().backward()

    assert hidden_states.grad is not None
    doc2_grad = hidden_states.grad[0, boundary:]
    assert torch.all(doc2_grad == 0), (
        f"cross-document gradient leak on {route}: max abs doc-2 grad {float(doc2_grad.abs().max()):.3e}"
    )
    doc1_grad = hidden_states.grad[0, :boundary]
    assert float(doc1_grad.abs().max()) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for dense doc-block parity.")
def test_docblock_bias_dense_route_matches_eager_on_padded_batch() -> None:
    """Full-output parity for the dense docblock_bias route on a padded batch.

    Eager attention zeroes inactive query rows after softmax; the dense
    flash-with-bias route must not emit a real CLS value mixture at padding
    positions. Compares the whole output tensor, padding rows included.
    """

    with _isolated_flash_modules():
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        if attention_mod.flashdeberta_fixed_import_error() is not None:
            pytest.skip("FlashDeBERTa kernels are unavailable in this environment.")
        _run_docblock_bias_padded_parity_check(attention_mod=attention_mod)


def _run_docblock_bias_padded_parity_check(*, attention_mod) -> None:
    from deberta.modeling.deberta_v2_native import DebertaV2Config
    from deberta.modeling.mask_utils import build_doc_block_mask

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 1024
    active_len = 800
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
    cfg.hf_flash = {}
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).to(device=device, dtype=dtype).eval()

    def _reject_eager(**_kwargs):
        raise AssertionError("dense doc-block route unexpectedly fell back to eager")

    attention._eager_forward_fallback = _reject_eager
    reference = attention_mod._EagerDisentangledSelfAttention(cfg)
    reference.load_state_dict(attention.state_dict())
    reference = reference.to(device=device, dtype=dtype).eval()

    doc_ids = torch.cat(
        (
            torch.ones((1, active_len // 2), dtype=torch.long),
            torch.full((1, active_len - active_len // 2), 2, dtype=torch.long),
            torch.zeros((1, seq_len - active_len), dtype=torch.long),
        ),
        dim=1,
    ).to(device=device)
    attention_mask = build_doc_block_mask(doc_ids)
    flash_meta = FlashBatchMeta(
        doc_ids=doc_ids,
        route_hint="docblock_bias",
    )

    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), device=device, dtype=dtype)
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), device=device, dtype=dtype)

    with torch.no_grad():
        flash_out, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=False,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )
    with torch.no_grad():
        eager_out, _ = reference(
            hidden_states=hidden_states,
            attention_mask=attention_mask.unsqueeze(1),
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )

    # Eager zeroes inactive query rows exactly; the flash route must match.
    padding_rows = flash_out[0, active_len:]
    assert torch.all(padding_rows == 0), (
        f"padding query rows carry real values: max abs {float(padding_rows.abs().max()):.3e}"
    )
    torch.testing.assert_close(flash_out, eager_out, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for padding-mask parity.")
def test_non_prefix_padding_mask_matches_eager_on_cuda() -> None:
    """A holey padding mask must produce eager outputs, not prefix-length flash outputs.

    Pre-fix, the fixed/varlen routes collapsed any padding mask to per-example
    lengths, silently attending the wrong key positions for masks with holes.
    """

    with _isolated_flash_modules():
        attention_mod = importlib.import_module("deberta.modeling.flashdeberta_attention")
        if attention_mod.flashdeberta_fixed_import_error() is not None:
            pytest.skip("FlashDeBERTa kernels are unavailable in this environment.")
        _run_non_prefix_padding_parity_check(attention_mod=attention_mod)


def _run_non_prefix_padding_parity_check(*, attention_mod) -> None:
    from deberta.modeling.deberta_v2_native import DebertaV2Config

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 1024
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
    cfg.hf_flash = {}
    attention = attention_mod.FlashDisentangledSelfAttention(cfg).to(device=device, dtype=dtype).eval()
    reference = attention_mod._EagerDisentangledSelfAttention(cfg)
    reference.load_state_dict(attention.state_dict())
    reference = reference.to(device=device, dtype=dtype).eval()
    eager_fallback = attention._eager_forward_fallback
    fallback_calls = 0

    def _record_eager_fallback(**kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        return eager_fallback(**kwargs)

    attention._eager_forward_fallback = _record_eager_fallback

    # A hole in the middle plus tail padding: same active-token count as a
    # prefix mask of length 824, but different key positions - collapsing it
    # to seq_lengths silently attends the wrong keys.
    keep = torch.ones((1, seq_len), dtype=torch.bool, device=device)
    keep[:, 100:300] = False
    keep[:, 900:] = False
    broadcast_mask = keep.view(1, 1, 1, seq_len)

    hidden_states = torch.randn((1, seq_len, cfg.hidden_size), device=device, dtype=dtype)
    rel_embeddings = torch.randn((cfg.position_buckets * 2, cfg.hidden_size), device=device, dtype=dtype)

    with torch.no_grad():
        flash_out, _ = attention(
            hidden_states=hidden_states,
            attention_mask=broadcast_mask,
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )
    assert fallback_calls == 1

    with torch.no_grad():
        eager_out, _ = reference(
            hidden_states=hidden_states,
            attention_mask=broadcast_mask,
            output_attentions=False,
            rel_embeddings=rel_embeddings,
        )
    torch.testing.assert_close(flash_out, eager_out)


def test_varlen_bwd_config_resolution_uses_conservative_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    monkeypatch.setattr(varlen_mod, "_varlen_repo_tuned_config", lambda **kwargs: None)

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

    assert kv_config == (16, 16, 1, 4)
    assert q_config == (16, 16, 1, 4)


def test_varlen_repo_tuned_config_uses_density_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    monkeypatch.setattr(varlen_mod, "device_compute_capability", lambda device: (12, 0))

    sparse_cfg = varlen_mod._varlen_repo_tuned_config(
        kind="bwd_kv",
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
    long_cfg = varlen_mod._varlen_repo_tuned_config(
        kind="bwd_q",
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

    # Recover the op from the process-global torch.library registry instead of
    # the module global: earlier fake-package tests can leave the canonical
    # module imported without ops even though the registered op is live.
    _, varlen_bwd_op = varlen_mod._build_varlen_custom_ops()
    if varlen_bwd_op is None:
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

        _, _, _, dpos_key, dpos_query = varlen_bwd_op(
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

    # Patch the repo-level capability seam: patching torch.cuda internals is
    # defeated by the per-index capability cache and initializes CUDA on
    # CPU-only machines via current_device().
    monkeypatch.setattr(fixed_mod, "device_compute_capability", lambda _device: (12, 0))

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


def test_bias_bwd_config_resolution_uses_conservative_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    monkeypatch.setattr(bias_mod, "_bias_repo_tuned_config", lambda **kwargs: None)

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

    assert kv_config == (16, 16, 1, 4)
    assert q_config == (16, 16, 1, 4)


def test_bias_repo_tuned_config_is_table_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod

    monkeypatch.setattr(bias_mod, "device_compute_capability", lambda _device: (12, 0))

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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the specialized-backward gate."
)
def test_specialized_docblock_bias_backward_enables_new_seq_len_from_table(tmp_path) -> None:
    import deberta.modeling.flashdeberta_bias_op as bias_mod
    from deberta.modeling.flashdeberta_kernel_tuning import configure_flashdeberta_kernel_overrides

    if bias_mod.triton is None:
        pytest.skip("Triton is required for the specialized-backward gate.")

    def _gate(seq_len: int) -> bool:
        q = torch.zeros((1, 2, seq_len, 64), dtype=torch.bfloat16, device="cuda")
        bias = torch.zeros((1, 2, seq_len, seq_len), dtype=torch.bfloat16, device="cuda")
        return bias_mod._should_use_specialized_docblock_bias_backward(q=q, k=q, v=q, bias=bias, causal=False)

    configure_flashdeberta_kernel_overrides(None)
    # The shipped table has no 512 row, so the gate stays closed.
    assert not _gate(512)
    # Adding a tuning-table row is sufficient to open the gate: shape
    # enablement must live in data, not in a hardcoded seq-len set in src.
    with _kernel_tuning_overrides(
        tmp_path,
        {
            "kernels": [
                {
                    "route": "bias_docblock_specialized",
                    "kind": "bwd",
                    "seq_bucket": "default",
                    "query_len": 512,
                    "key_len": 512,
                    "head_dim": 64,
                    "dtype": "bfloat16",
                    "causal": False,
                    "has_mask": True,
                    "block_m": 32,
                    "block_n": 32,
                    "num_stages": 1,
                    "num_warps": 2,
                }
            ]
        },
        filename="flash_specialized_512.json",
    ):
        assert _gate(512)


def test_dense_bias_repo_tuned_config_matches_sm120_docblock_1024(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_dense_bias_op as dense_bias_mod

    monkeypatch.setattr(dense_bias_mod, "device_compute_capability", lambda _device: (12, 0))

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

    monkeypatch.setattr(dv2, "is_torch_compiling", lambda: True)
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


def test_docblock_varlen_backward_uses_docblock_tuning_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    import deberta.modeling.flashdeberta_varlen_op as varlen_mod

    seen: dict[str, str] = {}

    def _fake_resolve(**context):
        assert context["guard"]()
        seen["route"] = context["route"]
        seen["kind"] = context["kind"]
        return (16, 32, 1, 4)

    monkeypatch.setattr(varlen_mod, "resolve_repo_tuned_config", _fake_resolve)
    monkeypatch.setattr(varlen_mod, "device_compute_capability", lambda _device: (12, 0))
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
