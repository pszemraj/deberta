"""FlashDeBERTa attention adapter for the native DeBERTa-v2/v3 backbone.

This revision is tuned for the current repo state and for ``torch.compile``.

What changed versus the earlier adapter
--------------------------------------
1. Stats are now debug-only and disabled by default. The prior adapter mutated
   a Python ``Counter`` inside the attention forward path, which is exactly the
   kind of Python/global state that TorchDynamo may guard on and recompile.
2. Runtime policy is read from the resolved DeBERTa config at construction time;
   environment variables are limited to debug instrumentation.
3. Dense-vs-varlen routing no longer inspects attention-mask contents inside the
   compiled forward path. In this repository's training loop, dense batches already
   arrive as ``attention_mask=None`` because the collator drops all-ones masks.
   Using that contract is both faster and more compile-friendly.

Important behavior
------------------
- Fixed-length flash remains the default path for dense / maskless batches.
- Varlen flash is used for padded batches when enabled by runtime policy.
- The padded varlen path now runs through an opaque custom op on CUDA so
  ``torch.compile`` does not trace into FlashDeBERTa's Python/Triton wrapper.
- Pairwise masks still fall back to eager attention for correctness.

Runtime controls
----------------
Use ``model.hf.attention_impl=flash`` and ``model.hf.flash.*`` for routing and
kernel policy. Set these optional instrumentation environment variables before
importing this module:

- ``FLASHDEBERTA_DEBUG_STATS`` (default: ``0``)
    Enable eager/debug-only path counters for benchmark scripts.
- ``FLASHDEBERTA_WARN_FALLBACKS`` (default: ``1``)
    Emit one warning per fallback reason outside compiled graphs.
"""

from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from deberta.modeling.deberta_v2_native import (
    DisentangledSelfAttention as _EagerDisentangledSelfAttention,
)
from deberta.modeling.deberta_v2_native import (
    build_relative_position as _build_relative_position,
)
from deberta.modeling.flashdeberta_bias_op import (
    flashdeberta_bias_from_positions,
    flashdeberta_bias_import_error,
    flashdeberta_compiled_bias_available,
    flashdeberta_compiled_position_bias_available,
)
from deberta.modeling.flashdeberta_docblock_op import (
    flashdeberta_compiled_docblock_available,
    flashdeberta_docblock,
    flashdeberta_docblock_import_error,
)
from deberta.modeling.flashdeberta_fixed_op import (
    flashdeberta_fixed,
    flashdeberta_fixed_import_error,
)
from deberta.modeling.flashdeberta_kernel_tuning import (
    configure_flashdeberta_kernel_overrides,
    flash_padding_route,
    flash_route_policy,
    flash_seq_bucket,
)
from deberta.modeling.flashdeberta_op_utils import device_compute_capability
from deberta.modeling.flashdeberta_varlen_op import (
    flashdeberta_compiled_varlen_available,
    flashdeberta_varlen_padded,
)
from deberta.modeling.flashdeberta_version import flashdeberta_version_error, require_flashdeberta_version
from deberta.modeling.mask_utils import (
    FlashBatchMeta,
    build_doc_block_mask,
    doc_ids_from_segments,
    is_pairwise_mask,
    is_torch_compiling,
    mask_to_2d_keep_mask,
    normalize_keep_mask,
)

_FLASH_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_FLASH_STATS: Counter[str] = Counter()
_TRUTHY = {"1", "true", "yes", "y", "on"}
_DENSE_BUCKET_INDEX_CACHE: dict[tuple[int, int, int, str, int | None], torch.Tensor] = {}


@dataclass(frozen=True)
class FlashDebertaRuntimeConfig:
    """Static runtime policy for the flash adapter.

    Values are intentionally read once per process to reduce Python work and
    guard surface inside compiled attention forwards.
    """

    force_varlen: bool = False
    varlen_min_seq_len: int | None = None
    docblock_bias_seq_len: int | None = None
    local_bias_seq_len: int | None = None
    local_bias_max_batch_size: int | None = None
    eager_dense_max_seq_len: int = 0
    kernel_overrides_path: str | None = None
    enable_debug_stats: bool = False
    warn_fallbacks: bool = True


def _truthy_env(name: str, default: str = "0") -> bool:
    """Parse a boolean-ish environment variable.

    :param str name: Environment variable name.
    :param str default: Default string to parse when the variable is unset.
    :return bool: Whether the value matches the repo's truthy set.
    """

    return os.environ.get(name, default).strip().lower() in _TRUTHY


def _optional_int(value: Any, *, default: int | None = None) -> int | None:
    """Return an integer override or None when unset.

    :param Any value: Raw config value.
    :param int | None default: Fallback value for invalid input.
    :return int | None: Parsed integer, or None.
    """

    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return default


def _read_runtime_config_from_env() -> FlashDebertaRuntimeConfig:
    """Load debug instrumentation toggles from environment variables.

    :return FlashDebertaRuntimeConfig: Default runtime policy plus debug toggles.
    """

    return FlashDebertaRuntimeConfig(
        force_varlen=False,
        varlen_min_seq_len=None,
        docblock_bias_seq_len=None,
        local_bias_seq_len=None,
        local_bias_max_batch_size=None,
        eager_dense_max_seq_len=0,
        kernel_overrides_path=None,
        enable_debug_stats=_truthy_env("FLASHDEBERTA_DEBUG_STATS", default="0"),
        warn_fallbacks=_truthy_env("FLASHDEBERTA_WARN_FALLBACKS", default="1"),
    )


_RUNTIME_CONFIG = _read_runtime_config_from_env()


def refresh_flashdeberta_runtime_config_from_env() -> None:
    """Reload debug instrumentation toggles from environment variables.

    This exists primarily for tests or benchmark scripts that intentionally
    mutate ``os.environ`` after the module was imported.
    """

    global _RUNTIME_CONFIG
    _RUNTIME_CONFIG = _read_runtime_config_from_env()


def _runtime_config_from_deberta_config(config: Any | None) -> FlashDebertaRuntimeConfig:
    """Resolve flash runtime policy from a native DeBERTa config object.

    :param Any | None config: Optional backbone config object.
    :return FlashDebertaRuntimeConfig: Instance-local runtime policy.
    """

    raw = getattr(config, "hf_flash", None) if config is not None else None
    if isinstance(raw, dict):
        getter = raw.get
    else:

        def getter(key: str, default: Any = None) -> Any:
            """Read one attribute-style config value.

            :param str key: Attribute name.
            :param Any default: Default value.
            :return Any: Resolved value.
            """

            return getattr(raw, key, default) if raw is not None else default

    return FlashDebertaRuntimeConfig(
        force_varlen=bool(getter("force_varlen", False)),
        varlen_min_seq_len=_optional_int(getter("varlen_min_seq_len", None)),
        docblock_bias_seq_len=_optional_int(getter("docblock_bias_seq_len", None)),
        local_bias_seq_len=_optional_int(getter("local_bias_seq_len", None)),
        local_bias_max_batch_size=_optional_int(getter("local_bias_max_batch_size", None)),
        eager_dense_max_seq_len=max(0, int(getter("eager_dense_max_seq_len", 0))),
        kernel_overrides_path=getter("kernel_overrides_path", None),
        enable_debug_stats=bool(_RUNTIME_CONFIG.enable_debug_stats),
        warn_fallbacks=bool(_RUNTIME_CONFIG.warn_fallbacks),
    )


def flashdeberta_import_error() -> Exception | None:
    """Return the fixed-kernel import error, if any.

    Flash attention construction requires the fixed kernel. The varlen kernel is
    optional and only affects padding-heavy workloads.

    :return Exception | None: Stored fixed-kernel import failure, if one occurred.
    """

    version_error = flashdeberta_version_error()
    if version_error is not None:
        return version_error
    return flashdeberta_fixed_import_error()


def flashdeberta_stats_snapshot() -> dict[str, int]:
    """Return a copy of eager/debug-only flash path counters.

    These counters are intentionally disabled by default and are not intended as
    a normal-training metric source.

    :return dict[str, int]: Counter snapshot keyed by path or fallback name.
    """

    return dict(_FLASH_STATS)


def reset_flashdeberta_stats() -> None:
    """Reset eager/debug-only flash path counters."""

    _FLASH_STATS.clear()


def _record_stat(name: str, value: int = 1) -> None:
    """Increment an eager/debug-only flash path counter.

    This is a strict no-op inside compiled graphs.

    :param str name: Counter key.
    :param int value: Increment amount, defaults to ``1``.
    """

    if not _RUNTIME_CONFIG.enable_debug_stats:
        return
    if is_torch_compiling():
        return
    _FLASH_STATS[name] += int(value)


def _mask4d_to_seqlens(attention_mask: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Convert a padding-style keep mask into per-example sequence lengths.

    Supported mask shapes:
    - ``(B, S)``
    - ``(B, 1, 1, S)``

    :param torch.Tensor attention_mask: Padding-style keep mask.
    :param int seq_len: Expected sequence length.
    :return torch.Tensor: Per-example sequence lengths with dtype ``int32``.
    """

    key_mask = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
    return key_mask.sum(dim=-1, dtype=torch.int32)


def _pairwise_mask_to_4d_keep_mask(
    attention_mask: torch.Tensor,
    *,
    query_len: int,
    key_len: int,
) -> torch.Tensor:
    """Extract a canonical pairwise keep mask ``(B,1,Q,K)``.

    :param torch.Tensor attention_mask: Pairwise keep-mask tensor.
    :param int query_len: Expected query length.
    :param int key_len: Expected key length.
    :raises ValueError: If the mask is not pairwise.
    :return torch.Tensor: Boolean keep mask with shape ``(B, 1, Q, K)``.
    """

    mask = normalize_keep_mask(attention_mask)
    if mask.ndim == 3:
        pairwise = mask.unsqueeze(1)
    elif mask.ndim == 4:
        pairwise = mask
    else:
        raise ValueError("FlashDeBERTa pairwise masks must be shaped (B,Q,K) or (B,1,Q,K).")
    if tuple(pairwise.shape[-2:]) != (int(query_len), int(key_len)):
        raise ValueError(
            "FlashDeBERTa pairwise masks must match the attention shape; "
            f"got mask={tuple(pairwise.shape)} expected=(*,{int(query_len)},{int(key_len)})."
        )
    return pairwise.to(dtype=torch.bool)


def _should_use_varlen(
    *,
    attention_mask: torch.Tensor | None,
    seq_len: int,
    runtime_config: FlashDebertaRuntimeConfig | None = None,
) -> bool:
    """Return whether the varlen kernel should be used for this call.

    This deliberately does not inspect tensor contents. In this repository's
    training path, the collator already drops all-ones masks, so
    ``attention_mask is None`` is the dense signal and ``attention_mask is not None``
    is the padded signal. When ``torch.compile`` is active we require the
    opaque custom-op varlen wrapper so Dynamo does not trace into
    FlashDeBERTa's Python/Triton launcher.

    Route policy is shared with the training loop through
    :func:`flash_padding_route`. This fallback only runs when no
    ``flash_meta`` route hint was provided, and it cannot know batch density
    without a host sync, so density-gated table buckets resolve to their
    density-agnostic fallback rows here; the training path passes the
    density-aware hint through ``FlashBatchMeta`` instead.

    :param torch.Tensor | None attention_mask: Optional attention mask.
    :param int seq_len: Sequence length for the current call.
    :param FlashDebertaRuntimeConfig | None runtime_config: Optional instance-local runtime policy.
    :return bool: True when the varlen kernel should run.
    """

    if attention_mask is None:
        return False

    if is_torch_compiling() and not flashdeberta_compiled_varlen_available():
        return False

    cfg = runtime_config or _RUNTIME_CONFIG
    return (
        flash_padding_route(
            seq_len=int(seq_len),
            force_varlen=bool(cfg.force_varlen),
            varlen_min_seq_len=cfg.varlen_min_seq_len,
            compute_capability=device_compute_capability(attention_mask.device),
        )
        == "varlen"
    )


def _resolve_docblock_scalars(
    flash_meta: FlashBatchMeta | None,
) -> tuple[torch.Tensor | int | None, torch.Tensor | int | None, torch.Tensor | int | None]:
    """Resolve doc-block active-token/segment scalars from flash metadata.

    Compiled routes prefer the CPU scalar tensors; eager metadata falls back
    to host integers.

    :param FlashBatchMeta | None flash_meta: Optional flash metadata bundle.
    :return tuple[torch.Tensor | int | None, torch.Tensor | int | None, torch.Tensor | int | None]:
        ``(active_tokens, num_segments, max_segment_length)`` as scalar
        tensors, host ints, or ``None`` per missing field.
    """

    if flash_meta is None:
        return None, None, None
    active_tokens = (
        flash_meta.active_tokens_scalar
        if flash_meta.active_tokens_scalar is not None
        else flash_meta.active_tokens_host
    )
    num_segments = (
        flash_meta.doc_num_segments_scalar
        if flash_meta.doc_num_segments_scalar is not None
        else flash_meta.doc_num_segments_host
    )
    max_segment_length = (
        flash_meta.doc_max_segment_length_scalar
        if flash_meta.doc_max_segment_length_scalar is not None
        else flash_meta.doc_max_segment_length_host
    )
    return active_tokens, num_segments, max_segment_length


def _seqlens_to_mask_2d(seq_lengths: torch.Tensor, *, seq_len: int) -> torch.Tensor:
    """Build a canonical prefix-padding keep mask from per-example lengths.

    :param torch.Tensor seq_lengths: Per-example active lengths with shape ``(B,)``.
    :param int seq_len: Padded sequence length.
    :return torch.Tensor: Boolean keep mask in ``(B,S)`` layout.
    """

    clipped = seq_lengths.to(dtype=torch.int32).clamp(min=0, max=int(seq_len))
    positions = torch.arange(int(seq_len), device=clipped.device, dtype=torch.int32)
    return positions.unsqueeze(0) < clipped.unsqueeze(-1)


def _dense_bucket_index_cache_key(
    *,
    seq_len: int,
    position_buckets: int,
    max_relative_distance: int,
    device: torch.device,
) -> tuple[int, int, int, str, int | None]:
    """Return the cache key for one dense bucket-index tensor.

    :param int seq_len: Sequence length.
    :param int position_buckets: Relative bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param torch.device device: Device for the cached tensor.
    :return tuple[int, int, int, str, int | None]: Cache key.
    """

    return (int(seq_len), int(position_buckets), int(max_relative_distance), str(device.type), device.index)


def _dense_bucket_index_tensor(
    *,
    seq_len: int,
    position_buckets: int,
    max_relative_distance: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a cached dense DeBERTa bucket-index matrix ``(S, S)``.

    The log-bucket math is shared with the eager backbone through
    :func:`deberta.modeling.deberta_v2_native.build_relative_position`; this
    helper only adds the ``+position_buckets`` embedding-row offset expected
    by the flash bias kernels. Sharing one implementation keeps eager-vs-flash
    relative-position parity by construction.

    :param int seq_len: Sequence length.
    :param int position_buckets: Relative bucket count.
    :param int max_relative_distance: Maximum relative distance.
    :param torch.device device: Target device.
    :return torch.Tensor: Dense int64 bucket-index tensor.
    """

    max_rel = int(max_relative_distance)
    if max_rel <= 0:
        max_rel = int(seq_len)
    key = _dense_bucket_index_cache_key(
        seq_len=seq_len,
        position_buckets=position_buckets,
        max_relative_distance=max_rel,
        device=device,
    )
    cached = _DENSE_BUCKET_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    bucket_index = (
        _build_relative_position(
            int(seq_len),
            int(seq_len),
            bucket_size=int(position_buckets),
            max_position=max_rel,
            device=device,
        )
        .add_(int(position_buckets))
        .clamp_(0, 2 * int(position_buckets) - 1)
    )
    _DENSE_BUCKET_INDEX_CACHE[key] = bucket_index
    if len(_DENSE_BUCKET_INDEX_CACHE) > 8:
        stale_keys = list(_DENSE_BUCKET_INDEX_CACHE.keys())[:4]
        for stale_key in stale_keys:
            _DENSE_BUCKET_INDEX_CACHE.pop(stale_key, None)
    return bucket_index


class FlashDisentangledSelfAttention(_EagerDisentangledSelfAttention):
    """FlashDeBERTa-backed variant of native disentangled self-attention."""

    _warned_reasons: set[str] = set()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize after enforcing the supported FlashDeBERTa package pin.

        :param Any args: Positional constructor arguments.
        :param Any kwargs: Keyword constructor arguments.
        """

        require_flashdeberta_version()
        config = args[0] if args else kwargs.get("config")
        self._runtime_config = _runtime_config_from_deberta_config(config)
        configure_flashdeberta_kernel_overrides(self._runtime_config.kernel_overrides_path)
        super().__init__(*args, **kwargs)

    @classmethod
    def _warn_once(cls, *, reason: str, message: str) -> None:
        """Emit one warning per process for a fallback reason.

        Warnings are skipped while executing inside compiled graphs.

        :param str reason: Stable fallback key.
        :param str message: Warning text to emit.
        """

        if not _RUNTIME_CONFIG.warn_fallbacks:
            return
        if is_torch_compiling():
            return
        if reason in cls._warned_reasons:
            return
        cls._warned_reasons.add(reason)
        warnings.warn(message, stacklevel=2)

    def _fallback_reason(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        query_states: torch.Tensor,
        rel_embeddings: torch.Tensor | None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> tuple[str, str] | None:
        """Return the first reason this call should use eager attention.

        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param torch.Tensor query_states: Query hidden states.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional out-of-graph routing metadata.
        :return tuple[str, str] | None: Fallback reason key and message, or ``None``.
        """

        query_len = int(query_states.shape[-2])
        key_len = int(hidden_states.shape[-2])
        dropout_p = float(getattr(self.dropout, "p", 0.0))
        normalized_route = flash_meta.normalized_route_hint() if flash_meta is not None else None

        dense_eager_limit = int(self._runtime_config.eager_dense_max_seq_len)
        if dense_eager_limit > 0 and attention_mask is None and key_len <= dense_eager_limit:
            return (
                "dense_short_policy",
                "FlashDeBERTa dense short-sequence policy selected eager attention for this batch.",
            )
        if (
            attention_mask is not None
            and is_pairwise_mask(attention_mask, query_len=query_len, key_len=key_len)
            and normalized_route != "docblock_bias"
        ):
            return (
                "pairwise_mask",
                "FlashDeBERTa attention does not support pairwise (B,S,S)/(B,1,S,S) masks; using eager attention.",
            )
        if "p2p" in self.pos_att_type:
            return (
                "p2p",
                "FlashDeBERTa attention does not support pos_att_type='p2p'; using eager attention.",
            )
        if self.training and dropout_p > 0.0:
            return (
                "attention_dropout",
                "FlashDeBERTa attention requires attention_probs_dropout_prob=0.0 during training; using eager attention.",
            )
        if not self.relative_attention:
            return (
                "relative_attention_disabled",
                "FlashDeBERTa integration currently targets relative-attention DeBERTa configs; using eager attention.",
            )
        if rel_embeddings is None:
            return (
                "missing_rel_embeddings",
                "FlashDeBERTa attention requires relative embeddings for the current config; using eager attention.",
            )
        if int(self.position_buckets) <= 0:
            return (
                "missing_position_buckets",
                "FlashDeBERTa attention requires config.position_buckets > 0; using eager attention.",
            )
        fixed_import_error = flashdeberta_fixed_import_error()
        if fixed_import_error is not None:
            detail = str(fixed_import_error)
            return (
                "missing_flashdeberta",
                f"FlashDeBERTa fixed-length attention is unavailable ({detail}); using eager attention.",
            )
        if hidden_states.device.type != "cuda":
            return (
                "device",
                "FlashDeBERTa attention requires CUDA tensors; using eager attention.",
            )
        if query_len != key_len:
            return (
                "query_key_length_mismatch",
                "FlashDeBERTa attention integration currently expects self-attention with matching query/key lengths; using eager attention.",
            )
        return None

    def _projected_qkv_fallback_reason(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> tuple[str, str] | None:
        """Return whether projected QKV tensors are unsupported by the flash kernels.

        :param torch.Tensor query_layer: Projected query tensor.
        :param torch.Tensor key_layer: Projected key tensor.
        :param torch.Tensor value_layer: Projected value tensor.
        :return tuple[str, str] | None: Fallback reason key and message, or ``None``.
        """

        qkv_dtypes = {query_layer.dtype, key_layer.dtype, value_layer.dtype}
        if len(qkv_dtypes) != 1:
            return (
                "mixed_qkv_dtype",
                "FlashDeBERTa attention requires projected query/key/value tensors to share one dtype; using eager attention.",
            )

        qkv_dtype = query_layer.dtype
        if qkv_dtype not in _FLASH_SUPPORTED_DTYPES:
            return (
                "dtype",
                "FlashDeBERTa attention currently supports float16/bfloat16 projected QKV activations; using eager attention.",
            )
        return None

    def _shape_varlen(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape projection output into varlen-friendly multi-head layout.

        The padded-varlen path operates more naturally in ``(B, S, H, D)``
        layout, which lets the custom op flatten ``(B*S)`` directly and use one
        flat gather/scatter instead of per-token advanced indexing over a
        ``(B, H, S, D)`` transpose.

        :param torch.Tensor x: Projected tensor with trailing dim ``all_head_size``.
        :return torch.Tensor: Tensor with shape ``(B, S, H, D)``.
        """

        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_attention_heads, self.attention_head_size)

    def _flash_fixed(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor | None,
        flash_meta: FlashBatchMeta | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run the fixed-length FlashDeBERTa kernel.

        :param torch.Tensor query_layer: Projected queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, H, S, D)`` layout.
        :param torch.Tensor | None attention_mask: Optional padding mask.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        seq_lengths = flash_meta.seq_lengths if flash_meta is not None else None
        if seq_lengths is None and attention_mask is not None:
            seq_lengths = _mask4d_to_seqlens(attention_mask, seq_len=int(key_layer.shape[-2]))
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_fixed_calls")
        return flashdeberta_fixed(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            seq_lengths=seq_lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            causal=False,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
        )

    def _should_use_local_bias(
        self,
        *,
        attention_mask: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        device: torch.device | None = None,
    ) -> bool:
        """Return whether the dense local-bias flash path should run.

        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param int batch_size: Runtime batch size.
        :param int seq_len: Sequence length.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param torch.device | None device: Activation device for capability-scoped table rows.
        :return bool: True when dense local-bias flash should run.
        """

        if attention_mask is not None:
            return False
        if not self.training:
            return False
        seq_bucket = flash_seq_bucket(seq_len=int(seq_len))
        local_bias_policy = flash_route_policy(
            policy="local_bias",
            seq_bucket=seq_bucket,
            compute_capability=device_compute_capability(device) if device is not None else None,
        )
        local_bias_max_batch_size = self._runtime_config.local_bias_max_batch_size
        if local_bias_max_batch_size is None:
            if local_bias_policy is None or str(local_bias_policy.get("choice", "")).strip() != "local_bias":
                return False
            try:
                local_bias_max_batch_size = int(local_bias_policy.get("max_batch_size", 0))
            except Exception:
                local_bias_max_batch_size = 0
        if local_bias_max_batch_size <= 0 or int(batch_size) > local_bias_max_batch_size:
            return False
        local_bias_seq_len = self._runtime_config.local_bias_seq_len
        if local_bias_seq_len is None:
            if local_bias_policy is None or str(local_bias_policy.get("choice", "")).strip() != "local_bias":
                return False
        elif int(local_bias_seq_len) <= 0 or int(seq_len) != int(local_bias_seq_len):
            return False
        if pos_key is None and pos_query is None:
            return False
        if flashdeberta_bias_import_error() is not None:
            return False
        if is_torch_compiling() and not (
            flashdeberta_compiled_bias_available() and flashdeberta_compiled_position_bias_available()
        ):
            return False
        return True

    def _flash_local_bias(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run dense local-bias flash attention.

        :param torch.Tensor query_layer: Projected queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, H, S, D)`` layout.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        batch_size, num_heads, seq_len, _ = query_layer.shape
        bucket_index = _dense_bucket_index_tensor(
            seq_len=seq_len,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            device=query_layer.device,
        )
        del batch_size, num_heads
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_bias_calls")
        return flashdeberta_bias_from_positions(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=None,
            bias_scale=float(sm_scale),
            sm_scale=sm_scale,
            causal=False,
        )

    def _flash_varlen(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        flash_meta: FlashBatchMeta | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run padded varlen FlashDeBERTa attention.

        :param torch.Tensor query_layer: Projected queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, S, H, D)`` layout.
        :param torch.Tensor attention_mask: Padding-style keep mask.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, S, H, D)`` layout.
        """

        seq_len = int(query_layer.shape[1])
        seq_lengths = flash_meta.seq_lengths if flash_meta is not None else None
        if seq_lengths is not None:
            mask_2d = _seqlens_to_mask_2d(seq_lengths, seq_len=seq_len)
        else:
            mask_2d = mask_to_2d_keep_mask(attention_mask, seq_len=seq_len)
        out = flashdeberta_varlen_padded(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            attention_mask_2d=mask_2d,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            causal=False,
        )
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_varlen_calls")
        return out

    def _flash_docblock(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        flash_meta: FlashBatchMeta,
        active_tokens: torch.Tensor | int | None,
        doc_num_segments: torch.Tensor | int | None,
        doc_max_segment_length: torch.Tensor | int | None,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run doc-block-aware FlashDeBERTa attention over packed document segments.

        :param torch.Tensor query_layer: Projected queries in ``(B, S, H, D)`` layout.
        :param torch.Tensor key_layer: Projected keys in ``(B, S, H, D)`` layout.
        :param torch.Tensor value_layer: Projected values in ``(B, S, H, D)`` layout.
        :param FlashBatchMeta flash_meta: FlashDeBERTa doc-block metadata bundle.
        :param torch.Tensor | int | None active_tokens: Resolved active-token scalar.
        :param torch.Tensor | int | None doc_num_segments: Resolved segment-count scalar.
        :param torch.Tensor | int | None doc_max_segment_length: Resolved max-segment scalar.
        :param torch.Tensor | None pos_key: Optional c2p term.
        :param torch.Tensor | None pos_query: Optional p2c term.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, S, H, D)`` layout.
        """

        if (
            flash_meta.doc_segment_offsets is None
            or flash_meta.doc_segment_lengths is None
            or flash_meta.doc_cu_seqlens is None
            or active_tokens is None
            or doc_num_segments is None
            or doc_max_segment_length is None
        ):
            raise RuntimeError("Doc-block flash route requires complete FlashBatchMeta.")
        out = flashdeberta_docblock(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            segment_offsets=flash_meta.doc_segment_offsets,
            segment_lengths=flash_meta.doc_segment_lengths,
            cu_seqlens=flash_meta.doc_cu_seqlens,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=sm_scale,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            num_segments=doc_num_segments,
            max_seqlen=doc_max_segment_length,
            total_tokens=active_tokens,
            causal=False,
        )
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_docblock_calls")
        return out

    def _flash_docblock_bias(
        self,
        *,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        pos_key: torch.Tensor | None,
        pos_query: torch.Tensor | None,
        sm_scale: float,
    ) -> torch.Tensor:
        """Run dense flash-with-bias for short packed doc-block batches.

        :param torch.Tensor query_layer: Queries in ``(B, H, S, D)`` layout.
        :param torch.Tensor key_layer: Keys in ``(B, H, S, D)`` layout.
        :param torch.Tensor value_layer: Values in ``(B, H, S, D)`` layout.
        :param torch.Tensor attention_mask: Pairwise keep-mask tensor.
        :param torch.Tensor | None pos_key: Optional c2p term in ``(B, H, S, P)`` layout.
        :param torch.Tensor | None pos_query: Optional p2c term in ``(B, H, S, P)`` layout.
        :param float sm_scale: Softmax scale.
        :return torch.Tensor: Flash output in ``(B, H, S, D)`` layout.
        """

        batch_size, num_heads, seq_len, _ = query_layer.shape
        keep_mask = _pairwise_mask_to_4d_keep_mask(
            attention_mask,
            query_len=seq_len,
            key_len=int(key_layer.shape[-2]),
        )
        del batch_size, num_heads
        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_docblock_bias_calls")
        bucket_index = _dense_bucket_index_tensor(
            seq_len=seq_len,
            position_buckets=int(self.position_buckets),
            max_relative_distance=int(self.max_relative_positions),
            device=query_layer.device,
        )
        return flashdeberta_bias_from_positions(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
            pos_key=pos_key,
            pos_query=pos_query,
            bucket_index=bucket_index,
            keep_mask=keep_mask,
            bias_scale=float(sm_scale),
            sm_scale=sm_scale,
            causal=False,
        )

    def _eager_fallback_attention_mask(
        self,
        *,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        flash_meta: FlashBatchMeta | None,
    ) -> torch.Tensor | None:
        """Rebuild pairwise doc-block masks for eager fallback when needed.

        :param torch.Tensor | None attention_mask: Original attention mask.
        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor query_states: Query hidden states.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return torch.Tensor | None: Eager-compatible attention mask.
        """

        if flash_meta is None or flash_meta.normalized_route_hint() != "docblock":
            return attention_mask
        if flash_meta.doc_segment_offsets is None or flash_meta.doc_segment_lengths is None:
            return attention_mask
        key_len = int(hidden_states.shape[-2])
        query_len = int(query_states.shape[-2])
        if query_len != key_len:
            return attention_mask
        doc_ids = doc_ids_from_segments(
            offsets=flash_meta.doc_segment_offsets,
            lengths=flash_meta.doc_segment_lengths,
            batch_size=int(hidden_states.shape[0]),
            seq_len=key_len,
        )
        return build_doc_block_mask(doc_ids).unsqueeze(1)

    def _eager_forward_fallback(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool,
        query_states: torch.Tensor,
        rel_embeddings: torch.Tensor | None,
        flash_meta: FlashBatchMeta | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run eager attention with doc-block fallback masks rebuilt only on demand.

        :param torch.Tensor hidden_states: Key/value hidden states.
        :param torch.Tensor | None attention_mask: Original attention mask.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor query_states: Query hidden states.
        :param torch.Tensor | None rel_embeddings: Relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Eager attention output and optional probs.
        """

        eager_attention_mask = self._eager_fallback_attention_mask(
            attention_mask=attention_mask,
            hidden_states=hidden_states,
            query_states=query_states,
            flash_meta=flash_meta,
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=eager_attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=None,
            rel_embeddings=rel_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: bool = False,
        query_states: torch.Tensor | None = None,
        relative_pos: torch.Tensor | None = None,
        rel_embeddings: torch.Tensor | None = None,
        flash_meta: FlashBatchMeta | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run flash-backed attention when the runtime contract is compatible.

        :param torch.Tensor hidden_states: Input hidden states.
        :param torch.Tensor | None attention_mask: Optional attention mask.
        :param bool output_attentions: Whether to return attention probabilities.
        :param torch.Tensor | None query_states: Optional query states.
        :param torch.Tensor | None relative_pos: Optional relative-position ids.
        :param torch.Tensor | None rel_embeddings: Optional relative embedding table.
        :param FlashBatchMeta | None flash_meta: Optional FlashDeBERTa metadata bundle.
        :return tuple[torch.Tensor, torch.Tensor | None]: Attention output and optional probs.
        """

        del relative_pos  # The flash kernels compute relative positions on device.

        if query_states is None:
            query_states = hidden_states

        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("forward_calls")

        reason = self._fallback_reason(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            query_states=query_states,
            rel_embeddings=rel_embeddings,
            flash_meta=flash_meta,
        )
        if reason is not None:
            key, message = reason
            if _RUNTIME_CONFIG.enable_debug_stats:
                _record_stat("fallback_calls")
                _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
            # The encoder-level get_rel_pos patch suppresses the shared (S,S)
            # allocation globally. Unsupported correctness fallbacks rebuild
            # relative-position bias inside eager attention instead.
            return self._eager_forward_fallback(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                rel_embeddings=rel_embeddings,
                flash_meta=flash_meta,
            )

        if flashdeberta_fixed_import_error() is not None:  # pragma: no cover - guarded above
            raise RuntimeError("FlashDeBERTa operator import unexpectedly unavailable during flash forward.")

        model_dtype = hidden_states.dtype
        bsz, query_len, _ = query_states.shape
        normalized_route = flash_meta.normalized_route_hint() if flash_meta is not None else None
        use_docblock_bias = normalized_route == "docblock_bias"
        use_docblock = normalized_route == "docblock"
        if use_docblock:
            use_varlen = True
        elif use_docblock_bias:
            use_varlen = False
        elif normalized_route == "varlen":
            use_varlen = True
        elif normalized_route in {"fixed", "dense", "pairwise"}:
            use_varlen = False
        else:
            use_varlen = _should_use_varlen(
                attention_mask=attention_mask,
                seq_len=int(hidden_states.shape[-2]),
                runtime_config=self._runtime_config,
            )

        if use_docblock_bias:
            if attention_mask is None or not is_pairwise_mask(
                attention_mask,
                query_len=query_len,
                key_len=int(hidden_states.shape[-2]),
            ):
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_bias_mask")
                self._warn_once(
                    reason="docblock_bias_mask",
                    message=(
                        "FlashDeBERTa dense doc-block bias routing requires a pairwise keep mask; "
                        "using eager attention."
                    ),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )
            bias_import_error = flashdeberta_bias_import_error()
            if bias_import_error is not None:
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_bias_missing")
                self._warn_once(
                    reason="docblock_bias_missing",
                    message=("FlashDeBERTa dense doc-block bias path is unavailable; using eager attention."),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )
            if is_torch_compiling() and not flashdeberta_compiled_bias_available():
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_bias_compile")
                self._warn_once(
                    reason="docblock_bias_compile",
                    message=(
                        "FlashDeBERTa dense doc-block bias path is not compile-visible on this build; "
                        "using eager attention."
                    ),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )

        if use_docblock:
            docblock_active_tokens, docblock_num_segments, docblock_max_segment = _resolve_docblock_scalars(
                flash_meta
            )
            if (
                flash_meta is None
                or flash_meta.doc_segment_offsets is None
                or flash_meta.doc_segment_lengths is None
                or flash_meta.doc_cu_seqlens is None
                or docblock_active_tokens is None
                or docblock_num_segments is None
                or docblock_max_segment is None
            ):
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_metadata_missing")
                self._warn_once(
                    reason="docblock_metadata_missing",
                    message=(
                        "FlashDeBERTa doc-block routing requires precomputed segment metadata and host stats; "
                        "using eager attention."
                    ),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )
            docblock_import_error = flashdeberta_docblock_import_error()
            if docblock_import_error is not None:
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_missing")
                self._warn_once(
                    reason="docblock_missing",
                    message=("FlashDeBERTa doc-block flash path is unavailable; using eager attention."),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )
            if is_torch_compiling() and not flashdeberta_compiled_docblock_available():
                if _RUNTIME_CONFIG.enable_debug_stats:
                    _record_stat("fallback_calls")
                    _record_stat("fallback_docblock_compile")
                self._warn_once(
                    reason="docblock_compile",
                    message=(
                        "FlashDeBERTa doc-block flash path is not compile-visible on this build; "
                        "using eager attention."
                    ),
                )
                return self._eager_forward_fallback(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    query_states=query_states,
                    rel_embeddings=rel_embeddings,
                    flash_meta=flash_meta,
                )

        if use_varlen:
            query_layer = self._shape_varlen(self.query_proj(query_states))
            key_layer = self._shape_varlen(self.key_proj(hidden_states))
            value_layer = self._shape_varlen(self.value_proj(hidden_states))
        else:
            query_layer = self._shape(self.query_proj(query_states)).contiguous()
            key_layer = self._shape(self.key_proj(hidden_states)).contiguous()
            value_layer = self._shape(self.value_proj(hidden_states)).contiguous()

        projected_reason = self._projected_qkv_fallback_reason(
            query_layer=query_layer,
            key_layer=key_layer,
            value_layer=value_layer,
        )
        if projected_reason is not None:
            key, message = projected_reason
            if _RUNTIME_CONFIG.enable_debug_stats:
                _record_stat("fallback_calls")
                _record_stat(f"fallback_{key}")
            self._warn_once(reason=key, message=message)
            # Keep the same eager fallback contract here for dtype/layout
            # mismatches instead of reviving the encoder-wide relative_pos tensor.
            return self._eager_forward_fallback(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                rel_embeddings=rel_embeddings,
                flash_meta=flash_meta,
            )

        pos_key: torch.Tensor | None = None
        pos_query: torch.Tensor | None = None
        rel_embeddings = self.pos_dropout(rel_embeddings)

        if "c2p" in self.pos_att_type:
            pos_key_layer = self._project_rel(rel_embeddings, use_query=False).to(dtype=query_layer.dtype)
            if use_varlen:
                pos_key = torch.einsum("bshd,hpd->bshp", query_layer, pos_key_layer)
            else:
                pos_key = torch.matmul(query_layer, pos_key_layer.unsqueeze(0).transpose(-1, -2))

        if "p2c" in self.pos_att_type:
            pos_query_layer = self._project_rel(rel_embeddings, use_query=True).to(dtype=key_layer.dtype)
            if use_varlen:
                pos_query = torch.einsum("bshd,hpd->bshp", key_layer, pos_query_layer)
            else:
                pos_query = torch.matmul(key_layer, pos_query_layer.unsqueeze(0).transpose(-1, -2))

        scale_factor = 1
        if pos_key is not None:
            scale_factor += 1
        if pos_query is not None:
            scale_factor += 1
        sm_scale = 1.0 / math.sqrt(float(self.attention_head_size * scale_factor))

        if _RUNTIME_CONFIG.enable_debug_stats:
            _record_stat("flash_eligible_calls")
        if use_docblock_bias:
            output = self._flash_docblock_bias(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        elif use_docblock:
            output = self._flash_docblock(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                flash_meta=flash_meta,
                active_tokens=docblock_active_tokens,
                doc_num_segments=docblock_num_segments,
                doc_max_segment_length=docblock_max_segment,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.contiguous().view(bsz, query_len, self.all_head_size)
        elif use_varlen:
            output = self._flash_varlen(
                query_layer=query_layer,
                key_layer=key_layer,
                value_layer=value_layer,
                attention_mask=attention_mask,
                flash_meta=flash_meta,
                pos_key=pos_key,
                pos_query=pos_query,
                sm_scale=sm_scale,
            )
            output = output.contiguous().view(bsz, query_len, self.all_head_size)
        else:
            if self._should_use_local_bias(
                attention_mask=attention_mask,
                batch_size=bsz,
                seq_len=query_len,
                pos_key=pos_key,
                pos_query=pos_query,
                device=query_layer.device,
            ):
                output = self._flash_local_bias(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    pos_key=pos_key,
                    pos_query=pos_query,
                    sm_scale=sm_scale,
                )
            else:
                output = self._flash_fixed(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    attention_mask=attention_mask,
                    flash_meta=flash_meta,
                    pos_key=pos_key,
                    pos_query=pos_query,
                    sm_scale=sm_scale,
                )
            output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.all_head_size)
        output = output.to(dtype=model_dtype)

        if output_attentions:
            return output, None
        return output, None


__all__ = [
    "FlashDisentangledSelfAttention",
    "_mask4d_to_seqlens",
    "flashdeberta_import_error",
    "flashdeberta_stats_snapshot",
    "refresh_flashdeberta_runtime_config_from_env",
    "reset_flashdeberta_stats",
]
