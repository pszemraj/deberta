"""Shared torch/CUDA infrastructure helpers for FlashDeBERTa custom-op modules."""

from __future__ import annotations

from collections import OrderedDict
from functools import cache
from typing import Any, TypeVar

import torch

try:  # pragma: no cover - optional Triton dependency
    import triton as _triton
    import triton.language as _tl
except Exception:  # pragma: no cover - optional Triton dependency
    _triton = None
    _tl = None

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class BoundedLRUCache(OrderedDict[_KeyT, _ValueT]):
    """Ordered mapping that refreshes reads and evicts least-recently-used entries.

    :param int max_entries: Maximum number of retained entries.
    :raises ValueError: If ``max_entries`` is not positive.
    """

    def __init__(self, *, max_entries: int) -> None:
        """Initialize a bounded least-recently-used mapping."""

        if int(max_entries) <= 0:
            raise ValueError(f"max_entries must be positive; got {max_entries}.")
        super().__init__()
        self.max_entries = int(max_entries)

    def __getitem__(self, key: _KeyT) -> _ValueT:
        """Return and refresh one cached value."""

        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def get(self, key: _KeyT, default: Any = None) -> _ValueT | Any:
        """Return and refresh one cached value, or ``default`` on a miss.

        :param _KeyT key: Cache key to look up.
        :param Any default: Value returned on a miss, defaults to None.
        :return _ValueT | Any: Cached value or ``default``.
        """

        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: _KeyT, value: _ValueT) -> None:
        """Insert one value and evict least-recently-used entries past the bound."""

        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, value)
        while len(self) > self.max_entries:
            self.popitem(last=False)


TRITON_ROW_COPY_PREFIX_PACK = 0
TRITON_ROW_COPY_PREFIX_PACK_STRIDED = 1
TRITON_ROW_COPY_PREFIX_UNPACK = 2
TRITON_ROW_COPY_SEGMENT_PACK = 3
TRITON_ROW_COPY_SEGMENT_PACK_STRIDED = 4
TRITON_ROW_COPY_SEGMENT_UNPACK = 5


def lookup_registered_op(namespace: str, name: str) -> Any | None:
    """Return a previously registered custom op overload, if one exists.

    :param str namespace: Operator namespace.
    :param str name: Operator name.
    :return Any | None: Registered overload or ``None``.
    """

    ns = getattr(torch.ops, namespace, None)
    if ns is None or not hasattr(ns, name):
        return None
    op = getattr(ns, name)
    return getattr(op, "default", op)


def lookup_existing_op_pair(namespace: str, fwd_name: str, bwd_name: str) -> tuple[Any, Any] | None:
    """Return an already-registered forward/backward custom-op pair, if complete.

    :param str namespace: Operator namespace.
    :param str fwd_name: Forward operator name.
    :param str bwd_name: Backward operator name.
    :return tuple[Any, Any] | None: ``(forward, backward)`` overloads, or ``None``
        unless both are registered.
    """

    existing_forward = lookup_registered_op(namespace, fwd_name)
    existing_backward = lookup_registered_op(namespace, bwd_name)
    if existing_forward is not None and existing_backward is not None:
        return existing_forward, existing_backward
    return None


def kernel_dtype_name(dtype: torch.dtype) -> str:
    """Return a compact dtype name for tuning-table matching.

    :param torch.dtype dtype: Torch dtype.
    :return str: Dtype name without the ``torch.`` prefix.
    """

    return str(dtype).removeprefix("torch.")


def strides_or_zeros(tensor: torch.Tensor | None, count: int) -> tuple[int, ...]:
    """Return leading tensor strides or zeros for an absent optional tensor.

    :param torch.Tensor | None tensor: Optional tensor.
    :param int count: Number of leading strides to return.
    :return tuple[int, ...]: Tensor strides or a same-length zero tuple.
    """

    if tensor is None:
        return (0,) * int(count)
    return tuple(int(tensor.stride(index)) for index in range(int(count)))


def is_flash_attention_impl(value: object) -> bool:
    """Return whether an attention-implementation value selects FlashDeBERTa.

    :param object value: Raw config value.
    :return bool: True only for the normalized ``flash`` implementation name.
    """

    return str(value).strip().lower() == "flash"


def optional_triton_jit(fn: object) -> object:
    """Apply ``triton.jit`` only when Triton imported successfully.

    :param object fn: Kernel function.
    :return object: JIT kernel or unchanged function in no-Triton environments.
    """

    if _triton is None:
        return fn
    return _triton.jit(fn)


def traceable_triton_kernel(kernel: object) -> object:
    """Return a traceable Triton kernel wrapper when PyTorch exposes one.

    :param object kernel: Raw Triton kernel or autotuned wrapper.
    :return object: Traceable wrapper when available, otherwise ``kernel`` unchanged.
    """

    if not hasattr(torch, "library") or not hasattr(torch.library, "wrap_triton"):
        return kernel
    try:
        return torch.library.wrap_triton(kernel)
    except Exception:
        return kernel


@optional_triton_jit
def _triton_row_copy_kernel(
    input_a_ptr: None,
    input_b_ptr: None,
    input_c_ptr: None,
    output_a_ptr: None,
    output_b_ptr: None,
    output_c_ptr: None,
    offsets_ptr: None,
    lengths_ptr: None,
    cu_seqlens_ptr: None,
    stride_a_b: int,
    stride_a_s: int,
    stride_a_h: int,
    stride_a_f: int,
    stride_b_b: int,
    stride_b_s: int,
    stride_b_h: int,
    stride_b_f: int,
    stride_c_b: int,
    stride_c_s: int,
    stride_c_h: int,
    stride_c_f: int,
    seq_len: int,
    row_size: int,
    num_heads: int,
    feature_size: int,
    col_tiles: int,
    ARITY: _tl.constexpr,
    ADDRESS_MODE: _tl.constexpr,
    BLOCK_ROWS: _tl.constexpr,
    BLOCK_COLS: _tl.constexpr,
) -> None:
    """Copy one to three tensors between padded and ragged row layouts.

    :param Any input_a_ptr: First source tensor pointer.
    :param Any input_b_ptr: Second source tensor pointer, used when ``ARITY >= 2``.
    :param Any input_c_ptr: Third source tensor pointer, used when ``ARITY >= 3``.
    :param Any output_a_ptr: First destination tensor pointer.
    :param Any output_b_ptr: Second destination tensor pointer, used when ``ARITY >= 2``.
    :param Any output_c_ptr: Third destination tensor pointer, used when ``ARITY >= 3``.
    :param Any offsets_ptr: Flat padded-row offsets for segment modes.
    :param Any lengths_ptr: Active row count per metadata record.
    :param Any cu_seqlens_ptr: Cumulative packed-row offsets.
    :param Any stride_a_b: First-input batch stride for rank-4 strided modes.
    :param Any stride_a_s: First-input sequence stride for rank-4 strided modes.
    :param Any stride_a_h: First-input head stride for rank-4 strided modes.
    :param Any stride_a_f: First-input feature stride for rank-4 strided modes.
    :param Any stride_b_b: Second-input batch stride for rank-4 strided modes.
    :param Any stride_b_s: Second-input sequence stride for rank-4 strided modes.
    :param Any stride_b_h: Second-input head stride for rank-4 strided modes.
    :param Any stride_b_f: Second-input feature stride for rank-4 strided modes.
    :param Any stride_c_b: Third-input batch stride for rank-4 strided modes.
    :param Any stride_c_s: Third-input sequence stride for rank-4 strided modes.
    :param Any stride_c_h: Third-input head stride for rank-4 strided modes.
    :param Any stride_c_f: Third-input feature stride for rank-4 strided modes.
    :param Any seq_len: Padded sequence length.
    :param Any row_size: Flattened trailing row width.
    :param Any num_heads: Rank-4 head count for strided modes.
    :param Any feature_size: Rank-4 per-head width for strided modes.
    :param Any col_tiles: Feature-tile count per head for strided modes.
    :param Any ARITY: Compile-time source/destination tensor count.
    :param Any ADDRESS_MODE: Compile-time prefix/segment and pack/unpack addressing mode.
    :param Any BLOCK_ROWS: Compile-time row tile size.
    :param Any BLOCK_COLS: Compile-time column tile size.
    :return None: This Triton kernel writes directly to destination pointers.
    """

    record_idx = _tl.program_id(0)
    tile_row = _tl.program_id(1)
    tile_col_or_hf = _tl.program_id(2)
    row_offsets = tile_row * BLOCK_ROWS + _tl.arange(0, BLOCK_ROWS)

    is_prefix = ADDRESS_MODE <= 2
    is_pack = ADDRESS_MODE == 0 or ADDRESS_MODE == 1 or ADDRESS_MODE == 3 or ADDRESS_MODE == 4
    is_strided = ADDRESS_MODE == 1 or ADDRESS_MODE == 4

    length = _tl.load(lengths_ptr + record_idx)
    packed_base = _tl.load(cu_seqlens_ptr + record_idx)
    if is_prefix:
        padded_base = record_idx * seq_len
    else:
        padded_base = _tl.load(offsets_ptr + record_idx)

    if is_strided:
        head_idx = tile_col_or_hf // col_tiles
        tile_col = tile_col_or_hf % col_tiles
        col_offsets = tile_col * BLOCK_COLS + _tl.arange(0, BLOCK_COLS)
        padded_rows = padded_base + row_offsets
        batch_idx = padded_rows // seq_len
        seq_idx = padded_rows % seq_len
        packed_rows = packed_base + row_offsets
        packed_cols = head_idx * feature_size + col_offsets
        copy_mask = (
            (head_idx < num_heads) & (row_offsets[:, None] < length) & (col_offsets[None, :] < feature_size)
        )

        src_a_ptrs = (
            input_a_ptr
            + batch_idx[:, None] * stride_a_b
            + seq_idx[:, None] * stride_a_s
            + head_idx * stride_a_h
            + col_offsets[None, :] * stride_a_f
        )
        dst_a_ptrs = output_a_ptr + packed_rows[:, None] * row_size + packed_cols[None, :]
        values_a = _tl.load(src_a_ptrs, mask=copy_mask, other=0)
        _tl.store(dst_a_ptrs, values_a, mask=copy_mask)
        if ARITY >= 2:
            src_b_ptrs = (
                input_b_ptr
                + batch_idx[:, None] * stride_b_b
                + seq_idx[:, None] * stride_b_s
                + head_idx * stride_b_h
                + col_offsets[None, :] * stride_b_f
            )
            dst_b_ptrs = output_b_ptr + packed_rows[:, None] * row_size + packed_cols[None, :]
            values_b = _tl.load(src_b_ptrs, mask=copy_mask, other=0)
            _tl.store(dst_b_ptrs, values_b, mask=copy_mask)
        if ARITY >= 3:
            src_c_ptrs = (
                input_c_ptr
                + batch_idx[:, None] * stride_c_b
                + seq_idx[:, None] * stride_c_s
                + head_idx * stride_c_h
                + col_offsets[None, :] * stride_c_f
            )
            dst_c_ptrs = output_c_ptr + packed_rows[:, None] * row_size + packed_cols[None, :]
            values_c = _tl.load(src_c_ptrs, mask=copy_mask, other=0)
            _tl.store(dst_c_ptrs, values_c, mask=copy_mask)
        return

    col_offsets = tile_col_or_hf * BLOCK_COLS + _tl.arange(0, BLOCK_COLS)
    padded_rows = padded_base + row_offsets
    packed_rows = packed_base + row_offsets
    active_mask = (row_offsets[:, None] < length) & (col_offsets[None, :] < row_size)
    if is_pack:
        src_rows = padded_rows
        dst_rows = packed_rows
        store_mask = active_mask
    else:
        src_rows = packed_rows
        dst_rows = padded_rows
        if ADDRESS_MODE == 2:
            store_mask = (row_offsets[:, None] < seq_len) & (col_offsets[None, :] < row_size)
        else:
            store_mask = active_mask

    src_a_ptrs = input_a_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
    dst_a_ptrs = output_a_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
    values_a = _tl.load(src_a_ptrs, mask=active_mask, other=0)
    _tl.store(dst_a_ptrs, values_a, mask=store_mask)
    if ARITY >= 2:
        src_b_ptrs = input_b_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
        dst_b_ptrs = output_b_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
        values_b = _tl.load(src_b_ptrs, mask=active_mask, other=0)
        _tl.store(dst_b_ptrs, values_b, mask=store_mask)
    if ARITY >= 3:
        src_c_ptrs = input_c_ptr + src_rows[:, None] * row_size + col_offsets[None, :]
        dst_c_ptrs = output_c_ptr + dst_rows[:, None] * row_size + col_offsets[None, :]
        values_c = _tl.load(src_c_ptrs, mask=active_mask, other=0)
        _tl.store(dst_c_ptrs, values_c, mask=store_mask)


def launch_triton_row_copy(
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    *,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    address_mode: int,
    seq_len: int,
    row_size: int,
    max_rows: int,
    num_heads: int = 1,
    feature_size: int = 1,
    block_rows: int = 32,
    block_cols: int = 128,
    num_warps: int = 4,
    num_stages: int = 2,
) -> None:
    """Launch the shared one-to-three-tensor Triton row-copy kernel.

    :param tuple[torch.Tensor, ...] inputs: One to three source tensors.
    :param tuple[torch.Tensor, ...] outputs: Matching destination tensors.
    :param torch.Tensor offsets: Flat padded-row offsets for segment modes; ignored for prefix modes.
    :param torch.Tensor lengths: Active row count per prefix or segment.
    :param torch.Tensor cu_seqlens: Packed cumulative row offsets.
    :param int address_mode: One of the ``TRITON_ROW_COPY_*`` addressing modes.
    :param int seq_len: Padded sequence length.
    :param int row_size: Flattened trailing row width.
    :param int max_rows: Maximum rows processed by any metadata record.
    :param int num_heads: Rank-4 head count for strided pack modes, defaults to 1.
    :param int feature_size: Rank-4 per-head width for strided pack modes, defaults to 1.
    :param int block_rows: Triton row tile size, defaults to 32.
    :param int block_cols: Triton column tile size, defaults to 128.
    :param int num_warps: Triton launch warp count, defaults to 4.
    :param int num_stages: Triton launch pipeline stages, defaults to 2.
    :raises RuntimeError: If Triton is unavailable.
    :raises ValueError: If arity or rank-4 strided inputs are invalid.
    """

    if _triton is None:
        raise RuntimeError("Triton row-copy launch requested without Triton installed.")
    arity = len(inputs)
    if arity not in (1, 2, 3) or len(outputs) != arity:
        raise ValueError(f"Triton row-copy expects matching arity 1..3; got {arity} and {len(outputs)}.")

    strided = address_mode in (
        TRITON_ROW_COPY_PREFIX_PACK_STRIDED,
        TRITON_ROW_COPY_SEGMENT_PACK_STRIDED,
    )
    if strided and any(tensor.ndim != 4 for tensor in inputs):
        raise ValueError("Strided Triton row-copy inputs must have shape (B,S,H,F).")

    padded_inputs = inputs + (inputs[-1],) * (3 - arity)
    padded_outputs = outputs + (outputs[-1],) * (3 - arity)
    if strided:
        input_strides = tuple(tuple(int(value) for value in tensor.stride()) for tensor in padded_inputs)
        col_programs = int(num_heads) * _triton.cdiv(int(feature_size), int(block_cols))
    else:
        input_strides = ((0, 0, 0, 0),) * 3
        col_programs = _triton.cdiv(int(row_size), int(block_cols))

    grid = (
        int(lengths.shape[0]),
        _triton.cdiv(max(1, int(max_rows)), int(block_rows)),
        col_programs,
    )
    traceable_triton_kernel(_triton_row_copy_kernel)[grid](
        *padded_inputs,
        *padded_outputs,
        offsets,
        lengths,
        cu_seqlens,
        *input_strides[0],
        *input_strides[1],
        *input_strides[2],
        int(seq_len),
        int(row_size),
        int(num_heads),
        int(feature_size),
        _triton.cdiv(int(feature_size), int(block_cols)) if strided else 1,
        ARITY=arity,
        ADDRESS_MODE=int(address_mode),
        BLOCK_ROWS=int(block_rows),
        BLOCK_COLS=int(block_cols),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )


def flatten_padded_rows(
    tensor: torch.Tensor, *, context: str
) -> tuple[torch.Tensor, tuple[int, ...], int, int]:
    """Flatten a contiguous ``(B, S, ...)`` tensor into row-major ``(B*S, F)`` form.

    :param torch.Tensor tensor: Tensor with at least two leading dims ``(B, S)``.
    :param str context: Caller name used in validation errors (e.g. ``"Prefix-pack"``).
    :raises ValueError: If the tensor rank is less than two or non-contiguous.
    :return tuple[torch.Tensor, tuple[int, ...], int, int]:
        Flattened tensor, trailing shape, batch size, and padded sequence length.
    """

    if tensor.ndim < 2:
        raise ValueError(f"Expected tensor with leading (B,S) dims; got shape={tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{context} Triton path requires contiguous tensors.")
    batch_size = int(tensor.shape[0])
    seq_len = int(tensor.shape[1])
    trailing_shape = tuple(int(dim) for dim in tensor.shape[2:])
    flat = tensor.view(batch_size * seq_len, -1)
    return flat, trailing_shape, batch_size, seq_len


def require_matching_tensor_layout(
    reference: torch.Tensor,
    *others: torch.Tensor,
    context: str,
    minimum_rank: int = 1,
) -> tuple[int, ...]:
    """Require tensors to have the same shape and device.

    :param torch.Tensor reference: Tensor defining the expected layout.
    :param torch.Tensor others: Tensors that must match ``reference``.
    :param str context: Caller name included in validation errors.
    :param int minimum_rank: Minimum accepted tensor rank.
    :raises ValueError: If rank, shape, or device differs.
    :return tuple[int, ...]: Shared tensor shape.
    """

    shape = tuple(int(dim) for dim in reference.shape)
    if len(shape) < int(minimum_rank):
        raise ValueError(f"{context} expects rank >= {minimum_rank}; got shape={shape}")
    for tensor in others:
        other_shape = tuple(int(dim) for dim in tensor.shape)
        if other_shape != shape:
            raise ValueError(f"{context} shape mismatch: expected {shape}, got {other_shape}")
        if tensor.device != reference.device:
            raise ValueError(f"{context} device mismatch: expected {reference.device}, got {tensor.device}")
    return shape


def can_use_triton_pack(
    *,
    available: bool,
    tensor: torch.Tensor,
    metadata_tensors: tuple[torch.Tensor, ...],
) -> bool:
    """Return whether a Triton pack/unpack path is usable for one call.

    :param bool available: Whether the caller's Triton kernels imported.
    :param torch.Tensor tensor: Input or output tensor.
    :param tuple[torch.Tensor, ...] metadata_tensors: Metadata tensors that must
        live on the same device as ``tensor``.
    :return bool: True when the Triton path is usable.
    """

    if not available:
        return False
    if tensor.device.type != "cuda":
        return False
    if not tensor.is_contiguous():
        return False
    return all(meta.device == tensor.device for meta in metadata_tensors)


def keep_mask_head_stride(keep_mask: torch.Tensor | None, *, num_heads: int) -> int:
    """Return the Triton head stride for an optional rank-4 keep mask.

    Broadcast masks ``(B,1,Q,K)`` must use stride 0 so every head reads the
    shared plane (their ``stride(1)`` is nonzero even though the dim is size
    1); per-head masks ``(B,H,Q,K)`` use their real head stride.

    :param torch.Tensor | None keep_mask: Optional rank-4 keep mask.
    :param int num_heads: Attention head count for the launch.
    :raises ValueError: If the mask head dimension is neither 1 nor ``num_heads``.
    :return int: Head stride to pass to Triton mask loads.
    """

    if keep_mask is None:
        return 0
    mask_heads = int(keep_mask.shape[1])
    if mask_heads == 1:
        return 0
    if mask_heads == int(num_heads):
        return int(keep_mask.stride(1))
    raise ValueError(
        f"keep_mask head dimension must be 1 or num_heads={int(num_heads)}; "
        f"got shape={tuple(keep_mask.shape)}"
    )


@cache
def _cuda_device_capability(index: int) -> tuple[int, int]:
    """Return the cached CUDA compute capability for one device index.

    :param int index: CUDA device index.
    :return tuple[int, int]: ``(major, minor)`` capability.
    """

    return torch.cuda.get_device_capability(index)


def device_compute_capability(device: torch.device) -> tuple[int, int]:
    """Return CUDA compute capability for one device.

    The capability is constant per device for the process lifetime and this
    helper runs per layer per forward/backward, so it caches per device index.

    :param torch.device device: Device to query.
    :return tuple[int, int]: ``(major, minor)`` capability, or ``(0, 0)`` off CUDA.
    """

    if device.type != "cuda":
        return (0, 0)
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return _cuda_device_capability(int(index))
