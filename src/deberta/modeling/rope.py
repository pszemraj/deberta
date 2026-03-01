"""Rotary position embedding utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _is_torch_compiling() -> bool:
    """Return whether execution is inside a torch.compile graph.

    :return bool: True when currently executing inside torch.compile.
    """
    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
        return False
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate even/odd channels for RoPE application.

    :param torch.Tensor x: Tensor with final dimension as rotary channels.
    :return torch.Tensor: Rotated tensor.
    """
    # (.., d) -> (.., d) with even/odd rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


@dataclass
class RotaryCache:
    """Cosine/sine cache for a specific sequence length/device/dtype."""

    cos: torch.Tensor  # (seq, rotary_dim)
    sin: torch.Tensor  # (seq, rotary_dim)


class RotaryEmbedding(nn.Module):
    """Standard RoPE (Rotary Position Embeddings).

    - Uses the "theta" parameterization popularized by GPT-NeoX / RoFormer.
    - Caches cos/sin on the current device+dtype.
    - Extends cache on demand for longer sequence lengths (length generalization).
    """

    def __init__(self, dim: int, base: float = 10000.0, full_dim: int | None = None) -> None:
        """Create rotary embedding helper.

        :param int dim: Rotary dimension (must be even).
        :param float base: RoPE base theta.
        :param int | None full_dim: Optional full head-dimension for frequency spacing.
        """
        super().__init__()
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError(f"Rotary dim must be even, got {self.dim}")
        self.base = float(base)
        denom = int(full_dim) if full_dim is not None else self.dim

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / float(denom)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cache: RotaryCache | None = None
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None

    def prefill_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> None:
        """Prefill module-local cache for compile-safe slice-only access.

        :param int seq_len: Maximum sequence length to cache.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Target dtype.
        :raises ValueError: If ``seq_len`` is not positive.
        :return None: This method mutates module cache state.
        """
        if int(seq_len) <= 0:
            raise ValueError(f"seq_len must be > 0 for rotary cache prefill, got {seq_len}.")
        self._cache = self._build_cache(int(seq_len), device=device, dtype=dtype)
        self._cache_device = device
        self._cache_dtype = dtype

    def _build_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> RotaryCache:
        """Build cosine/sine cache tensors.

        :param int seq_len: Sequence length.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Target dtype.
        :return RotaryCache: Cache for the requested length/device/dtype.
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq, dim/2)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # (seq, dim)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return RotaryCache(cos=cos, sin=sin)

    def get_cos_sin(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached cosine/sine tensors for requested sequence length.

        :param int seq_len: Sequence length.
        :param torch.device device: Target device.
        :param torch.dtype dtype: Target dtype.
        :return tuple[torch.Tensor, torch.Tensor]: Cosine and sine tensors.
        """
        # Under compile, this must be a pure slice over an already-prefilled
        # module cache to avoid per-step allocations and storage mutation.
        if _is_torch_compiling():
            if self._cache is None or self._cache_device != device or self._cache_dtype != dtype:
                raise RuntimeError(
                    "Rotary cache is not prefilled for compiled execution. "
                    "Call RotaryEmbedding.prefill_cache(max_seq_len, device, dtype) before torch.compile."
                )
            if self._cache.cos.shape[0] < seq_len:
                raise RuntimeError(
                    "Compiled rotary cache is too short for requested sequence length: "
                    f"cached={int(self._cache.cos.shape[0])}, requested={int(seq_len)}. "
                    "Prefill cache with the maximum runtime sequence length before compile."
                )
            return self._cache.cos[:seq_len], self._cache.sin[:seq_len]

        if (
            self._cache is None
            or self._cache_device != device
            or self._cache_dtype != dtype
            or self._cache.cos.shape[0] < seq_len
        ):
            self._cache = self._build_cache(seq_len, device=device, dtype=dtype)
            self._cache_device = device
            self._cache_dtype = dtype
        cos = self._cache.cos[:seq_len]
        sin = self._cache.sin[:seq_len]
        return cos, sin

    def apply(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to q and k.

        :param torch.Tensor q: Query tensor shaped (batch, heads, seq, head_dim).
        :param torch.Tensor k: Key tensor shaped (batch, heads, seq, head_dim).
        :return tuple[torch.Tensor, torch.Tensor]: Rotated query and key tensors.
        """
        seq_len = q.shape[-2]
        device = q.device
        dtype = q.dtype
        cos, sin = self.get_cos_sin(seq_len, device=device, dtype=dtype)  # (seq, dim)

        cos = cos[None, None, :, :]  # (1,1,seq,dim)
        sin = sin[None, None, :, :]

        if self.dim == q.shape[-1]:
            # Fast path when rotary_pct=1.0: avoid split/cat with empty tails.
            q_out = (q * cos) + (_rotate_half(q) * sin)
            k_out = (k * cos) + (_rotate_half(k) * sin)
            return q_out, k_out

        q1, q2 = q[..., : self.dim], q[..., self.dim :]
        k1, k2 = k[..., : self.dim], k[..., self.dim :]

        q1 = (q1 * cos) + (_rotate_half(q1) * sin)
        k1 = (k1 * cos) + (_rotate_half(k1) * sin)

        q_out = torch.cat((q1, q2), dim=-1)
        k_out = torch.cat((k1, k2), dim=-1)
        return q_out, k_out
