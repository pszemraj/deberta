from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # (.., d) -> (.., d) with even/odd rotation
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


@dataclass
class RotaryCache:
    cos: torch.Tensor  # (seq, rotary_dim)
    sin: torch.Tensor  # (seq, rotary_dim)


class RotaryEmbedding(nn.Module):
    """Standard RoPE (Rotary Position Embeddings).

    - Uses the "theta" parameterization popularized by GPT-NeoX / RoFormer.
    - Caches cos/sin on the current device+dtype.
    - Extends cache on demand for longer sequence lengths (length generalization).
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        if self.dim % 2 != 0:
            raise ValueError(f"Rotary dim must be even, got {self.dim}")
        self.base = float(base)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._cache: RotaryCache | None = None
        self._cache_device: torch.device | None = None
        self._cache_dtype: torch.dtype | None = None

    def _build_cache(self, seq_len: int, *, device: torch.device, dtype: torch.dtype) -> RotaryCache:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq, dim/2)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)  # (seq, dim)
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return RotaryCache(cos=cos, sin=sin)

    def get_cos_sin(
        self, seq_len: int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        Args:
          q,k: (batch, heads, seq, head_dim)

        Returns:
          rotated q,k (same shapes)
        """
        seq_len = q.shape[-2]
        device = q.device
        dtype = q.dtype
        cos, sin = self.get_cos_sin(seq_len, device=device, dtype=dtype)  # (seq, dim)

        cos = cos[None, None, :, :]  # (1,1,seq,dim)
        sin = sin[None, None, :, :]

        q1, q2 = q[..., : self.dim], q[..., self.dim :]
        k1, k2 = k[..., : self.dim], k[..., self.dim :]

        q1 = (q1 * cos) + (_rotate_half(q1) * sin)
        k1 = (k1 * cos) + (_rotate_half(k1) * sin)

        q_out = torch.cat((q1, q2), dim=-1)
        k_out = torch.cat((k1, k2), dim=-1)
        return q_out, k_out
