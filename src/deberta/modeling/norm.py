"""Normalization layers used by modernized DeBERTa backbones."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm (root-mean-square normalization).

    We implement RMSNorm explicitly to avoid version skew and to keep parameter names stable
    across PyTorch versions (useful for FSDP and weight-decay filtering).
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        """Create RMSNorm layer.

        :param int hidden_size: Final hidden dimension.
        :param float eps: Numerical epsilon.
        :param bool elementwise_affine: Whether to learn multiplicative weight.
        """
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize activations by root-mean-square magnitude.

        :param torch.Tensor x: Input tensor.
        :return torch.Tensor: RMS-normalized tensor.
        """
        # x: (..., hidden)
        # rms = sqrt(mean(x^2))
        dtype = x.dtype
        x_float = x.float() if dtype in (torch.float16, torch.bfloat16) else x
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        y = (x_float / rms).to(dtype)
        if self.weight is not None:
            y = y * self.weight.to(dtype)
        return y
