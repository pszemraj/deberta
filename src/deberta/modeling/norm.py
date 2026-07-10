"""Normalization layers used by modernized DeBERTa backbones."""

from __future__ import annotations

import torch.nn as nn


class RMSNorm(nn.RMSNorm):
    """PyTorch RMSNorm with the repository's ``hidden_size`` argument name."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
        """Create RMSNorm layer.

        :param int hidden_size: Final hidden dimension.
        :param float eps: Numerical epsilon.
        :param bool elementwise_affine: Whether to learn multiplicative weight.
        """
        super().__init__(int(hidden_size), eps=float(eps), elementwise_affine=elementwise_affine)
