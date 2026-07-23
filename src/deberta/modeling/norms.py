"""Normalization layers shared across model families."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedPrecisionRMSNorm(nn.RMSNorm):
    """Run native RMSNorm with dtype-aligned operands and FP32 master weights."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Normalize activations through PyTorch's fused same-dtype path.

        :param torch.Tensor input: Input activations.
        :return torch.Tensor: RMS-normalized activations.
        """
        weight = self.weight
        if weight is not None and weight.dtype != input.dtype:
            weight = weight.to(dtype=input.dtype)
        return F.rms_norm(input, self.normalized_shape, weight, self.eps)
