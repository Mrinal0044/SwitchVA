"""Tests for Independent Valence and Arousal Regression Heads."""

import torch
import pytest
from src.models.regression import DualRegressionHeads


def test_dual_regression_heads():
    batch_size = 4
    dim = 64
    fused_dim = 3 * dim  # 192

    heads = DualRegressionHeads(
        input_dim=fused_dim,
        hidden_dim=32,
        dropout=0.1,
    )

    z = torch.randn(batch_size, fused_dim)
    val, aro = heads(z)

    assert val.shape == (batch_size, 1)
    assert aro.shape == (batch_size, 1)

    # Sigmoid guarantees output strictly in [0, 1]
    assert (val >= 0.0).all() and (val <= 1.0).all()
    assert (aro >= 0.0).all() and (aro <= 1.0).all()
