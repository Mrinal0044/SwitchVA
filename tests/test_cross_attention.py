"""Tests for Aspect-Guided Mutual Cross-Attention and Fused Z in R^(3d)."""

import torch
import pytest
from src.models.cross_attention import AspectGuidedMutualCrossAttention


def test_cross_attention_and_fused_z():
    batch_size = 2
    num_nodes = 14
    dim = 64
    num_heads = 4

    cross_attn = AspectGuidedMutualCrossAttention(
        hidden_dim=dim,
        num_heads=num_heads,
        dropout=0.1,
    )

    aspect_span = torch.randn(batch_size, dim)
    graph_nodes = torch.randn(batch_size, num_nodes, dim)
    graph_aspect = torch.randn(batch_size, dim)

    Z, attn_weights = cross_attn(aspect_span, graph_nodes, graph_aspect)

    # Output Z must strictly be in R^(3d)
    assert Z.shape == (batch_size, 3 * dim)
    assert not torch.isnan(Z).any()
