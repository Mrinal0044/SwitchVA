"""Tests for Switch-Point Gated Self-Attention (SP-GSA)."""

import torch
import pytest
from src.models.sp_gsa import SPGSA


def test_sp_gsa_forward():
    batch_size = 2
    seq_len = 12
    hidden_dim = 64
    num_heads = 4

    sp_gsa = SPGSA(hidden_dim=hidden_dim, num_heads=num_heads, dropout=0.1)

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    switch_embeds = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -3:] = 0  # Pad last 3 tokens

    out, G = sp_gsa(hidden_states, switch_embeds, attention_mask=attention_mask)

    assert out.shape == (batch_size, seq_len, hidden_dim)
    assert G.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(out).any()
    assert not torch.isnan(G).any()
