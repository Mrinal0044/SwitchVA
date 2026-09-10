"""Tests for Signed Switch-Distance computation and Switch Embedding."""

import torch
import pytest
from src.models.switch_embedding import SwitchEmbedding, compute_signed_switch_distance


def test_signed_switch_distance():
    # Example: HI (0), HI (0), EN (1), EN (1), HI (0)
    # Positions: 0, 1, 2, 3, 4
    # Switch point 1 is at index 2 (0 -> 1), Switch point 2 is at index 4 (1 -> 0)
    lang_ids = torch.tensor([[0, 0, 1, 1, 0]])
    max_dist = 10
    dist_indices = compute_signed_switch_distance(lang_ids, pad_id=3, max_dist=max_dist)

    # Output indices should be clamped in [0, 2 * max_dist]
    assert dist_indices.shape == lang_ids.shape
    assert (dist_indices >= 0).all()
    assert (dist_indices <= 2 * max_dist).all()

    # Index 2 is a switch point, so signed distance = 0 -> index = max_dist + 1 (11)
    assert dist_indices[0, 2].item() == max_dist + 1
    # Index 4 is a switch point -> index = max_dist + 1 (11)
    assert dist_indices[0, 4].item() == max_dist + 1


def test_switch_embedding_forward():
    batch_size, seq_len = 2, 16
    hidden_dim = 128
    num_languages = 4
    max_dist = 32

    module = SwitchEmbedding(
        num_languages=num_languages,
        lang_embed_dim=32,
        max_signed_distance=max_dist,
        dist_embed_dim=32,
        output_dim=hidden_dim,
    )

    lang_ids = torch.randint(0, 3, (batch_size, seq_len))
    embeds = module(lang_ids)

    assert embeds.shape == (batch_size, seq_len, hidden_dim)
    assert not torch.isnan(embeds).any()
