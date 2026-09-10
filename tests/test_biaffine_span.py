"""Tests for Biaffine Aspect-Opinion Span Extractor."""

import torch
import pytest
from src.models.biaffine_span import BiaffineSpanExtractor


def test_biaffine_span_extractor():
    batch_size = 2
    seq_len = 10
    input_dim = 64
    mlp_dim = 32
    num_labels = 3

    extractor = BiaffineSpanExtractor(
        input_dim=input_dim,
        mlp_dim=mlp_dim,
        num_labels=num_labels,
        dropout=0.1,
    )

    hidden_states = torch.randn(batch_size, seq_len, input_dim)
    aspect_spans = torch.tensor([[1, 3], [2, 4]])
    opinion_spans = torch.tensor([[4, 6], [5, 7]])

    outputs = extractor(
        hidden_states,
        aspect_spans=aspect_spans,
        opinion_spans=opinion_spans,
    )

    assert "aspect_scores" in outputs
    assert "opinion_scores" in outputs
    assert "aspect_repr" in outputs
    assert "opinion_repr" in outputs

    assert outputs["aspect_scores"].shape == (batch_size, seq_len, seq_len)
    assert outputs["opinion_scores"].shape == (batch_size, seq_len, seq_len)
    assert outputs["aspect_repr"].shape == (batch_size, input_dim)
    assert outputs["opinion_repr"].shape == (batch_size, input_dim)
    assert not torch.isnan(outputs["aspect_scores"]).any()
