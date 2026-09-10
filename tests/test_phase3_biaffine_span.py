"""Unit tests for Phase 3: Biaffine Aspect-Opinion Span Extractor (Branch 1)."""

import pytest
import torch

from src.models.biaffine_span import (
    BiaffineScorer,
    BiaffineAspectOpinionSpanExtractor,
)
from src.losses.span_loss import BiaffineSpanLoss, construct_gold_span_grid
from src.data.tokenizer_alignment import SubwordAligner, map_gold_spans_to_subwords


# TEST 1: Correct input dimensions
def test_input_dimensions():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=64, span_dim=32)
    H_gated = torch.randn(2, 12, 64)
    attention_mask = torch.ones(2, 12)

    outputs = extractor(H_gated, attention_mask=attention_mask)
    assert outputs["aspect_scores"].shape == (2, 12, 12)
    assert outputs["opinion_scores"].shape == (2, 12, 12)


# TEST 2: Aspect projection MLPs
def test_aspect_projections():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=64, span_dim=32)
    H_gated = torch.randn(2, 8, 64)
    s = extractor.aspect_start_mlp(H_gated)
    e = extractor.aspect_end_mlp(H_gated)
    assert s.shape == (2, 8, 32)
    assert e.shape == (2, 8, 32)


# TEST 3: Opinion projection MLPs
def test_opinion_projections():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=64, span_dim=32)
    H_gated = torch.randn(2, 8, 64)
    s = extractor.opinion_start_mlp(H_gated)
    e = extractor.opinion_end_mlp(H_gated)
    assert s.shape == (2, 8, 32)
    assert e.shape == (2, 8, 32)


# TEST 4: Biaffine score grid dimensions (B, N, N)
def test_biaffine_score_dimensions():
    scorer = BiaffineScorer(span_dim=32)
    start_repr = torch.randn(3, 10, 32)
    end_repr = torch.randn(3, 10, 32)
    scores = scorer(start_repr, end_repr)
    assert scores.shape == (3, 10, 10)


# TEST 5: Start <= End masking
def test_start_le_end_masking():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16, max_span_length=10)
    mask = extractor.compute_valid_span_mask(batch_size=1, seq_len=5)
    # When start > end (i > j), mask must be False
    assert not mask[0, 3, 1].item()
    assert not mask[0, 4, 0].item()
    # When start <= end (i <= j), mask is True
    assert mask[0, 1, 3].item()
    assert mask[0, 2, 2].item()


# TEST 6: Padding masking
def test_padding_masking():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])  # Tokens 3, 4 are padding
    mask = extractor.compute_valid_span_mask(batch_size=1, seq_len=5, attention_mask=attention_mask)

    # Valid non-pad candidate
    assert mask[0, 0, 2].item()
    # Spans involving padding must be False
    assert not mask[0, 0, 3].item()
    assert not mask[0, 3, 4].item()


# TEST 7: Maximum span length constraint
def test_max_span_length():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16, max_span_length=3)
    mask = extractor.compute_valid_span_mask(batch_size=1, seq_len=8)
    # Span of length 3: indices [1, 3] -> 3 - 1 + 1 = 3 (Valid)
    assert mask[0, 1, 3].item()
    # Span of length 4: indices [1, 4] -> 4 - 1 + 1 = 4 (Invalid > max 3)
    assert not mask[0, 1, 4].item()


# TEST 8: Exact gold span mapping
def test_exact_gold_span_mapping():
    word_to_subword = [[1], [2, 3], [4]]  # 3 words
    s, e, ok = map_gold_spans_to_subwords(0, 1, word_to_subword, is_training_safe=True)
    assert ok is True
    assert s == 1
    assert e == 3


# TEST 9: Normalized gold span mapping
def test_normalized_gold_span_mapping():
    word_to_subword = [[1, 2], [3], [4, 5]]
    s, e, ok = map_gold_spans_to_subwords(1, 2, word_to_subword, is_training_safe=True)
    assert ok is True
    assert s == 3
    assert e == 5


# TEST 10: Approximate span masking
def test_approximate_span_masking():
    word_to_subword = [[1], [2], [3]]
    # is_training_safe=False for approximate spans
    s, e, ok = map_gold_spans_to_subwords(0, 1, word_to_subword, is_training_safe=False)
    assert ok is False
    assert s == -1
    assert e == -1


# TEST 11: Unmatched span masking
def test_unmatched_span_masking():
    word_to_subword = [[1], [2], [3]]
    # Unmatched span has word_start=-1
    s, e, ok = map_gold_spans_to_subwords(-1, -1, word_to_subword, is_training_safe=False)
    assert ok is False


# TEST 12: Multiple gold spans per sentence
def test_multiple_gold_spans():
    batch_size = 1
    seq_len = 8
    valid_mask = torch.ones((batch_size, seq_len, seq_len), dtype=torch.bool)
    gold_spans = [[
        {"start": 1, "end": 2, "is_training_safe": True},
        {"start": 4, "end": 6, "is_training_safe": True},
    ]]

    gold_grid, sup_mask = construct_gold_span_grid(batch_size, seq_len, gold_spans, valid_mask)
    assert gold_grid[0, 1, 2].item() == 1.0
    assert gold_grid[0, 4, 6].item() == 1.0
    assert gold_grid[0, 0, 0].item() == 0.0
    assert gold_grid.sum().item() == 2.0


# TEST 13: Word-to-subword span mapping edge cases
def test_word_to_subword_edge_cases():
    word_to_subword = [[1], [2]]
    # Out of bounds
    assert not map_gold_spans_to_subwords(0, 5, word_to_subword, is_training_safe=True)[2]
    # Inverted indices
    assert not map_gold_spans_to_subwords(2, 0, word_to_subword, is_training_safe=True)[2]


# TEST 14: Decoded span validity (start <= end)
def test_decoded_span_validity():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16)
    H_gated = torch.randn(2, 6, 32)
    outputs = extractor(H_gated, threshold=-10.0, top_k=5)

    for b in range(2):
        for span in outputs["decoded_aspect_spans"][b]:
            assert span["start"] <= span["end"]
            assert span["start"] >= 0 and span["end"] < 6
        for span in outputs["decoded_opinion_spans"][b]:
            assert span["start"] <= span["end"]
            assert span["start"] >= 0 and span["end"] < 6


# TEST 15: Span representation dimensions
def test_span_representation_dimensions():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=64, span_dim=32)
    H_gated = torch.randn(3, 8, 64)
    aspect_spans = torch.tensor([[1, 2], [3, 4], [0, 5]])

    repr_vec = extractor.extract_span_representation(H_gated, spans=aspect_spans)
    assert repr_vec.shape == (3, 64)
    assert not torch.isnan(repr_vec).any()


# TEST 16: Gradient propagation through span extractor and loss
def test_gradient_propagation_span():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16)
    loss_fn = BiaffineSpanLoss()

    H_gated = torch.randn(2, 6, 32, requires_grad=True)
    attention_mask = torch.ones(2, 6)

    outputs = extractor(H_gated, attention_mask=attention_mask)
    gold_aspects = [[{"start": 1, "end": 2, "is_training_safe": True}], []]
    gold_opinions = [[{"start": 3, "end": 4, "is_training_safe": True}], [{"start": 1, "end": 1, "is_training_safe": True}]]

    loss_dict = loss_fn(
        outputs["aspect_scores"],
        outputs["opinion_scores"],
        outputs["valid_span_mask"],
        gold_aspects,
        gold_opinions,
    )

    total_loss = loss_dict["total_loss"]
    total_loss.backward()

    assert H_gated.grad is not None
    assert H_gated.grad.abs().sum() > 0
    assert extractor.aspect_scorer.U.grad is not None


# TEST 17: No NaN/Inf detection in span outputs
def test_no_nan_inf_span():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16)
    H_gated = torch.randn(2, 8, 32)
    outputs = extractor(H_gated)

    for k in ["aspect_scores", "opinion_scores", "aspect_repr", "opinion_repr"]:
        t = outputs[k]
        assert not torch.isnan(t).any(), f"NaN in {k}"
        assert not torch.isinf(t).any(), f"Inf in {k}"


# TEST 18: Batch processing with variable sequence lengths
def test_batch_variable_lengths():
    extractor = BiaffineAspectOpinionSpanExtractor(input_dim=32, span_dim=16)
    H_gated = torch.randn(2, 8, 32)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]])

    outputs = extractor(H_gated, attention_mask=attention_mask)
    # Mask for padded positions in second sequence should be False
    valid_mask = outputs["valid_span_mask"]
    assert valid_mask[1, 0, 3].item() is True
    assert valid_mask[1, 0, 4].item() is False
    assert valid_mask[1, 4, 5].item() is False
