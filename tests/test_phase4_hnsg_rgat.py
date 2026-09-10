"""Tests for Phase 4: Heterogeneous Neuro-Symbolic Graph (H-NSG) & Relational Graph Attention Network (RGAT).

This suite tests:
1. Token graph node creation
2. Bidirectional token <-> graph node alignment
3. Syntactic relation adjacency (REL_SYNTACTIC = 0)
4. Code-switch bridge relation adjacency (REL_SWITCH = 1)
5. Aspect-opinion relation adjacency (REL_ASPECT_OPINION = 2)
6. Non-collapsed 3-relation adjacency tensor (B, 3, N, N)
7. NRC-VAD lexicon normalized lookup in [0, 1]
8. NRC-VAD OOV handling and zero-masking
9. NRC-VAD symbolic prior injection
10. Target leakage prevention (no ground truth VA in graph inputs)
11. RGAT relation-specific projection weights W_r
12. Multi-head relational attention normalization
13. RGAT LayerNorm and residual connections
14. Multi-layer RGAT stack forward pass
15. Batch independence and sentence isolation
16. Graph span representation pooling (mean mode)
17. Graph span representation pooling (max & endpoints modes)
18. End-to-end gradient flow through RGAT and prior projections
19. Numerical stability (no NaN/Inf on isolated nodes or padded inputs)
20. End-to-end Branch 2 pipeline execution
"""

import pytest
import torch
import torch.nn as nn

from src.data.nrc_vad import NRCVADLexicon
from src.data.dependency_parser import DependencyParserInterface, HeuristicDependencyBuilder
from src.models.hnsg import (
    HeterogeneousNeuroSymbolicGraph,
    REL_SYNTACTIC,
    REL_SWITCH,
    REL_ASPECT_OPINION,
    NUM_RELATIONS,
)
from src.models.rgat import RGAT, RGATLayer, extract_graph_span_representation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_lexicon():
    entries = {
        "good": (0.85, 0.55),
        "bad": (0.15, 0.65),
        "accha": (0.80, 0.50),
        "kharab": (0.20, 0.60),
        "battery": (0.50, 0.40),
        "camera": (0.55, 0.45),
        "phone": (0.50, 0.50),
    }
    return NRCVADLexicon(custom_lexicon=entries)


@pytest.fixture
def sample_tokens_and_langs():
    # Sentence: "Yeh phone ka camera bohot accha hai"
    # Langs: [HI, HI, HI, EN, HI, HI, HI] -> [0, 0, 0, 1, 0, 0, 0]
    tokens = ["yeh", "phone", "ka", "camera", "bohot", "accha", "hai"]
    lang_ids = [0, 0, 0, 1, 0, 0, 0]
    aspect_spans = [(3, 3)]  # "camera"
    opinion_spans = [(5, 5)]  # "accha"
    aspect_opinion_pairs = [((3, 3), (5, 5))]
    return tokens, lang_ids, aspect_spans, opinion_spans, aspect_opinion_pairs


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_1_graph_node_creation_token_nodes(mock_lexicon, sample_tokens_and_langs):
    """1. Test that token nodes are correctly indexed for input sequences."""
    tokens, lang_ids, _, _, _ = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    vad_priors, vad_mask = hnsg.extract_lexicon_priors([tokens])
    assert vad_priors.shape == (1, 7, 2)
    assert vad_mask.shape == (1, 7)
    assert vad_mask[0, 1].item() is True   # "phone" is in lexicon
    assert vad_mask[0, 3].item() is True   # "camera" is in lexicon
    assert vad_mask[0, 5].item() is True   # "accha" is in lexicon


def test_2_token_to_graph_node_bidirectional_alignment(mock_lexicon, sample_tokens_and_langs):
    """2. Test bidirectional token <-> graph node alignment."""
    tokens, _, _, _, _ = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    adj, node_info = hnsg.build_graph(
        tokens_batch=[tokens],
        max_seq_len=10,
        return_node_info=True,
    )
    mapping = node_info[0]["token_to_node"]
    assert len(mapping) == 7
    for i in range(7):
        assert mapping[i] == i
        assert node_info[0]["node_to_token"][mapping[i]] == i


def test_3_relation_rel_syntactic_adjacency(mock_lexicon, sample_tokens_and_langs):
    """3. Test that REL_SYNTACTIC edges (rel=0) contain syntax/sequential dependency edges."""
    tokens, _, _, _, _ = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    adj = hnsg.build_graph(tokens_batch=[tokens], max_seq_len=7)
    # Self-loops and linear/syntactic adjacency in rel 0
    syn_adj = adj[0, REL_SYNTACTIC]
    assert syn_adj[0, 0] == 1.0  # self-loop
    assert syn_adj[0, 1] == 1.0  # sequential syntax edge
    assert syn_adj[1, 2] == 1.0


def test_4_relation_rel_switch_adjacency(mock_lexicon, sample_tokens_and_langs):
    """4. Test that REL_SWITCH edges (rel=1) connect switch-point flanking tokens."""
    tokens, lang_ids, _, _, _ = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    # Switch points: ka(2, HI) -> camera(3, EN) and camera(3, EN) -> bohot(4, HI)
    adj = hnsg.build_graph(
        tokens_batch=[tokens],
        lang_ids_batch=[lang_ids],
        max_seq_len=7,
    )
    sw_adj = adj[0, REL_SWITCH]
    # Edge between token 2 and token 3
    assert sw_adj[2, 3] == 1.0
    assert sw_adj[3, 2] == 1.0
    # Edge between token 3 and token 4
    assert sw_adj[3, 4] == 1.0
    assert sw_adj[4, 3] == 1.0
    # No switch edge between token 0 and token 1 (both HI)
    assert sw_adj[0, 1] == 0.0


def test_5_relation_rel_aspect_opinion_adjacency(mock_lexicon, sample_tokens_and_langs):
    """5. Test that REL_ASPECT_OPINION edges (rel=2) connect aspect and opinion tokens."""
    tokens, lang_ids, _, _, aspect_opinion_pairs = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    adj = hnsg.build_graph(
        tokens_batch=[tokens],
        lang_ids_batch=[lang_ids],
        aspect_opinion_pairs_batch=[aspect_opinion_pairs],
        max_seq_len=7,
    )
    ao_adj = adj[0, REL_ASPECT_OPINION]
    # Aspect is (3, 3) "camera", Opinion is (5, 5) "accha"
    assert ao_adj[3, 5] == 1.0
    assert ao_adj[5, 3] == 1.0
    # Other pairs have no relation 2 edge
    assert ao_adj[0, 1] == 0.0
    assert ao_adj[2, 3] == 0.0


def test_6_no_collapsed_adjacency_matrix(mock_lexicon, sample_tokens_and_langs):
    """6. Verify adjacency is strictly (B, 3, N, N) and relation channels do not collapse."""
    tokens, lang_ids, _, _, pairs = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=64, lexicon=mock_lexicon)
    
    adj = hnsg.build_graph(
        tokens_batch=[tokens],
        lang_ids_batch=[lang_ids],
        aspect_opinion_pairs_batch=[pairs],
        max_seq_len=8,
    )
    assert adj.shape == (1, 3, 8, 8)
    assert not torch.equal(adj[0, 0], adj[0, 1])
    assert not torch.equal(adj[0, 1], adj[0, 2])


def test_7_nrc_vad_lexicon_lookup(mock_lexicon):
    """7. Test normalized [0, 1] valence and arousal lookup."""
    v, a, mask = mock_lexicon.get_vad("good")
    assert mask is True
    assert 0.0 <= v <= 1.0
    assert 0.0 <= a <= 1.0
    assert v == 0.85
    assert a == 0.55


def test_8_nrc_vad_oov_handling(mock_lexicon):
    """8. Test out-of-vocabulary handling in NRC-VAD lexicon."""
    v, a, mask = mock_lexicon.get_vad("nonexistentword12345")
    assert mask is False
    assert v == 0.0
    assert a == 0.0


def test_9_nrc_vad_feature_injection(mock_lexicon, sample_tokens_and_langs):
    """9. Test symbolic prior projection addition into H_gated."""
    tokens, _, _, _, _ = sample_tokens_and_langs
    d_model = 32
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=d_model, lexicon=mock_lexicon)
    
    H_gated = torch.randn(1, 7, d_model)
    H_init, vad_priors, vad_mask = hnsg(H_gated, [tokens])
    
    assert H_init.shape == (1, 7, d_model)
    # For OOV tokens (e.g. token 0 "yeh"), H_init equals H_gated
    # For known tokens (e.g. token 1 "phone"), H_init differs from H_gated
    assert torch.allclose(H_init[0, 0], H_gated[0, 0], atol=1e-6)
    assert not torch.allclose(H_init[0, 1], H_gated[0, 1], atol=1e-6)


def test_10_no_ground_truth_va_leakage(mock_lexicon, sample_tokens_and_langs):
    """10. Verify that H-NSG and RGAT modules have no parameters or inputs accepting ground-truth VA."""
    tokens, lang_ids, _, _, pairs = sample_tokens_and_langs
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=32, lexicon=mock_lexicon)
    rgat = RGAT(d_model=32, num_relations=3, num_layers=2)
    
    # Check signature of forward methods
    import inspect
    hnsg_args = inspect.signature(hnsg.forward).parameters
    rgat_args = inspect.signature(rgat.forward).parameters
    
    assert "ground_truth" not in hnsg_args
    assert "target_va" not in hnsg_args
    assert "ground_truth" not in rgat_args
    assert "target_va" not in rgat_args


def test_11_rgat_layer_relation_specific_projections():
    """11. Test RGATLayer has separate projection weights for each of the 3 relations."""
    d_model = 32
    layer = RGATLayer(d_model=d_model, num_relations=3, num_heads=4)
    assert len(layer.W_r) == 3
    # Verify weights are independent tensors
    assert not torch.allclose(layer.W_r[0].weight, layer.W_r[1].weight)
    assert not torch.allclose(layer.W_r[1].weight, layer.W_r[2].weight)


def test_12_rgat_multihead_attention():
    """12. Test multi-head relational attention forward pass and shape contract."""
    d_model = 64
    num_heads = 4
    layer = RGATLayer(d_model=d_model, num_relations=3, num_heads=num_heads)
    
    x = torch.randn(2, 6, d_model)
    adj = torch.zeros(2, 3, 6, 6)
    # Give self-loops
    adj[:, 0] = torch.eye(6).unsqueeze(0).expand(2, -1, -1)
    
    out = layer(x, adj)
    assert out.shape == (2, 6, d_model)


def test_13_rgat_layer_residual_layernorm():
    """13. Test LayerNorm and residual connection in RGATLayer."""
    d_model = 32
    layer = RGATLayer(d_model=d_model, num_relations=3, num_heads=2)
    
    assert hasattr(layer, "layer_norm")
    x = torch.randn(2, 5, d_model)
    adj = torch.zeros(2, 3, 5, 5)
    adj[:, 0] = torch.eye(5).unsqueeze(0).expand(2, -1, -1)
    
    out = layer(x, adj)
    assert out.shape == x.shape


def test_14_rgat_stack_multi_layer():
    """14. Test multi-layer RGAT stack forward pass produces expected shape."""
    d_model = 64
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=3, num_heads=4)
    
    x = torch.randn(2, 8, d_model)
    adj = torch.zeros(2, 3, 8, 8)
    adj[:, 0] = torch.eye(8).unsqueeze(0).expand(2, -1, -1)
    mask = torch.ones(2, 8)
    
    out = rgat(x, adj, attention_mask=mask)
    assert out.shape == (2, 8, d_model)


def test_15_rgat_batch_independence():
    """15. Test that modifying item b1 does not affect output of item b0."""
    d_model = 32
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=2, num_heads=2)
    rgat.eval()
    
    x1 = torch.randn(1, 5, d_model)
    x2 = torch.randn(1, 5, d_model)
    adj1 = torch.zeros(1, 3, 5, 5)
    adj1[:, 0] = torch.eye(5).unsqueeze(0)
    adj2 = torch.zeros(1, 3, 5, 5)
    adj2[:, 0] = torch.eye(5).unsqueeze(0)
    
    batch_x = torch.cat([x1, x2], dim=0)
    batch_adj = torch.cat([adj1, adj2], dim=0)
    
    with torch.no_grad():
        out_batch = rgat(batch_x, batch_adj)
        out_single = rgat(x1, adj1)
        
    assert torch.allclose(out_batch[0], out_single[0], atol=1e-5)


def test_16_graph_span_pooling_mean():
    """16. Test graph span representation extraction using mean pooling."""
    d_model = 16
    H_graph = torch.tensor([
        [[1.0]*d_model, [3.0]*d_model, [5.0]*d_model, [7.0]*d_model]
    ])  # shape (1, 4, 16)
    
    spans = torch.tensor([[[1, 2]]])  # tokens 1 and 2: mean of 3.0 and 5.0 -> 4.0
    pooled = extract_graph_span_representation(H_graph, spans, pool_mode="mean")
    
    assert pooled.shape == (1, 1, d_model)
    assert torch.allclose(pooled[0, 0], torch.tensor([4.0]*d_model))


def test_17_graph_span_pooling_max_and_endpoints():
    """17. Test graph span representation extraction using max and endpoints pooling."""
    d_model = 16
    H_graph = torch.tensor([
        [[1.0]*d_model, [3.0]*d_model, [5.0]*d_model, [7.0]*d_model]
    ])
    spans = torch.tensor([[[1, 2]]])
    
    # Max pooling
    pooled_max = extract_graph_span_representation(H_graph, spans, pool_mode="max")
    assert pooled_max.shape == (1, 1, d_model)
    assert torch.allclose(pooled_max[0, 0], torch.tensor([5.0]*d_model))
    
    # Endpoints pooling
    pooled_endpoints = extract_graph_span_representation(H_graph, spans, pool_mode="endpoints")
    assert pooled_endpoints.shape == (1, 1, 2 * d_model)


def test_18_gradient_flow_through_rgat(mock_lexicon, sample_tokens_and_langs):
    """18. Test that backward pass computes valid non-zero gradients for RGAT and prior projections."""
    tokens, lang_ids, _, _, pairs = sample_tokens_and_langs
    d_model = 32
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=d_model, lexicon=mock_lexicon)
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=2, num_heads=2)
    
    H_gated = torch.randn(1, 7, d_model, requires_grad=True)
    adj = hnsg.build_graph([tokens], [lang_ids], [pairs], max_seq_len=7)
    
    H_init, _, _ = hnsg(H_gated, [tokens])
    H_graph = rgat(H_init, adj)
    
    loss = H_graph.sum()
    loss.backward()
    
    assert H_gated.grad is not None
    assert torch.norm(H_gated.grad.double()) > 0.0
    assert hnsg.prior_proj[0].weight.grad is not None
    assert torch.norm(hnsg.prior_proj[0].weight.grad.double()) > 0.0


def test_19_no_nan_or_inf_in_rgat_forward_backward():
    """19. Test numerical stability with empty adjacency and padding masks."""
    d_model = 32
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=2, num_heads=2)
    
    x = torch.randn(2, 6, d_model, requires_grad=True)
    # Empty adjacency (isolated nodes)
    adj = torch.zeros(2, 3, 6, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.float32)
    
    out = rgat(x, adj, attention_mask=mask)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    loss = out.sum()
    loss.backward()
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


def test_20_end_to_end_branch2_pipeline(mock_lexicon, sample_tokens_and_langs):
    """20. Test full Branch 2 end-to-end forward pass producing H_graph, G_aspect, G_opinion."""
    tokens, lang_ids, aspect_spans, opinion_spans, pairs = sample_tokens_and_langs
    d_model = 64
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=d_model, lexicon=mock_lexicon)
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=2, num_heads=4)
    
    H_gated = torch.randn(1, 7, d_model)
    adj = hnsg.build_graph([tokens], [lang_ids], [pairs], max_seq_len=7)
    
    H_init, vad_priors, vad_mask = hnsg(H_gated, [tokens])
    H_graph = rgat(H_init, adj)
    
    asp_tensor = torch.tensor([[aspect_spans[0]]])  # (1, 1, 2)
    op_tensor = torch.tensor([[opinion_spans[0]]])   # (1, 1, 2)
    
    G_aspect = extract_graph_span_representation(H_graph, asp_tensor, pool_mode="mean")
    G_opinion = extract_graph_span_representation(H_graph, op_tensor, pool_mode="mean")
    
    assert H_graph.shape == (1, 7, d_model)
    assert G_aspect.shape == (1, 1, d_model)
    assert G_opinion.shape == (1, 1, d_model)
    assert not torch.isnan(G_aspect).any()
    assert not torch.isnan(G_opinion).any()
