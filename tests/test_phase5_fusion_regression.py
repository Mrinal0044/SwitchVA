"""Tests for Phase 5: Aspect-Guided Mutual Cross-Attention, Fused Representation Z in R^(3d), & Continuous VA Regression.

Test Suite Coverage:
1. Cross-attention forward pass
2. Mutual attention in both directions (Span -> Graph, Graph -> Span)
3. Aspect conditioning and guidance
4. Fused representation exact 3d dimension check
5. Valence prediction range in [0, 1]
6. Arousal prediction range in [0, 1]
7. Independent VA regression heads
8. Multi-aspect sentence support (Z_k -> V_k, A_k)
9. Strict target-leakage prevention
10. CCC loss numerical accuracy
11. Smooth L1 calculation
12. Joint VA loss calculation
13. Small batch / zero-variance CCC stability
14. End-to-end gradient propagation
15. Numerical stability (no NaN/Inf)
16. Batch independence and sentence isolation
17. Phase 3 & Phase 4 output compatibility
18. Complete NSSG-DimNet end-to-end forward pass with structured output formatting
"""

import pytest
import torch
import torch.nn as nn

from src.models.cross_attention import AspectGuidedMutualCrossAttention
from src.models.fusion import AspectGuidedFusion
from src.models.regression import ValenceRegressionHead, ArousalRegressionHead, DualRegressionHeads
from src.models.va_regression import DimensionalVARegression
from src.losses.va_loss import CCCLoss, JointVALoss
from src.models.nssg_dimnet import NSSGDimNet


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_1_cross_attention_forward_pass():
    """1. Test basic forward pass and output shape of AspectGuidedMutualCrossAttention."""
    B, d = 2, 64
    num_heads = 4
    cross_attn = AspectGuidedMutualCrossAttention(hidden_dim=d, num_heads=num_heads)

    asp_repr = torch.randn(B, d)
    op_repr = torch.randn(B, d)
    graph_asp = torch.randn(B, d)
    graph_op = torch.randn(B, d)

    z_cross, attn_dict = cross_attn(
        aspect_span_repr=asp_repr,
        opinion_span_repr=op_repr,
        graph_aspect_repr=graph_asp,
        graph_opinion_repr=graph_op,
    )

    assert z_cross.shape == (B, d)
    assert not torch.isnan(z_cross).any()


def test_2_mutual_attention_both_directions():
    """2. Test that cross-attention produces attention maps for both directions."""
    B, d = 2, 32
    cross_attn = AspectGuidedMutualCrossAttention(hidden_dim=d, num_heads=2)

    asp_repr = torch.randn(B, d)
    op_repr = torch.randn(B, d)
    graph_nodes = torch.randn(B, 6, d)

    z_cross, attn_dict = cross_attn(
        aspect_span_repr=asp_repr,
        opinion_span_repr=op_repr,
        graph_node_features=graph_nodes,
    )

    assert "span_to_graph" in attn_dict
    assert "graph_to_span" in attn_dict
    assert attn_dict["span_to_graph"].shape[1] == 2  # num_heads


def test_3_aspect_representation_conditioning():
    """3. Verify that changing the aspect vector modifies the cross-attention output."""
    B, d = 1, 32
    cross_attn = AspectGuidedMutualCrossAttention(hidden_dim=d, num_heads=2)
    cross_attn.eval()

    asp_1 = torch.randn(B, d)
    asp_2 = asp_1 + 2.0  # Modified aspect vector
    op = torch.randn(B, d)
    g_asp = torch.randn(B, d)
    g_op = torch.randn(B, d)

    with torch.no_grad():
        z_cross_1, _ = cross_attn(asp_1, op, graph_aspect_repr=g_asp, graph_opinion_repr=g_op)
        z_cross_2, _ = cross_attn(asp_2, op, graph_aspect_repr=g_asp, graph_opinion_repr=g_op)

    assert not torch.allclose(z_cross_1, z_cross_2, atol=1e-4)


def test_4_fused_representation_dimension_is_exact_3d():
    """4. Test that AspectGuidedFusion produces exact 3d dimension and verifies assertion."""
    B, d = 2, 64
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=4)

    asp = torch.randn(B, d)
    op = torch.randn(B, d)
    g_asp = torch.randn(B, d)
    g_op = torch.randn(B, d)

    out = fusion(
        aspect_span_repr=asp,
        opinion_span_repr=op,
        graph_aspect_repr=g_asp,
        graph_opinion_repr=g_op,
    )

    Z = out["z"]
    assert Z.shape == (B, 3 * d)
    assert out["z_span"].shape == (B, d)
    assert out["z_graph"].shape == (B, d)
    assert out["z_cross"].shape == (B, d)


def test_5_valence_output_range_zero_to_one():
    """5. Test that Valence regression outputs are strictly bounded in [0, 1]."""
    B, d_in = 4, 192  # 3 * 64
    v_head = ValenceRegressionHead(input_dim=d_in, hidden_dim=64)

    # Test on diverse/extreme input values
    z = torch.randn(B, d_in) * 10.0
    v_pred = v_head(z)

    assert v_pred.shape == (B, 1)
    assert (v_pred >= 0.0).all()
    assert (v_pred <= 1.0).all()


def test_6_arousal_output_range_zero_to_one():
    """6. Test that Arousal regression outputs are strictly bounded in [0, 1]."""
    B, d_in = 4, 192
    a_head = ArousalRegressionHead(input_dim=d_in, hidden_dim=64)

    z = torch.randn(B, d_in) * 10.0
    a_pred = a_head(z)

    assert a_pred.shape == (B, 1)
    assert (a_pred >= 0.0).all()
    assert (a_pred <= 1.0).all()


def test_7_independent_va_regression_heads():
    """7. Test that Valence and Arousal heads have distinct, unshared parameters."""
    d_in = 96
    dual_heads = DualRegressionHeads(input_dim=d_in, hidden_dim=32)

    v_params = list(dual_heads.valence_head.parameters())
    a_params = list(dual_heads.arousal_head.parameters())

    assert len(v_params) == len(a_params)
    for p1, p2 in zip(v_params, a_params):
        assert p1.data_ptr() != p2.data_ptr()
        assert not torch.equal(p1, p2)


def test_8_multiple_aspect_predictions_per_sentence():
    """8. Test multi-aspect sentence support where B=1 has K=3 distinct aspect-opinion pairs."""
    B, K, d = 1, 3, 32
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=2)
    va_reg = DimensionalVARegression(input_dim=3 * d, hidden_dim=32)

    asp_spans = torch.randn(B, K, d)
    op_spans = torch.randn(B, K, d)
    g_asp = torch.randn(B, K, d)
    g_op = torch.randn(B, K, d)

    fused_out = fusion(
        aspect_span_repr=asp_spans,
        opinion_span_repr=op_spans,
        graph_aspect_repr=g_asp,
        graph_opinion_repr=g_op,
    )
    Z = fused_out["z"]
    assert Z.shape == (B, K, 3 * d)

    va_out = va_reg(Z)
    valence = va_out["valence"]
    arousal = va_out["arousal"]

    assert valence.shape == (B, K, 1)
    assert arousal.shape == (B, K, 1)
    assert (valence >= 0.0).all() and (valence <= 1.0).all()


def test_9_target_leakage_prevention():
    """9. Strict check that model forward and fusion signatures have no target VA arguments."""
    import inspect
    fusion_args = inspect.signature(AspectGuidedFusion.forward).parameters
    va_reg_args = inspect.signature(DimensionalVARegression.forward).parameters
    model_args = inspect.signature(NSSGDimNet.forward).parameters

    for forbidden in ["target_va", "target_valence", "target_arousal", "gold_valence", "gold_arousal"]:
        assert forbidden not in fusion_args
        assert forbidden not in va_reg_args
        assert forbidden not in model_args


def test_10_ccc_calculation_numerical_accuracy():
    """10. Test CCC loss calculation against known ground truth correlation."""
    ccc_fn = CCCLoss()

    # Identical predictions and targets -> CCC = 1.0 -> Loss = 0.0
    pred = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    target = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    loss_id = ccc_fn(pred, target)
    assert abs(loss_id.item() - 0.0) < 1e-4

    # Inverse correlation -> CCC = -1.0 -> Loss = 2.0
    target_inv = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
    loss_inv = ccc_fn(pred, target_inv)
    assert abs(loss_inv.item() - 2.0) < 1e-4


def test_11_smooth_l1_calculation():
    """11. Test Smooth L1 loss in JointVALoss."""
    loss_module = JointVALoss(ccc_weight_v=0.0, ccc_weight_a=0.0, smooth_l1_weight_v=1.0, smooth_l1_weight_a=1.0)
    p_v = torch.tensor([0.5, 0.6])
    t_v = torch.tensor([0.5, 0.6])
    p_a = torch.tensor([0.4, 0.8])
    t_a = torch.tensor([0.4, 0.8])

    out = loss_module(p_v, p_a, t_v, t_a)
    assert out["total_loss"].item() < 1e-5


def test_12_joint_va_loss_calculation():
    """12. Test weighted sum in JointVALoss."""
    loss_module = JointVALoss(
        ccc_weight_v=1.0,
        ccc_weight_a=1.0,
        smooth_l1_weight_v=0.5,
        smooth_l1_weight_a=0.5,
    )
    p_v = torch.tensor([0.2, 0.8])
    p_a = torch.tensor([0.3, 0.7])
    t_v = torch.tensor([0.4, 0.6])
    t_a = torch.tensor([0.5, 0.5])

    res = loss_module(p_v, p_a, t_v, t_a)
    assert "total_loss" in res
    assert "loss_ccc_v" in res
    assert "loss_ccc_a" in res
    assert "loss_smooth_l1_v" in res
    assert "loss_smooth_l1_a" in res
    assert res["total_loss"].item() > 0.0


def test_13_zero_variance_and_single_sample_ccc_stability():
    """13. Test CCC numerical stability with batch size 1 and zero-variance inputs."""
    ccc_fn = CCCLoss()

    # Batch size 1
    p_single = torch.tensor([0.6])
    t_single = torch.tensor([0.4])
    loss_single = ccc_fn(p_single, t_single)
    assert not torch.isnan(loss_single)

    # Zero variance (constant predictions)
    p_const = torch.tensor([0.5, 0.5, 0.5, 0.5])
    t_const = torch.tensor([0.2, 0.4, 0.6, 0.8])
    loss_const = ccc_fn(p_const, t_const)
    assert not torch.isnan(loss_const)
    assert not torch.isinf(loss_const)


def test_14_end_to_end_gradient_propagation():
    """14. Test backward gradients propagate from VA loss through all Phase 5 modules."""
    d = 32
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=2)
    va_reg = DimensionalVARegression(input_dim=3 * d, hidden_dim=32)

    asp = torch.randn(2, d, requires_grad=True)
    op = torch.randn(2, d, requires_grad=True)
    g_asp = torch.randn(2, d, requires_grad=True)
    g_op = torch.randn(2, d, requires_grad=True)

    fused = fusion(asp, op, g_asp, g_op)
    Z = fused["z"]
    out = va_reg(Z)

    loss = out["valence"].sum() + out["arousal"].sum()
    loss.backward()

    assert asp.grad is not None and torch.norm(asp.grad) > 0.0
    assert g_asp.grad is not None and torch.norm(g_asp.grad) > 0.0
    assert va_reg.dual_heads.valence_head.mlp[0].weight.grad is not None
    assert va_reg.dual_heads.arousal_head.mlp[0].weight.grad is not None


def test_15_no_nan_or_inf_in_forward_backward():
    """15. Test numerical stability with random large and small values."""
    d = 32
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=2)
    va_reg = DimensionalVARegression(input_dim=3 * d, hidden_dim=32)

    asp = (torch.randn(2, d) * 50.0).detach().requires_grad_(True)
    op = (torch.randn(2, d) * 50.0).detach().requires_grad_(True)

    fused = fusion(asp, op)
    out = va_reg(fused["z"])

    loss = out["valence"].sum()
    loss.backward()

    assert not torch.isnan(out["valence"]).any()
    assert not torch.isnan(out["arousal"]).any()
    assert not torch.isnan(asp.grad).any()


def test_16_batch_independence_and_isolation():
    """16. Test that sentence b0 outputs are unaffected when sentence b1 is modified."""
    d = 32
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=2)
    va_reg = DimensionalVARegression(input_dim=3 * d, hidden_dim=32)
    fusion.eval()
    va_reg.eval()

    asp1 = torch.randn(1, d)
    asp2 = torch.randn(1, d)
    op1 = torch.randn(1, d)
    op2 = torch.randn(1, d)

    with torch.no_grad():
        # Single execution of item 1
        z1 = fusion(asp1, op1)["z"]
        v1 = va_reg(z1)["valence"]

        # Batch execution of [item 1, item 2]
        batch_asp = torch.cat([asp1, asp2], dim=0)
        batch_op = torch.cat([op1, op2], dim=0)
        batch_z = fusion(batch_asp, batch_op)["z"]
        batch_v = va_reg(batch_z)["valence"]

    assert torch.allclose(v1[0], batch_v[0], atol=1e-5)


def test_17_compatibility_with_phase3_and_phase4_outputs():
    """17. Test that Phase 5 fusion consumes Phase 3 span representations and Phase 4 RGAT vectors."""
    d = 64
    fusion = AspectGuidedFusion(hidden_dim=d, num_heads=4)
    va_reg = DimensionalVARegression(input_dim=3 * d, hidden_dim=64)

    # Simulated outputs from Phase 3 BiaffineSpanExtractor
    phase3_asp_repr = torch.randn(2, d)
    phase3_op_repr = torch.randn(2, d)

    # Simulated outputs from Phase 4 RGAT + extract_graph_span_representation
    phase4_graph_asp = torch.randn(2, d)
    phase4_graph_op = torch.randn(2, d)
    phase4_all_nodes = torch.randn(2, 10, d)

    fused = fusion(
        aspect_span_repr=phase3_asp_repr,
        opinion_span_repr=phase3_op_repr,
        graph_aspect_repr=phase4_graph_asp,
        graph_opinion_repr=phase4_graph_op,
        graph_node_features=phase4_all_nodes,
    )
    assert fused["z"].shape == (2, 3 * d)

    preds = va_reg(fused["z"])
    assert preds["valence"].shape == (2, 1)
    assert preds["arousal"].shape == (2, 1)


def test_18_complete_nssg_dimnet_forward_pass():
    """18. Test full end-to-end forward pass of NSSGDimNet with structured output formatting."""
    B, seq_len, d = 2, 8, 64
    model = NSSGDimNet(
        hidden_dim=d,
        num_languages=4,
        max_signed_distance=16,
        num_spgsa_heads=4,
        span_mlp_dim=32,
        nrc_vad_dim=2,
        num_relations=3,
        rgat_layers=2,
        rgat_heads=4,
        cross_attn_heads=4,
        regression_hidden_dim=32,
        use_mock_backbone=True,
    )

    input_ids = torch.randint(0, 500, (B, seq_len))
    lang_ids = torch.randint(0, 3, (B, seq_len))
    attention_mask = torch.ones(B, seq_len)
    aspect_spans = torch.tensor([[1, 2], [3, 4]])
    opinion_spans = torch.tensor([[3, 4], [5, 6]])
    sentences = ["Yeh phone accha hai", "Camera quality bahut badiya"]
    aspect_texts = [["phone"], ["camera quality"]]
    opinion_texts = [["accha"], ["bahut badiya"]]

    outputs = model(
        input_ids=input_ids,
        lang_ids=lang_ids,
        attention_mask=attention_mask,
        aspect_spans=aspect_spans,
        opinion_spans=opinion_spans,
        sentences=sentences,
        aspect_texts_batch=aspect_texts,
        opinion_texts_batch=opinion_texts,
    )

    assert "valence" in outputs
    assert "arousal" in outputs
    assert "fused_representation" in outputs
    assert "structured_predictions" in outputs

    assert outputs["fused_representation"].shape == (B, 3 * d)
    assert outputs["valence"].shape == (B, 1)
    assert outputs["arousal"].shape == (B, 1)

    struct_preds = outputs["structured_predictions"]
    assert len(struct_preds) == 2
    assert "sentence" in struct_preds[0]
    assert "predictions" in struct_preds[0]
    assert len(struct_preds[0]["predictions"]) >= 1
    assert "valence" in struct_preds[0]["predictions"][0]
    assert "arousal" in struct_preds[0]["predictions"][0]
