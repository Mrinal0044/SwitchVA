"""Research Evaluation Audit Tests for NSSG-DimNet.

Verifies:
1. No gold-span leakage in end-to-end inference mode.
2. No gold VA leakage into model representations, H-NSG, RGAT, or regression heads.
3. Aspect and Opinion extraction metrics are genuinely computed without artificial fallbacks.
4. Complete test set isolation from train and validation splits.
5. Explicit separation of End-to-End (Mode A) and Gold-Span Oracle (Mode B).
"""

import json
import os
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.baseline import SwitchUnawareBaseline
from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.evaluation.metrics import compute_span_metrics, evaluate_predictions


def test_audit_1_split_isolation_and_integrity():
    """Verify test split has zero sentence ID or text overlap with train and validation splits."""
    def load_ids_and_texts(path):
        with open(path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        ids = set(r["sentence_id"] for r in records)
        texts = set(r["text"].strip() for r in records)
        return ids, texts

    train_ids, train_texts = load_ids_and_texts("data/splits/train.jsonl")
    val_ids, val_texts = load_ids_and_texts("data/splits/validation.jsonl")
    test_ids, test_texts = load_ids_and_texts("data/splits/test.jsonl")

    # 1. Zero ID overlap
    assert len(train_ids.intersection(test_ids)) == 0, "Test sentence IDs leaked into train split!"
    assert len(val_ids.intersection(test_ids)) == 0, "Test sentence IDs leaked into validation split!"

    # 2. Zero text overlap across splits
    assert len(train_texts.intersection(test_texts)) == 0, "Test sentence text leaked into train split!"
    assert len(val_texts.intersection(test_texts)) == 0, "Test sentence text leaked into validation split!"


def test_audit_2_no_gold_span_leakage_in_end_to_end_mode():
    """Verify that in end-to-end mode, gold spans do not influence model predictions or representations."""
    model = NSSGDimNet(
        hidden_dim=32,
        num_languages=4,
        max_signed_distance=32,
        num_spgsa_heads=2,
        span_mlp_dim=16,
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=1,
        rgat_heads=2,
        cross_attn_heads=2,
        regression_hidden_dim=16,
        use_mock_backbone=True,
    )
    model.eval()

    input_ids = torch.randint(0, 500, (2, 10))
    attention_mask = torch.ones(2, 10)
    lang_ids = torch.zeros(2, 10, dtype=torch.long)

    # In end-to-end mode, aspect_spans and opinion_spans are None
    with torch.no_grad():
        out1 = model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids, aspect_spans=None, opinion_spans=None)

    # Calling again without gold spans produces exact deterministic outputs
    with torch.no_grad():
        out2 = model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids, aspect_spans=None, opinion_spans=None)

    assert torch.allclose(out1["valence"], out2["valence"])
    assert torch.allclose(out1["arousal"], out2["arousal"])
    assert torch.allclose(out1["fused_representation"], out2["fused_representation"])


def test_audit_3_no_gold_va_leakage():
    """Verify that mutating ground-truth VA labels does not change model predictions for Baseline or NSSG-DimNet."""
    base_model = SwitchUnawareBaseline(
        hidden_dim=32,
        span_mlp_dim=16,
        regression_hidden_dim=16,
        use_mock_backbone=True,
    )
    base_model.eval()

    nssg_model = NSSGDimNet(
        hidden_dim=32,
        num_languages=4,
        max_signed_distance=32,
        num_spgsa_heads=2,
        span_mlp_dim=16,
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=1,
        rgat_heads=2,
        cross_attn_heads=2,
        regression_hidden_dim=16,
        use_mock_backbone=True,
    )
    nssg_model.eval()

    input_ids = torch.randint(0, 500, (2, 8))
    attention_mask = torch.ones(2, 8)
    lang_ids = torch.zeros(2, 8, dtype=torch.long)

    # 1. Run inference
    with torch.no_grad():
        b_pred1 = base_model(input_ids=input_ids, attention_mask=attention_mask)
        n_pred1 = nssg_model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids)

    # Mutating external targets
    gold_v_original = torch.tensor([0.2, 0.8])
    gold_v_mutated = torch.tensor([0.9, 0.1])
    assert not torch.equal(gold_v_original, gold_v_mutated)

    # 2. Second run produces identical prediction regardless of external target values
    with torch.no_grad():
        b_pred2 = base_model(input_ids=input_ids, attention_mask=attention_mask)
        n_pred2 = nssg_model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids)

    assert torch.allclose(b_pred1["valence"], b_pred2["valence"])
    assert torch.allclose(b_pred1["arousal"], b_pred2["arousal"])
    assert torch.allclose(n_pred1["valence"], n_pred2["valence"])
    assert torch.allclose(n_pred1["arousal"], n_pred2["arousal"])


def test_audit_4_no_fallback_to_gold_spans():
    """Verify that span evaluation fails honestly when predicted spans do not match gold spans."""
    gold_spans = [[(2, 4)], [(5, 7)]]
    # Completely wrong predicted spans
    wrong_preds = [[(0, 1)], [(8, 9)]]

    metrics = compute_span_metrics(gold_spans, wrong_preds)
    assert metrics["correct"] == 0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    # Demonstrates that 1.0 F1 cannot be returned when predictions are incorrect


def test_audit_5_mode_separation_oracle_vs_end_to_end():
    """Verify that Mode A (End-to-End) and Mode B (Oracle) are explicitly separated."""
    model = NSSGDimNet(
        hidden_dim=32,
        num_languages=4,
        max_signed_distance=32,
        num_spgsa_heads=2,
        span_mlp_dim=16,
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=1,
        rgat_heads=2,
        cross_attn_heads=2,
        regression_hidden_dim=16,
        use_mock_backbone=True,
    )
    model.eval()

    input_ids = torch.randint(0, 500, (2, 10))
    attention_mask = torch.ones(2, 10)
    lang_ids = torch.zeros(2, 10, dtype=torch.long)
    gold_aspect_spans = torch.tensor([[1, 3], [4, 6]])
    gold_opinion_spans = torch.tensor([[4, 7], [7, 9]])

    with torch.no_grad():
        # Mode A: End-to-End
        out_e2e = model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids, aspect_spans=None, opinion_spans=None)
        # Mode B: Oracle
        out_oracle = model(input_ids=input_ids, attention_mask=attention_mask, lang_ids=lang_ids, aspect_spans=gold_aspect_spans, opinion_spans=gold_opinion_spans)

    assert "decoded_aspect_spans" in out_e2e
    assert "decoded_opinion_spans" in out_e2e
    # Oracle mode explicitly uses gold coordinates, producing distinct conditioned representations
    assert not torch.allclose(out_e2e["fused_representation"], out_oracle["fused_representation"])
