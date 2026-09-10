"""Demonstration, verification, and loss smoke test script for Phase 5:

1. Synthetic End-to-End Test (Multi-Aspect sentences, Z in R^(3d), continuous VA in [0, 1]).
2. Actual Dataset End-to-End Test on `data/splits/train.jsonl`.
3. Loss Smoke Test (CCCLoss + Smooth L1 + JointVALoss).
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.nssg_dimnet import NSSGDimNet
from src.models.fusion import AspectGuidedFusion
from src.models.va_regression import DimensionalVARegression
from src.losses.va_loss import CCCLoss, JointVALoss


def run_synthetic_test():
    print("\n============================================================")
    print("1. SYNTHETIC END-TO-END TEST (MULTI-ASPECT)")
    print("============================================================")
    B, seq_len, d = 1, 10, 64
    num_pairs = 2  # 2 aspects in sentence

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
    lang_ids = torch.tensor([[0, 0, 1, 1, 0, 0, 1, 0, 0, 0]])
    attention_mask = torch.ones(B, seq_len)
    aspect_spans = torch.tensor([[[2, 3], [6, 6]]])   # (1, 2, 2)
    opinion_spans = torch.tensor([[[4, 5], [8, 9]]])  # (1, 2, 2)
    sentences = ["Yeh phone ka camera aur battery life bohot mast hai"]
    aspect_texts = [["camera", "battery life"]]
    opinion_texts = [["bohot mast", "bohot mast"]]

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

    Z = outputs["fused_representation"]
    v_preds = outputs["valence"]
    a_preds = outputs["arousal"]

    print(f"Fused Representation Z shape: {Z.shape} (Expected: (1, 2, {3*d}))")
    print(f"Valence Predictions shape:    {v_preds.shape} (Values: {v_preds.squeeze().tolist()})")
    print(f"Arousal Predictions shape:    {a_preds.shape} (Values: {a_preds.squeeze().tolist()})")

    assert Z.shape[-1] == 3 * d, f"Expected {3*d}, got {Z.shape[-1]}"
    assert (v_preds >= 0.0).all() and (v_preds <= 1.0).all()
    assert (a_preds >= 0.0).all() and (a_preds <= 1.0).all()

    # Verify structured predictions schema
    print("\nFormatted Structured Predictions:")
    print(json.dumps(outputs["structured_predictions"], indent=2))


def run_real_dataset_test():
    print("\n============================================================")
    print("2. ACTUAL DATASET END-TO-END TEST (data/splits/train.jsonl)")
    print("============================================================")

    train_path = Path("data/splits/train.jsonl")
    if not train_path.exists():
        print("data/splits/train.jsonl not found, skipping.")
        return

    # Load first 4 real records
    samples = []
    with open(train_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= 4:
                break
            samples.append(json.loads(line))

    d = 64
    model = NSSGDimNet(
        hidden_dim=d,
        num_languages=4,
        max_signed_distance=32,
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

    print(f"Loaded {len(samples)} real training examples.")

    for i, s in enumerate(samples):
        s_id = s.get("sentence_id", s.get("id", f"sample_{i+1}"))
        s_text = s.get("text", s.get("sentence", ""))
        tokens = s.get("tokens", s_text.split())
        l_ids = s.get("language_ids", [0] * len(tokens))
        lang_map = {"HI": 0, "HINDI": 0, "EN": 1, "ENGLISH": 1, "MIXED": 2, "MIX": 2, "OTHER": 3, "SPECIAL": 3, "PAD": 3}
        int_l_ids = [lang_map.get(str(x).upper(), 0) if isinstance(x, str) else int(x) for x in l_ids]

        annotations = s.get("annotations", s.get("quadruplets", []))
        asp_spans = []
        op_spans = []
        asp_texts = []
        op_texts = []
        gold_v = []
        gold_a = []

        for ann in annotations:
            asp_s = ann.get("aspect_start", ann.get("aspect_span", [0, 0])[0])
            asp_e = ann.get("aspect_end", ann.get("aspect_span", [0, 0])[1])
            op_s = ann.get("opinion_start", ann.get("opinion_span", [0, 0])[0])
            op_e = ann.get("opinion_end", ann.get("opinion_span", [0, 0])[1])

            asp_spans.append([asp_s, asp_e])
            op_spans.append([op_s, op_e])
            asp_texts.append(ann.get("aspect_text", ""))
            op_texts.append(ann.get("opinion_text", ""))
            gold_v.append(float(ann.get("valence", 0.5)))
            gold_a.append(float(ann.get("arousal", 0.5)))

        if not asp_spans:
            asp_spans = [[0, 0]]
            op_spans = [[0, 0]]
            asp_texts = [""]
            op_texts = [""]
            gold_v = [0.5]
            gold_a = [0.5]

        seq_len = len(tokens)
        input_ids = torch.randint(0, 500, (1, seq_len))
        lang_tensor = torch.tensor([int_l_ids], dtype=torch.long)
        asp_tensor = torch.tensor([asp_spans], dtype=torch.long)
        op_tensor = torch.tensor([op_spans], dtype=torch.long)

        out = model(
            input_ids=input_ids,
            lang_ids=lang_tensor,
            attention_mask=torch.ones(1, seq_len),
            aspect_spans=asp_tensor,
            opinion_spans=op_tensor,
            tokens_batch=[tokens],
            sentences=[s_text],
            aspect_texts_batch=[asp_texts],
            opinion_texts_batch=[op_texts],
            sentence_ids=[s_id],
        )

        preds = out["structured_predictions"][0]
        print(f"\n[Record {i+1}] Sentence ID: {s_id}")
        print(f"Sentence: \"{s_text}\"")
        print(f"Predicted Aspects: {len(preds['predictions'])} | Fused Dim: {out['fused_representation'].shape[-1]}")
        for p_idx, p in enumerate(preds["predictions"]):
            print(f"  -> Pair {p_idx+1}: Aspect='{p['aspect']}' | Opinion='{p['opinion']}' | Pred (V,A)=({p['valence']:.3f}, {p['arousal']:.3f}) | Gold (V,A)=({gold_v[p_idx]:.3f}, {gold_a[p_idx]:.3f})")


def run_loss_smoke_test():
    print("\n============================================================")
    print("3. LOSS SMOKE TEST (CCCLoss + Smooth L1 + JointVALoss)")
    print("============================================================")
    joint_loss_fn = JointVALoss(
        ccc_weight_v=1.0,
        ccc_weight_a=1.0,
        smooth_l1_weight_v=0.5,
        smooth_l1_weight_a=0.5,
    )

    pred_v = torch.tensor([0.25, 0.65, 0.85, 0.40], requires_grad=True)
    pred_a = torch.tensor([0.30, 0.70, 0.50, 0.80], requires_grad=True)
    target_v = torch.tensor([0.20, 0.70, 0.80, 0.45])
    target_a = torch.tensor([0.35, 0.65, 0.55, 0.75])

    loss_dict = joint_loss_fn(pred_v, pred_a, target_v, target_a)

    print(f"Valence Smooth L1 Loss: {loss_dict['loss_smooth_l1_v'].item():.4f}")
    print(f"Arousal Smooth L1 Loss: {loss_dict['loss_smooth_l1_a'].item():.4f}")
    print(f"Valence CCC Loss:       {loss_dict['loss_ccc_v'].item():.4f}")
    print(f"Arousal CCC Loss:       {loss_dict['loss_ccc_a'].item():.4f}")
    print(f"Total Joint Loss:       {loss_dict['total_loss'].item():.4f}")

    loss_dict["total_loss"].backward()
    assert pred_v.grad is not None and pred_a.grad is not None
    print(f"Gradients computed successfully! Grad norm V: {pred_v.grad.norm().item():.4f}, A: {pred_a.grad.norm().item():.4f}")
    print("\nPhase 5 verification script completed successfully!")


if __name__ == "__main__":
    run_synthetic_test()
    run_real_dataset_test()
    run_loss_smoke_test()
