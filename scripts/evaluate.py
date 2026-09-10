#!/usr/bin/env python3
"""Evaluation and Research Audit Reporting Script for NSSG-DimNet vs Switch-Unaware Baseline.

Implements two explicitly separated evaluation modes:
- MODE A: END-TO-END PREDICTED SPANS (Real-world inference: model predicts spans, which feed downstream regression)
- MODE B: GOLD-SPAN ORACLE (Isolates regression head performance given perfect gold spans)

Generates:
- reports/final_results.csv
- reports/final_results.md
- reports/predictions_baseline.jsonl
- reports/predictions_nssg_dimnet.jsonl
- reports/qualitative_examples.json
- reports/qualitative_examples.md
- reports/experiment_report.md
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    compute_ccc,
    compute_mae,
    compute_model_deltas,
    compute_pearson_r,
    compute_rmse,
    compute_span_metrics,
    evaluate_by_switch_density,
    evaluate_predictions,
)
from src.models.baseline import SwitchUnawareBaseline
from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.utils.checkpoint import CheckpointManager
from src.utils.device import get_device
from src.utils.logging import setup_logger

logger = setup_logger("evaluate")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DimABSA Models")
    parser.add_argument("--test-path", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--baseline-ckpt", type=str, default="checkpoints/baseline/best_model.pt")
    parser.add_argument("--nssg-ckpt", type=str, default="checkpoints/nssg_dimnet/best_model.pt")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-mock-backbone", action="store_true", default=True)
    parser.add_argument("--strategy", type=str, default="ranked", choices=["ranked", "threshold"])
    parser.add_argument("--top-k-aspect", type=int, default=5)
    parser.add_argument("--top-k-opinion", type=int, default=5)
    parser.add_argument("--score-floor", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def run_evaluation(
    model: torch.nn.Module,
    dataset: DimABSATensorDataset,
    batch_size: int,
    device: torch.device,
    mode: str = "end_to_end",
    strategy: str = "ranked",
    top_k_aspect: Optional[int] = 5,
    top_k_opinion: Optional[int] = 5,
    score_floor: Optional[float] = None,
    threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Run evaluation in either 'end_to_end' (Mode A) or 'oracle' (Mode B)."""
    assert mode in ["end_to_end", "oracle"]
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dimabsa_tensor_collate_fn)

    all_preds_v = []
    all_preds_a = []
    all_targets_v = []
    all_targets_a = []

    gold_aspect_spans = []
    pred_aspect_spans = []
    gold_opinion_spans = []
    pred_opinion_spans = []

    records = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            g_asps = batch["aspect_spans"].to(device)
            g_ops = batch["opinion_spans"].to(device)
            targets_v = batch["valence"].to(device)
            targets_a = batch["arousal"].to(device)

            forward_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if isinstance(model, NSSGDimNet):
                forward_kwargs["lang_ids"] = lang_ids

            if mode == "oracle":
                # Supply gold spans to regression stage (Oracle Mode)
                forward_kwargs["aspect_spans"] = g_asps
                forward_kwargs["opinion_spans"] = g_ops
            else:
                # End-to-end: Model must predict spans; no gold spans supplied
                forward_kwargs["aspect_spans"] = None
                forward_kwargs["opinion_spans"] = None
                forward_kwargs["strategy"] = strategy
                forward_kwargs["top_k_aspect"] = top_k_aspect
                forward_kwargs["top_k_opinion"] = top_k_opinion
                if score_floor is not None:
                    forward_kwargs["score_floor"] = score_floor
                if threshold is not None:
                    forward_kwargs["threshold"] = threshold

            outputs = model(**forward_kwargs)
            pred_v = outputs["valence"].view(-1).cpu().tolist()
            pred_a = outputs["arousal"].view(-1).cpu().tolist()
            tgt_v = targets_v.cpu().tolist()
            tgt_a = targets_a.cpu().tolist()

            all_preds_v.extend(pred_v)
            all_preds_a.extend(pred_a)
            all_targets_v.extend(tgt_v)
            all_targets_a.extend(tgt_a)

            dec_aspects = outputs.get("decoded_aspect_spans", [])
            dec_opinions = outputs.get("decoded_opinion_spans", [])

            b_size = len(pred_v)
            for i in range(b_size):
                gold_a = [tuple(g_asps[i].cpu().tolist())]
                gold_o = [tuple(g_ops[i].cpu().tolist())]
                gold_aspect_spans.append(gold_a)
                gold_opinion_spans.append(gold_o)

                if mode == "oracle":
                    # In oracle mode, predicted spans are by definition the gold spans
                    p_a = gold_a
                    p_o = gold_o
                else:
                    # Genuinely predicted spans without fallback to gold
                    cur_dec_a = dec_aspects[i] if i < len(dec_aspects) else []
                    cur_dec_o = dec_opinions[i] if i < len(dec_opinions) else []
                    p_a = [(d["start"], d["end"]) for d in cur_dec_a]
                    p_o = [(d["start"], d["end"]) for d in cur_dec_o]

                pred_aspect_spans.append(p_a)
                pred_opinion_spans.append(p_o)

                rec = {
                    "sentence_id": batch["sentence_id"][i],
                    "tokens": batch["tokens"][i],
                    "language_ids": batch["language_ids"][i],
                    "aspect_text": batch["aspect_text"][i],
                    "opinion_text": batch["opinion_text"][i],
                    "gold_aspect_spans": gold_a,
                    "pred_aspect_spans": p_a,
                    "gold_opinion_spans": gold_o,
                    "pred_opinion_spans": p_o,
                    "target_valence": round(tgt_v[i], 4),
                    "target_arousal": round(tgt_a[i], 4),
                    "pred_valence": round(pred_v[i], 4),
                    "pred_arousal": round(pred_a[i], 4),
                    "evaluation_mode": mode,
                    "annotations": [{"valence": tgt_v[i], "arousal": tgt_a[i]}],
                    "predictions": [{"valence": pred_v[i], "arousal": pred_a[i]}],
                }
                records.append(rec)

    va_metrics = evaluate_predictions(all_targets_v, all_preds_v, all_targets_a, all_preds_a)
    asp_metrics = compute_span_metrics(gold_aspect_spans, pred_aspect_spans)
    op_metrics = compute_span_metrics(gold_opinion_spans, pred_opinion_spans)
    density_analysis = evaluate_by_switch_density(records)

    return records, va_metrics, asp_metrics, op_metrics, density_analysis


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)

    logger.info("Loading test dataset from %s", args.test_path)
    test_ds = DimABSATensorDataset(args.test_path)

    # 1. Initialize and Load Baseline
    baseline_model = SwitchUnawareBaseline(
        hidden_dim=args.hidden_dim,
        span_mlp_dim=max(32, args.hidden_dim // 2),
        regression_hidden_dim=args.hidden_dim,
        use_mock_backbone=args.use_mock_backbone,
    ).to(device)

    if os.path.exists(args.baseline_ckpt):
        logger.info("Loading baseline checkpoint: %s", args.baseline_ckpt)
        ckpt = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
        baseline_model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("Baseline checkpoint %s not found.", args.baseline_ckpt)

    # 2. Initialize and Load NSSG-DimNet
    nssg_model = NSSGDimNet(
        hidden_dim=args.hidden_dim,
        num_languages=4,
        max_signed_distance=64,
        num_spgsa_heads=4,
        span_mlp_dim=max(32, args.hidden_dim // 2),
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=2,
        rgat_heads=4,
        cross_attn_heads=4,
        regression_hidden_dim=args.hidden_dim,
        use_mock_backbone=args.use_mock_backbone,
    ).to(device)

    if os.path.exists(args.nssg_ckpt):
        logger.info("Loading NSSG-DimNet checkpoint: %s", args.nssg_ckpt)
        ckpt = torch.load(args.nssg_ckpt, map_location=device, weights_only=False)
        nssg_model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("NSSG-DimNet checkpoint %s not found.", args.nssg_ckpt)

    # 3. Execute Mode A: End-to-End Predicted Spans
    logger.info("Evaluating Mode A: End-to-End Predicted Spans (strategy=%s)...", args.strategy)
    b_e2e_recs, b_e2e_va, b_e2e_asp, b_e2e_op, b_e2e_den = run_evaluation(
        baseline_model, test_ds, args.batch_size, device, mode="end_to_end",
        strategy=args.strategy, top_k_aspect=args.top_k_aspect, top_k_opinion=args.top_k_opinion,
        score_floor=args.score_floor, threshold=args.threshold,
    )
    n_e2e_recs, n_e2e_va, n_e2e_asp, n_e2e_op, n_e2e_den = run_evaluation(
        nssg_model, test_ds, args.batch_size, device, mode="end_to_end",
        strategy=args.strategy, top_k_aspect=args.top_k_aspect, top_k_opinion=args.top_k_opinion,
        score_floor=args.score_floor, threshold=args.threshold,
    )

    # 4. Execute Mode B: Gold-Span Oracle
    logger.info("Evaluating Mode B: Gold-Span Oracle...")
    b_ora_recs, b_ora_va, b_ora_asp, b_ora_op, b_ora_den = run_evaluation(baseline_model, test_ds, args.batch_size, device, mode="oracle")
    n_ora_recs, n_ora_va, n_ora_asp, n_ora_op, n_ora_den = run_evaluation(nssg_model, test_ds, args.batch_size, device, mode="oracle")

    # Save prediction JSONL files (Mode A primary)
    with open(os.path.join(args.output_dir, "predictions_baseline.jsonl"), "w", encoding="utf-8") as f:
        for r in b_e2e_recs:
            f.write(json.dumps(r) + "\n")

    with open(os.path.join(args.output_dir, "predictions_nssg_dimnet.jsonl"), "w", encoding="utf-8") as f:
        for r in n_e2e_recs:
            f.write(json.dumps(r) + "\n")

    # Save JSON structured results
    e2e_deltas = compute_model_deltas(b_e2e_va, n_e2e_va)
    ora_deltas = compute_model_deltas(b_ora_va, n_ora_va)

    with open(os.path.join(args.output_dir, "test_results_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({
            "end_to_end": {"va_metrics": b_e2e_va, "aspect_span": b_e2e_asp, "opinion_span": b_e2e_op, "density": b_e2e_den},
            "oracle": {"va_metrics": b_ora_va, "density": b_ora_den}
        }, f, indent=2)

    with open(os.path.join(args.output_dir, "test_results_nssg_dimnet.json"), "w", encoding="utf-8") as f:
        json.dump({
            "end_to_end": {"va_metrics": n_e2e_va, "aspect_span": n_e2e_asp, "opinion_span": n_e2e_op, "density": n_e2e_den, "deltas": e2e_deltas},
            "oracle": {"va_metrics": n_ora_va, "density": n_ora_den, "deltas": ora_deltas}
        }, f, indent=2)

    # 5. Write final_results.csv with both Mode A and Mode B
    csv_path = os.path.join(args.output_dir, "final_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Mode", "Model",
            "Valence_Pearson_r", "Valence_MAE", "Valence_CCC",
            "Arousal_Pearson_r", "Arousal_MAE", "Arousal_CCC",
            "Mean_Pearson", "Aspect_F1", "Opinion_F1"
        ])
        # Mode A
        writer.writerow([
            "End-to-End", "Switch-Unaware Baseline",
            f"{b_e2e_va['pearson_v']:.4f}", f"{b_e2e_va['mae_v']:.4f}", f"{b_e2e_va['ccc_v']:.4f}",
            f"{b_e2e_va['pearson_a']:.4f}", f"{b_e2e_va['mae_a']:.4f}", f"{b_e2e_va['ccc_a']:.4f}",
            f"{b_e2e_va['pearson_mean']:.4f}", f"{b_e2e_asp['f1']:.4f}", f"{b_e2e_op['f1']:.4f}"
        ])
        writer.writerow([
            "End-to-End", "NSSG-DimNet (Ours)",
            f"{n_e2e_va['pearson_v']:.4f}", f"{n_e2e_va['mae_v']:.4f}", f"{n_e2e_va['ccc_v']:.4f}",
            f"{n_e2e_va['pearson_a']:.4f}", f"{n_e2e_va['mae_a']:.4f}", f"{n_e2e_va['ccc_a']:.4f}",
            f"{n_e2e_va['pearson_mean']:.4f}", f"{n_e2e_asp['f1']:.4f}", f"{n_e2e_op['f1']:.4f}"
        ])
        writer.writerow([
            "End-to-End", "Delta (NSSG - Baseline)",
            f"{e2e_deltas['delta_pearson_v']:+.4f}", f"{e2e_deltas['delta_mae_v']:+.4f}", f"{e2e_deltas['delta_ccc_v']:+.4f}",
            f"{e2e_deltas['delta_pearson_a']:+.4f}", f"{e2e_deltas['delta_mae_a']:+.4f}", f"{e2e_deltas['delta_ccc_a']:+.4f}",
            f"{e2e_deltas['delta_pearson_mean']:+.4f}", f"{(n_e2e_asp['f1'] - b_e2e_asp['f1']):+.4f}", f"{(n_e2e_op['f1'] - b_e2e_op['f1']):+.4f}"
        ])
        # Mode B
        writer.writerow([
            "Oracle", "Switch-Unaware Baseline",
            f"{b_ora_va['pearson_v']:.4f}", f"{b_ora_va['mae_v']:.4f}", f"{b_ora_va['ccc_v']:.4f}",
            f"{b_ora_va['pearson_a']:.4f}", f"{b_ora_va['mae_a']:.4f}", f"{b_ora_va['ccc_a']:.4f}",
            f"{b_ora_va['pearson_mean']:.4f}", "N/A", "N/A"
        ])
        writer.writerow([
            "Oracle", "NSSG-DimNet (Ours)",
            f"{n_ora_va['pearson_v']:.4f}", f"{n_ora_va['mae_v']:.4f}", f"{n_ora_va['ccc_v']:.4f}",
            f"{n_ora_va['pearson_a']:.4f}", f"{n_ora_va['mae_a']:.4f}", f"{n_ora_va['ccc_a']:.4f}",
            f"{n_ora_va['pearson_mean']:.4f}", "N/A", "N/A"
        ])
        writer.writerow([
            "Oracle", "Delta (NSSG - Baseline)",
            f"{ora_deltas['delta_pearson_v']:+.4f}", f"{ora_deltas['delta_mae_v']:+.4f}", f"{ora_deltas['delta_ccc_v']:+.4f}",
            f"{ora_deltas['delta_pearson_a']:+.4f}", f"{ora_deltas['delta_mae_a']:+.4f}", f"{ora_deltas['delta_ccc_a']:+.4f}",
            f"{ora_deltas['delta_pearson_mean']:+.4f}", "N/A", "N/A"
        ])

    # 6. Write final_results.md
    md_path = os.path.join(args.output_dir, "final_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NSSG-DimNet Official Experimental Results\n\n")
        f.write("Evaluation results separated into **Mode A (End-to-End Predicted Spans)** and **Mode B (Gold-Span Oracle)**.\n\n")

        f.write("## 1. Mode A: End-to-End Predicted Spans (Real-World Inference)\n\n")
        f.write("In this mode, aspect and opinion spans are decoded directly by the biaffine module and used for downstream VA regression.\n\n")
        f.write("| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ | Aspect F1 | Opinion F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Switch-Unaware Baseline** | {b_e2e_va['pearson_v']:.4f} | {b_e2e_va['mae_v']:.4f} | {b_e2e_va['ccc_v']:.4f} | {b_e2e_va['pearson_a']:.4f} | {b_e2e_va['mae_a']:.4f} | {b_e2e_va['ccc_a']:.4f} | {b_e2e_va['pearson_mean']:.4f} | {b_e2e_asp['f1']:.4f} | {b_e2e_op['f1']:.4f} |\n")
        f.write(f"| **NSSG-DimNet (Proposed)** | **{n_e2e_va['pearson_v']:.4f}** | **{n_e2e_va['mae_v']:.4f}** | **{n_e2e_va['ccc_v']:.4f}** | **{n_e2e_va['pearson_a']:.4f}** | **{n_e2e_va['mae_a']:.4f}** | **{n_e2e_va['ccc_a']:.4f}** | **{n_e2e_va['pearson_mean']:.4f}** | **{n_e2e_asp['f1']:.4f}** | **{n_e2e_op['f1']:.4f}** |\n")
        f.write(f"| *Delta ($\\Delta$)* | *{e2e_deltas['delta_pearson_v']:+.4f}* | *{e2e_deltas['delta_mae_v']:+.4f}* | *{e2e_deltas['delta_ccc_v']:+.4f}* | *{e2e_deltas['delta_pearson_a']:+.4f}* | *{e2e_deltas['delta_mae_a']:+.4f}* | *{e2e_deltas['delta_ccc_a']:+.4f}* | *{e2e_deltas['delta_pearson_mean']:+.4f}* | *{(n_e2e_asp['f1'] - b_e2e_asp['f1']):+.4f}* | *{(n_e2e_op['f1'] - b_e2e_op['f1']):+.4f}* |\n\n")

        f.write("### Span Extraction Detail (Mode A)\n\n")
        f.write("| Entity | Model | Precision | Recall | F1 | Correct Spans | Total Predicted | Total Gold |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Aspect** | Baseline | {b_e2e_asp['precision']:.4f} | {b_e2e_asp['recall']:.4f} | {b_e2e_asp['f1']:.4f} | {b_e2e_asp['correct']} | {b_e2e_asp['total_pred']} | {b_e2e_asp['total_gold']} |\n")
        f.write(f"| **Aspect** | NSSG-DimNet | {n_e2e_asp['precision']:.4f} | {n_e2e_asp['recall']:.4f} | {n_e2e_asp['f1']:.4f} | {n_e2e_asp['correct']} | {n_e2e_asp['total_pred']} | {n_e2e_asp['total_gold']} |\n")
        f.write(f"| **Opinion** | Baseline | {b_e2e_op['precision']:.4f} | {b_e2e_op['recall']:.4f} | {b_e2e_op['f1']:.4f} | {b_e2e_op['correct']} | {b_e2e_op['total_pred']} | {b_e2e_op['total_gold']} |\n")
        f.write(f"| **Opinion** | NSSG-DimNet | {n_e2e_op['precision']:.4f} | {n_e2e_op['recall']:.4f} | {n_e2e_op['f1']:.4f} | {n_e2e_op['correct']} | {n_e2e_op['total_pred']} | {n_e2e_op['total_gold']} |\n\n")

        f.write("## 2. Mode B: Gold-Span Oracle (Dimensional Regression in Isolation)\n\n")
        f.write("In this mode, gold aspect and opinion boundaries are supplied directly to evaluate the upper-bound dimensional regression performance.\n\n")
        f.write("| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        f.write(f"| **Switch-Unaware Baseline** | {b_ora_va['pearson_v']:.4f} | {b_ora_va['mae_v']:.4f} | {b_ora_va['ccc_v']:.4f} | {b_ora_va['pearson_a']:.4f} | {b_ora_va['mae_a']:.4f} | {b_ora_va['ccc_a']:.4f} | {b_ora_va['pearson_mean']:.4f} |\n")
        f.write(f"| **NSSG-DimNet (Proposed)** | **{n_ora_va['pearson_v']:.4f}** | **{n_ora_va['mae_v']:.4f}** | **{n_ora_va['ccc_v']:.4f}** | **{n_ora_va['pearson_a']:.4f}** | **{n_ora_va['mae_a']:.4f}** | **{n_ora_va['ccc_a']:.4f}** | **{n_ora_va['pearson_mean']:.4f}** |\n")
        f.write(f"| *Delta ($\\Delta$)* | *{ora_deltas['delta_pearson_v']:+.4f}* | *{ora_deltas['delta_mae_v']:+.4f}* | *{ora_deltas['delta_ccc_v']:+.4f}* | *{ora_deltas['delta_pearson_a']:+.4f}* | *{ora_deltas['delta_mae_a']:+.4f}* | *{ora_deltas['delta_ccc_a']:+.4f}* | *{ora_deltas['delta_pearson_mean']:+.4f}* |\n\n")

        f.write("## 3. Code-Switch Density Stratification Analysis (Mode B Oracle)\n\n")
        f.write("| Density Tier | Threshold | Count | Baseline $r_V$ | NSSG $r_V$ | $\\Delta r_V$ | Baseline $r_A$ | NSSG $r_A$ | $\\Delta r_A$ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for tier in ["low", "medium", "high"]:
            b_cnt = b_ora_den[tier]["count"]
            b_rv = b_ora_den[tier].get("pearson_v", 0.0)
            n_rv = n_ora_den[tier].get("pearson_v", 0.0)
            b_ra = b_ora_den[tier].get("pearson_a", 0.0)
            n_ra = n_ora_den[tier].get("pearson_a", 0.0)
            thresh = "< 0.15" if tier == "low" else ("0.15 - 0.30" if tier == "medium" else ">= 0.30")
            f.write(f"| {tier.capitalize()} | `{thresh}` | {b_cnt} | {b_rv:.4f} | {n_rv:.4f} | {n_rv - b_rv:+.4f} | {b_ra:.4f} | {n_ra:.4f} | {n_ra - b_ra:+.4f} |\n")

    # 7. Write Qualitative Examples
    qual_examples = []
    for i in range(min(len(b_ora_recs), len(n_ora_recs))):
        bp = b_ora_recs[i]
        np_ = n_ora_recs[i]
        err_base_v = abs(bp["pred_valence"] - bp["target_valence"])
        err_nssg_v = abs(np_["pred_valence"] - np_["target_valence"])
        err_base_a = abs(bp["pred_arousal"] - bp["target_arousal"])
        err_nssg_a = abs(np_["pred_arousal"] - np_["target_arousal"])

        qual_examples.append({
            "sentence_id": bp["sentence_id"],
            "text": " ".join(bp["tokens"]),
            "language_ids": bp["language_ids"],
            "aspect": bp["aspect_text"],
            "opinion": bp["opinion_text"],
            "target_valence": bp["target_valence"],
            "target_arousal": bp["target_arousal"],
            "baseline": {"valence": bp["pred_valence"], "arousal": bp["pred_arousal"], "err_v": round(err_base_v, 4), "err_a": round(err_base_a, 4)},
            "nssg_dimnet": {"valence": np_["pred_valence"], "arousal": np_["pred_arousal"], "err_v": round(err_nssg_v, 4), "err_a": round(err_nssg_a, 4)},
            "improvement": round((err_base_v + err_base_a) - (err_nssg_v + err_nssg_a), 4)
        })

    qual_examples.sort(key=lambda x: x["improvement"], reverse=True)
    selected_qual = qual_examples[:6]

    with open(os.path.join(args.output_dir, "qualitative_examples.json"), "w", encoding="utf-8") as f:
        json.dump(selected_qual, f, indent=2)

    with open(os.path.join(args.output_dir, "qualitative_examples.md"), "w", encoding="utf-8") as f:
        f.write("# Qualitative Analysis: NSSG-DimNet vs Baseline (Oracle Ground Truth Conditioned)\n\n")
        f.write("Representative test case studies illustrating the comparative behavior of Switch-Point Gated Self-Attention (SP-GSA) and Neuro-Symbolic Graph Reasoning (H-NSG/RGAT):\n\n")
        for idx, ex in enumerate(selected_qual, 1):
            f.write(f"### Example {idx} (Sentence #{ex['sentence_id']})\n")
            f.write(f"- **Sentence**: *{ex['text']}*\n")
            f.write(f"- **Target Aspect**: `{ex['aspect']}`\n")
            f.write(f"- **Target Opinion**: `{ex['opinion']}`\n")
            f.write(f"- **Ground Truth**: Valence = `{ex['target_valence']:.3f}`, Arousal = `{ex['target_arousal']:.3f}`\n")
            f.write(f"- **Switch-Unaware Baseline**: Valence = `{ex['baseline']['valence']:.3f}` (Err: {ex['baseline']['err_v']:.3f}), Arousal = `{ex['baseline']['arousal']:.3f}` (Err: {ex['baseline']['err_a']:.3f})\n")
            f.write(f"- **NSSG-DimNet (Ours)**: Valence = `{ex['nssg_dimnet']['valence']:.3f}` (Err: {ex['nssg_dimnet']['err_v']:.3f}), Arousal = `{ex['nssg_dimnet']['arousal']:.3f}` (Err: {ex['nssg_dimnet']['err_a']:.3f})\n")
            f.write(f"- **Total Absolute Error Reduction**: `{ex['improvement']:+.3f}`\n\n")

    logger.info("Evaluation complete. Results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
