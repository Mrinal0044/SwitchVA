"""Step 8: Threshold vs Ranked Top-K analysis on the validation set."""

import json
import os
import torch
from torch.utils.data import DataLoader

from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.evaluation.metrics import compute_span_metrics, evaluate_predictions

def eval_val_with_config(model, val_loader, strategy="ranked", top_k=5, threshold=0.0, score_floor=None):
    model.eval()
    all_preds_v = []
    all_preds_a = []
    all_targets_v = []
    all_targets_a = []
    
    pred_aspects = []
    gold_aspects = []
    pred_opinions = []
    gold_opinions = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                lang_ids=batch["lang_ids"],
                strategy=strategy,
                top_k=top_k,
                top_k_aspect=top_k,
                top_k_opinion=top_k,
                threshold=threshold,
                score_floor=score_floor,
            )
            
            pred_v = outputs["valence"].view(-1).cpu().tolist()
            pred_a = outputs["arousal"].view(-1).cpu().tolist()
            tgt_v = batch["valence"].cpu().tolist()
            tgt_a = batch["arousal"].cpu().tolist()
            
            all_preds_v.extend(pred_v)
            all_preds_a.extend(pred_a)
            all_targets_v.extend(tgt_v)
            all_targets_a.extend(tgt_a)
            
            dec_a = outputs.get("decoded_aspect_spans", [])
            dec_o = outputs.get("decoded_opinion_spans", [])
            
            for i in range(len(pred_v)):
                cur_a = dec_a[i] if i < len(dec_a) else []
                cur_o = dec_o[i] if i < len(dec_o) else []
                
                pred_aspects.append([(d["start"], d["end"]) for d in cur_a])
                pred_opinions.append([(d["start"], d["end"]) for d in cur_o])
                gold_aspects.append(batch["gold_aspect_spans"][i])
                gold_opinions.append(batch["gold_opinion_spans"][i])
                
    asp_metrics = compute_span_metrics(gold_aspects, pred_aspects)
    op_metrics = compute_span_metrics(gold_opinions, pred_opinions)
    va_metrics = evaluate_predictions(all_preds_v, all_targets_v, all_preds_a, all_targets_a)
    
    return {
        "aspect_f1": asp_metrics["f1"],
        "aspect_precision": asp_metrics["precision"],
        "aspect_recall": asp_metrics["recall"],
        "opinion_f1": op_metrics["f1"],
        "opinion_precision": op_metrics["precision"],
        "opinion_recall": op_metrics["recall"],
        "pearson_v": va_metrics["pearson_v"],
        "pearson_a": va_metrics["pearson_a"],
        "pearson_mean": va_metrics["pearson_mean"],
        "mae_v": va_metrics["mae_v"],
        "mae_a": va_metrics["mae_a"],
        "ccc_v": va_metrics["ccc_v"],
        "ccc_a": va_metrics["ccc_a"],
        "total_pred_aspects": asp_metrics["total_pred"],
        "total_pred_opinions": op_metrics["total_pred"],
    }

def main():
    val_ds = DimABSATensorDataset("data/splits/validation.jsonl")
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=dimabsa_tensor_collate_fn)
    
    model = NSSGDimNet(
        hidden_dim=128,
        num_languages=4,
        max_signed_distance=64,
        num_spgsa_heads=4,
        span_mlp_dim=64,
        nrc_vad_dim=3,
        num_relations=6,
        rgat_layers=2,
        rgat_heads=4,
        cross_attn_heads=4,
        regression_hidden_dim=128,
        use_mock_backbone=True,
    )
    
    ckpt_path = "checkpoints/nssg_dimnet/best_model_corrected.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    
    configs_to_test = [
        ("Old: threshold=0.0", {"strategy": "threshold", "threshold": 0.0}),
        ("Old: threshold=-0.5", {"strategy": "threshold", "threshold": -0.5}),
        ("Old: threshold=-1.0", {"strategy": "threshold", "threshold": -1.0}),
        ("New: ranked top_k=3", {"strategy": "ranked", "top_k": 3}),
        ("New: ranked top_k=5 (default)", {"strategy": "ranked", "top_k": 5}),
        ("New: ranked top_k=7", {"strategy": "ranked", "top_k": 7}),
        ("New: ranked top_k=5 + floor=-1.0", {"strategy": "ranked", "top_k": 5, "score_floor": -1.0}),
    ]
    
    results = {}
    print(f"{'Configuration':32s} | {'Asp F1':7s} | {'Op F1':7s} | {'V-r':6s} | {'A-r':6s} | {'Mean-r':6s} | {'PredAsp':7s} | {'PredOp':7s}")
    print("-" * 90)
    for name, cfg in configs_to_test:
        res = eval_val_with_config(model, val_loader, **cfg)
        results[name] = res
        print(f"{name:32s} | {res['aspect_f1']:7.4f} | {res['opinion_f1']:7.4f} | {res['pearson_v']:6.3f} | {res['pearson_a']:6.3f} | {res['pearson_mean']:6.3f} | {res['total_pred_aspects']:7d} | {res['total_pred_opinions']:7d}")
        
    os.makedirs("reports/corrections", exist_ok=True)
    with open("reports/corrections/step8_threshold_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Step 8 threshold analysis to reports/corrections/step8_threshold_analysis.json")

if __name__ == "__main__":
    main()
