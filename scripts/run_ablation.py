#!/usr/bin/env python3
"""Ablation Study Runner for NSSG-DimNet.

Runs specified ablation studies from configs/ablations.yaml:
- wo_switch_embedding (Ablation A: w/o Switch Distance & Switch Embedding)
- wo_sp_gsa           (Ablation B: w/o SP-GSA)
- wo_hnsg_rgat        (Ablation C: w/o H-NSG & RGAT)
- wo_nrc_vad          (Ablation D: w/o NRC-VAD Affective Priors)
- wo_cross_attention  (Ablation E: w/o Mutual Cross-Attention)
- wo_ccc_loss         (Ablation F: w/o CCC Loss)

Outputs results to reports/ablation_results.json and reports/ablation_results.md.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
import yaml

from src.evaluation.metrics import compute_model_deltas, evaluate_predictions
from src.losses.va_loss import JointVALoss
from src.models.baseline import SwitchUnawareBaseline
from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.training.trainer import NSSGDimNetTrainer, build_optimizer, build_warmup_scheduler
from src.utils.checkpoint import CheckpointManager
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("ablation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run NSSG-DimNet Ablation Studies")
    parser.add_argument("--config", type=str, default="configs/ablations.yaml")
    parser.add_argument("--study", type=str, default="all",
                        choices=["all", "wo_switch_embedding", "wo_sp_gsa", "wo_hnsg_rgat", "wo_nrc_vad", "wo_cross_attention", "wo_ccc_loss"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="reports/ablations")
    parser.add_argument("--use-mock-backbone", action="store_true", default=True)
    return parser.parse_args()


def build_ablation_model_and_loss(study_name: str, hidden_dim: int, use_mock: bool):
    """Constructs model and loss tailored to the specific ablation study."""
    loss_fn = JointVALoss()

    if study_name == "wo_hnsg_rgat":
        # Removes Branch 2: uses Baseline architecture (Branch 1 spans -> regression)
        model = SwitchUnawareBaseline(
            hidden_dim=hidden_dim,
            span_mlp_dim=max(32, hidden_dim // 2),
            regression_hidden_dim=hidden_dim,
            use_mock_backbone=use_mock,
        )
    elif study_name == "wo_ccc_loss":
        model = NSSGDimNet(
            hidden_dim=hidden_dim,
            num_languages=4,
            max_signed_distance=64,
            num_spgsa_heads=4,
            span_mlp_dim=max(32, hidden_dim // 2),
            nrc_vad_dim=3,
            num_relations=6,
            rgat_layers=2,
            rgat_heads=4,
            cross_attn_heads=4,
            regression_hidden_dim=hidden_dim,
            use_mock_backbone=use_mock,
        )
        loss_fn = JointVALoss(ccc_weight_v=0.0, ccc_weight_a=0.0, smooth_l1_weight_v=1.0, smooth_l1_weight_a=1.0)
    else:
        model = NSSGDimNet(
            hidden_dim=hidden_dim,
            num_languages=4,
            max_signed_distance=64,
            num_spgsa_heads=4,
            span_mlp_dim=max(32, hidden_dim // 2),
            nrc_vad_dim=3,
            num_relations=6,
            rgat_layers=2,
            rgat_heads=4,
            cross_attn_heads=4,
            regression_hidden_dim=hidden_dim,
            use_mock_backbone=use_mock,
        )

    return model, loss_fn


def run_single_ablation(
    study_name: str,
    study_info: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    logger.info("Running Ablation: %s - %s", study_name, study_info.get("description", ""))
    set_seed(args.seed)

    model, loss_fn = build_ablation_model_and_loss(study_name, args.hidden_dim, args.use_mock_backbone)
    model.to(device)

    optimizer = build_optimizer(model, learning_rate=args.lr, backbone_learning_rate=args.backbone_lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = build_warmup_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

    ckpt_dir = os.path.join(args.output_dir, study_name)
    ckpt_mgr = CheckpointManager(checkpoint_dir=ckpt_dir, monitor_metric="val_combined_pearson", mode="max")

    trainer = NSSGDimNetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=ckpt_mgr,
        device=device,
        early_stopping_patience=5,
    )

    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # Evaluate on test set
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt_mgr.load(best_path, model)

    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test results for %s: %s", study_name, test_metrics)

    return {
        "study": study_name,
        "description": study_info.get("description", ""),
        "test_metrics": test_metrics,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device(args.device)

    # Load ablation configs
    if os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
            studies_dict = full_cfg.get("ablation_studies", {})
    else:
        studies_dict = {
            "wo_switch_embedding": {"description": "Removes switch embedding and signed distance."},
            "wo_sp_gsa": {"description": "Replaces SP-GSA with standard self-attention."},
            "wo_hnsg_rgat": {"description": "Removes Branch 2 (H-NSG + RGAT)."},
            "wo_nrc_vad": {"description": "Removes NRC-VAD affective priors from graph."},
            "wo_cross_attention": {"description": "Replaces cross-attention with concatenation."},
            "wo_ccc_loss": {"description": "Ablates CCC loss; trained with Smooth L1 only."},
        }

    # Data loaders
    train_ds = DimABSATensorDataset("data/splits/train.jsonl")
    val_ds = DimABSATensorDataset("data/splits/validation.jsonl")
    test_ds = DimABSATensorDataset("data/splits/test.jsonl")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=dimabsa_tensor_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dimabsa_tensor_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dimabsa_tensor_collate_fn)

    target_studies = list(studies_dict.keys()) if args.study == "all" else [args.study]
    all_results = {}

    for s_name in target_studies:
        s_info = studies_dict.get(s_name, {})
        res = run_single_ablation(s_name, s_info, train_loader, val_loader, test_loader, args, device)
        all_results[s_name] = res

    # Save results JSON
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save results Markdown table
    md_path = os.path.join(args.output_dir, "ablation_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NSSG-DimNet Ablation Study Results\n\n")
        f.write("| Ablation Study | Description | Valence $r$ | Valence MAE | Arousal $r$ | Arousal MAE | Combined $r$ |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for s_name, res in all_results.items():
            m = res["test_metrics"]
            desc = res["description"]
            f.write(f"| `{s_name}` | {desc} | {m.get('pearson_v', 0.0):.4f} | {m.get('mae_v', 0.0):.4f} | {m.get('pearson_a', 0.0):.4f} | {m.get('mae_a', 0.0):.4f} | {m.get('pearson_mean', 0.0):.4f} |\n")

    logger.info("Ablation study complete. Results saved to %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
