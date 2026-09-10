#!/usr/bin/env python3
"""Training script for NSSG-DimNet and Switch-Unaware Baseline.

Usage:
    python3 scripts/train.py --model nssg_dimnet --epochs 25 --batch-size 16
    python3 scripts/train.py --model baseline --epochs 25 --batch-size 16
"""

import argparse
import json
import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.models.baseline import SwitchUnawareBaseline
from src.models.nssg_dimnet import NSSGDimNet
from src.training.data_utils import DimABSATensorDataset, dimabsa_tensor_collate_fn
from src.training.trainer import NSSGDimNetTrainer, build_optimizer, build_warmup_scheduler
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.device import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DimABSA Models")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model", type=str, default="nssg_dimnet", choices=["nssg_dimnet", "baseline"])
    parser.add_argument("--train-path", type=str, default="data/splits/train.jsonl")
    parser.add_argument("--val-path", type=str, default="data/splits/validation.jsonl")
    parser.add_argument("--test-path", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot-dir", type=str, default="reports/training")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--use-mock-backbone", action="store_true", default=True,
                        help="Use lightweight backbone for fast/offline execution")
    parser.add_argument("--best-filename", type=str, default="best_model.pt")
    parser.add_argument("--last-filename", type=str, default="last_model.pt")
    parser.add_argument("--history-filename", type=str, default="training_history.json")
    parser.add_argument("--lambda-span", type=float, default=0.5)
    parser.add_argument("--aspect-pos-weight", type=float, default=50.0)
    parser.add_argument("--opinion-pos-weight", type=float, default=50.0)
    return parser.parse_args()


def plot_training_curves(history: Dict[str, list], plot_dir: str, model_name: str):
    """Plot and save training/validation loss, Pearson, and MAE curves."""
    os.makedirs(plot_dir, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    if not epochs:
        return

    # 1. Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.get("train_loss", []), "b-o", label="Train Loss")
    plt.plot(epochs, history.get("val_loss", []), "r--s", label="Val Loss")
    plt.title(f"{model_name.upper()} - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"loss_curve_{model_name}.png"), dpi=150)
    plt.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 2. Valence Pearson
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.get("val_pearson_v", []), "g-^", label="Val Valence Pearson (r)")
    plt.title(f"{model_name.upper()} - Valence Pearson Correlation")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"valence_pearson_{model_name}.png"), dpi=150)
    plt.savefig(os.path.join(plot_dir, "valence_pearson.png"), dpi=150)
    plt.close()

    # 3. Arousal Pearson
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.get("val_pearson_a", []), "m-d", label="Val Arousal Pearson (r)")
    plt.title(f"{model_name.upper()} - Arousal Pearson Correlation")
    plt.xlabel("Epoch")
    plt.ylabel("Pearson r")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"arousal_pearson_{model_name}.png"), dpi=150)
    plt.savefig(os.path.join(plot_dir, "arousal_pearson.png"), dpi=150)
    plt.close()

    # 4. Valence MAE
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.get("val_mae_v", []), "c-x", label="Val Valence MAE")
    plt.title(f"{model_name.upper()} - Valence Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"valence_mae_{model_name}.png"), dpi=150)
    plt.savefig(os.path.join(plot_dir, "valence_mae.png"), dpi=150)
    plt.close()

    # 5. Arousal MAE
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.get("val_mae_a", []), "y-v", label="Val Arousal MAE")
    plt.title(f"{model_name.upper()} - Arousal Mean Absolute Error")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"arousal_mae_{model_name}.png"), dpi=150)
    plt.savefig(os.path.join(plot_dir, "arousal_mae.png"), dpi=150)
    plt.close()


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        if hasattr(cfg, "experiment"):
            if hasattr(cfg.experiment, "output_dir") and not args.output_dir:
                args.output_dir = cfg.experiment.output_dir
            if hasattr(cfg.experiment, "best_checkpoint_name"):
                args.best_filename = cfg.experiment.best_checkpoint_name
            if hasattr(cfg.experiment, "last_checkpoint_name"):
                args.last_filename = cfg.experiment.last_checkpoint_name
            if hasattr(cfg.experiment, "history_filename"):
                args.history_filename = cfg.experiment.history_filename
        if hasattr(cfg, "loss"):
            if hasattr(cfg.loss, "lambda_span"):
                args.lambda_span = cfg.loss.lambda_span
            if hasattr(cfg.loss, "aspect_pos_weight"):
                args.aspect_pos_weight = cfg.loss.aspect_pos_weight
            if hasattr(cfg.loss, "opinion_pos_weight"):
                args.opinion_pos_weight = cfg.loss.opinion_pos_weight
        if hasattr(cfg, "training"):
            if hasattr(cfg.training, "num_epochs"):
                args.epochs = cfg.training.num_epochs
            if hasattr(cfg.training, "learning_rate"):
                args.lr = cfg.training.learning_rate

    set_seed(args.seed)

    output_dir = args.output_dir or f"checkpoints/{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # 1. Datasets and DataLoaders
    logger.info("Loading dataset splits...")
    train_ds = DimABSATensorDataset(args.train_path)
    val_ds = DimABSATensorDataset(args.val_path)
    test_ds = DimABSATensorDataset(args.test_path)

    logger.info("Dataset samples - Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dimabsa_tensor_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dimabsa_tensor_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dimabsa_tensor_collate_fn,
    )

    # 2. Build Model
    if args.model == "baseline":
        logger.info("Initializing SwitchUnawareBaseline...")
        model = SwitchUnawareBaseline(
            hidden_dim=args.hidden_dim,
            span_mlp_dim=max(32, args.hidden_dim // 2),
            regression_hidden_dim=args.hidden_dim,
            use_mock_backbone=args.use_mock_backbone,
        )
    else:
        logger.info("Initializing NSSGDimNet...")
        model = NSSGDimNet(
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
        )

    # 3. Optimizer and Scheduler
    optimizer = build_optimizer(
        model,
        learning_rate=args.lr,
        backbone_learning_rate=args.backbone_lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs
    scheduler = build_warmup_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

    # 4. Checkpoint Manager & Trainer
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=output_dir,
        monitor_metric="val_combined_pearson",
        mode="max",
        best_filename=args.best_filename,
        last_filename=args.last_filename,
    )

    trainer = NSSGDimNetTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=ckpt_mgr,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        lambda_span=args.lambda_span,
        aspect_pos_weight=args.aspect_pos_weight,
        opinion_pos_weight=args.opinion_pos_weight,
    )

    # 5. Run Training
    logger.info("Starting training for %d epochs...", args.epochs)
    train_results = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
    )

    # 6. Save Training History & Curves
    history = train_results.get("history", {})
    history_path = os.path.join(output_dir, args.history_filename)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    report_history_path = os.path.join(args.plot_dir, f"training_history_{args.model}.json")
    with open(report_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, args.plot_dir, args.model)
    logger.info("Saved training curves and history to %s and %s", output_dir, args.plot_dir)

    # 7. Final Test Evaluation using best model
    best_ckpt_path = os.path.join(output_dir, args.best_filename)
    if os.path.exists(best_ckpt_path):
        logger.info("Loading best model from %s for test evaluation...", best_ckpt_path)
        ckpt_mgr.load(best_ckpt_path, model)

    test_metrics = trainer.evaluate(test_loader)
    logger.info("Test Evaluation Results for %s: %s", args.model, test_metrics)

    test_results_path = os.path.join(output_dir, "test_results.json")
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    report_test_path = f"reports/test_results_{args.model}.json"
    with open(report_test_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
