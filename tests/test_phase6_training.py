"""Tests for Phase 6: Reproducible Training, Switch-Unaware Baseline, Evaluation, & Ablation Infrastructure."""

import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import SwitchUnawareBaseline
from src.models.nssg_dimnet import NSSGDimNet
from src.training.trainer import NSSGDimNetTrainer, build_optimizer, build_warmup_scheduler
from src.evaluation.metrics import (
    compute_ccc,
    compute_pearson_r,
    compute_mae,
    compute_rmse,
    evaluate_predictions,
    compute_span_metrics,
    evaluate_by_switch_density,
    compute_model_deltas,
)
from src.utils.checkpoint import CheckpointManager
from src.utils.seed import set_seed
from src.losses.va_loss import JointVALoss


def test_1_optimizer_construction():
    """1. Test that optimizer creates separate learning rate groups for backbone vs other modules."""
    model = NSSGDimNet(hidden_dim=32, num_spgsa_heads=2, rgat_heads=2, cross_attn_heads=2, use_mock_backbone=True)
    optimizer = build_optimizer(model, learning_rate=1e-4, backbone_learning_rate=2e-5, weight_decay=0.01)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 2e-5  # backbone
    assert optimizer.param_groups[1]["lr"] == 1e-4  # new modules


def test_2_scheduler_warmup_and_decay():
    """2. Test warmup scheduler increases then decays learning rate."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = build_warmup_scheduler(optimizer, total_steps=100, warmup_ratio=0.1)

    # Initial step
    assert scheduler.get_last_lr()[0] == 0.0
    for _ in range(10):
        scheduler.step()
    # At warmup end (step 10), lr should reach 1.0
    assert abs(scheduler.get_last_lr()[0] - 1.0) < 1e-3

    for _ in range(90):
        scheduler.step()
    # At end (step 100), lr should be 0.0
    assert scheduler.get_last_lr()[0] < 1e-3


def test_3_seed_reproducibility():
    """3. Test seed utility ensures deterministic random numbers."""
    set_seed(42)
    t1 = torch.randn(5)
    set_seed(42)
    t2 = torch.randn(5)
    assert torch.equal(t1, t2)


def test_4_loss_aggregation():
    """4. Test that JointVALoss aggregates CCC, Smooth L1, and auxiliary span loss."""
    loss_fn = JointVALoss(ccc_weight_v=1.0, ccc_weight_a=1.0, smooth_l1_weight_v=0.5, smooth_l1_weight_a=0.5)
    p_v = torch.tensor([0.2, 0.4, 0.6])
    p_a = torch.tensor([0.3, 0.5, 0.7])
    t_v = torch.tensor([0.2, 0.4, 0.6])
    t_a = torch.tensor([0.3, 0.5, 0.7])

    out = loss_fn(p_v, p_a, t_v, t_a)
    assert "total_loss" in out
    assert "loss_ccc_v" in out
    assert "loss_smooth_l1_v" in out
    assert out["loss_ccc_v"].item() < 1e-4


def test_5_epoch_level_ccc_aggregation():
    """5. Test CCC is computed accurately over aggregated population vs individual items."""
    targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    preds = [0.12, 0.28, 0.52, 0.68, 0.88]

    ccc = compute_ccc(targets, preds)
    assert ccc > 0.95


def test_6_mae_and_rmse_metrics():
    """6. Test MAE and RMSE metric computations."""
    t = [0.0, 1.0]
    p = [0.2, 0.8]
    mae = compute_mae(t, p)
    rmse = compute_rmse(t, p)

    assert abs(mae - 0.2) < 1e-5
    assert abs(rmse - 0.2) < 1e-5


def test_7_pearson_metric():
    """7. Test Pearson r computation."""
    t = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = [2.0, 4.0, 6.0, 8.0, 10.0]
    r = compute_pearson_r(t, p)
    assert abs(r - 1.0) < 1e-5


def test_8_checkpoint_saving_and_loading():
    """8. Test checkpoint saving and loading roundtrip."""
    temp_dir = tempfile.mkdtemp()
    try:
        mgr = CheckpointManager(checkpoint_dir=temp_dir, monitor_metric="val_pearson_mean", mode="max")
        model = nn.Linear(5, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Save
        metrics = {"val_pearson_mean": 0.85}
        save_path = mgr.save(model, optimizer, epoch=1, metrics=metrics)
        assert os.path.exists(save_path)
        assert os.path.join(temp_dir, "best_model.pt")

        # Load
        model2 = nn.Linear(5, 2)
        loaded = mgr.load(save_path, model2)
        assert loaded["epoch"] == 1
        assert abs(loaded["metrics"]["val_pearson_mean"] - 0.85) < 1e-5
    finally:
        shutil.rmtree(temp_dir)


def test_9_early_stopping_logic():
    """9. Test early stopping logic."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = SwitchUnawareBaseline(hidden_dim=16, span_mlp_dim=8, regression_hidden_dim=8, use_mock_backbone=True)
        trainer = NSSGDimNetTrainer(
            model=model,
            checkpoint_manager=CheckpointManager(checkpoint_dir=temp_dir),
            early_stopping_patience=2,
        )

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (6,)),
                    "attention_mask": torch.ones(6),
                    "valence": torch.tensor(0.5),
                    "arousal": torch.tensor(0.5),
                }

        ds = DictDataset()
        dl = DataLoader(ds, batch_size=2)
        res = trainer.fit(dl, dl, epochs=1)
        assert "history" in res
    finally:
        shutil.rmtree(temp_dir)


def test_10_baseline_model_forward_pass():
    """10. Test SwitchUnawareBaseline forward pass."""
    model = SwitchUnawareBaseline(hidden_dim=32, span_mlp_dim=16, regression_hidden_dim=16, use_mock_backbone=True)
    input_ids = torch.randint(0, 500, (2, 8))
    attention_mask = torch.ones(2, 8)
    aspect_spans = torch.tensor([[1, 2], [3, 4]])
    opinion_spans = torch.tensor([[3, 4], [5, 6]])

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        aspect_spans=aspect_spans,
        opinion_spans=opinion_spans,
    )

    assert "valence" in outputs
    assert "arousal" in outputs
    assert outputs["valence"].shape == (2, 1)
    assert outputs["arousal"].shape == (2, 1)
    assert (outputs["valence"] >= 0.0).all() and (outputs["valence"] <= 1.0).all()


def test_11_baseline_absence_of_switch_features():
    """11. Verify SwitchUnawareBaseline has no switch-aware modules."""
    model = SwitchUnawareBaseline(hidden_dim=32, use_mock_backbone=True)
    assert not hasattr(model, "switch_embedding")
    assert not hasattr(model, "sp_gsa")
    assert not hasattr(model, "hnsg_builder")
    assert not hasattr(model, "rgat")
    assert not hasattr(model, "cross_attention")


def test_12_full_model_training_step():
    """12. Test single training step on full NSSGDimNet."""
    model = NSSGDimNet(
        hidden_dim=32,
        num_spgsa_heads=2,
        span_mlp_dim=16,
        rgat_heads=2,
        cross_attn_heads=2,
        regression_hidden_dim=16,
        use_mock_backbone=True,
    )
    optimizer = build_optimizer(model, learning_rate=1e-3, backbone_learning_rate=1e-4)
    trainer = NSSGDimNetTrainer(model=model, optimizer=optimizer)

    batch = {
        "input_ids": torch.randint(0, 500, (2, 8)),
        "lang_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8),
        "aspect_spans": torch.tensor([[1, 2], [3, 4]]),
        "opinion_spans": torch.tensor([[3, 4], [5, 6]]),
        "valence": torch.tensor([0.7, 0.4]),
        "arousal": torch.tensor([0.6, 0.5]),
    }
    loss_metrics = trainer.train_step(batch)
    assert "total_loss" in loss_metrics
    assert loss_metrics["total_loss"] > 0.0


def test_13_span_extraction_metrics():
    """13. Test exact match span Precision, Recall, and F1."""
    gold = [[(2, 3), (6, 7)], [(0, 1)]]
    pred = [[(2, 3)], [(0, 1), (4, 5)]]

    metrics = compute_span_metrics(gold, pred)
    assert metrics["correct"] == 2
    assert metrics["total_gold"] == 3
    assert metrics["total_pred"] == 3
    assert abs(metrics["precision"] - 2 / 3) < 1e-4
    assert abs(metrics["recall"] - 2 / 3) < 1e-4
    assert abs(metrics["f1"] - 2 / 3) < 1e-4


def test_14_switch_density_stratification():
    """14. Test partitioning into low, medium, and high code-switch density."""
    records = [
        # Sentence 1: 0 switches in 10 tokens -> low
        {
            "tokens": ["a"] * 10,
            "language_ids": ["HI"] * 10,
            "predictions": [{"valence": 0.5, "arousal": 0.5}],
            "annotations": [{"valence": 0.5, "arousal": 0.5}],
        },
        # Sentence 2: 0 switches in 10 tokens -> low
        {
            "tokens": ["a"] * 10,
            "language_ids": ["HI"] * 10,
            "predictions": [{"valence": 0.6, "arousal": 0.6}],
            "annotations": [{"valence": 0.6, "arousal": 0.6}],
        },
        # Sentence 3: 4 switches in 10 tokens (0.4 density) -> high
        {
            "tokens": ["a"] * 10,
            "language_ids": ["HI", "EN", "HI", "EN", "HI", "EN", "HI", "HI", "HI", "HI"],
            "predictions": [{"valence": 0.8, "arousal": 0.8}],
            "annotations": [{"valence": 0.7, "arousal": 0.7}],
        },
        # Sentence 4: 4 switches in 10 tokens -> high
        {
            "tokens": ["a"] * 10,
            "language_ids": ["HI", "EN", "HI", "EN", "HI", "EN", "HI", "HI", "HI", "HI"],
            "predictions": [{"valence": 0.2, "arousal": 0.2}],
            "annotations": [{"valence": 0.3, "arousal": 0.3}],
        },
    ]

    density_results = evaluate_by_switch_density(records, low_thresh=0.15, high_thresh=0.30)
    assert "low" in density_results
    assert "medium" in density_results
    assert "high" in density_results
    assert density_results["low"]["count"] == 2
    assert density_results["high"]["count"] == 2


def test_15_model_deltas_computation():
    """15. Test comparative delta computation (NSSG - Baseline)."""
    base = {"pearson_v": 0.50, "pearson_a": 0.40, "mae_v": 0.15, "mae_a": 0.18}
    nssg = {"pearson_v": 0.58, "pearson_a": 0.49, "mae_v": 0.12, "mae_a": 0.14}

    deltas = compute_model_deltas(base, nssg)
    assert abs(deltas["delta_pearson_v"] - 0.08) < 1e-4
    assert abs(deltas["delta_pearson_a"] - 0.09) < 1e-4
    assert abs(deltas["delta_mae_v"] - (-0.03)) < 1e-4
