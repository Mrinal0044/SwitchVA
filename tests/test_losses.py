"""Tests for CCC Loss, Joint Loss, and Evaluation Metrics."""

import torch
import numpy as np
import pytest
from src.losses.ccc_loss import CCCLoss
from src.losses.joint_loss import JointLoss
from src.evaluation.metrics import evaluate_predictions, compute_ccc


def test_ccc_loss_identical():
    ccc_fn = CCCLoss()
    y_true = torch.tensor([0.2, 0.4, 0.6, 0.8])
    y_pred = torch.tensor([0.2, 0.4, 0.6, 0.8])

    loss = ccc_fn(y_pred, y_true)
    # Perfect agreement => CCC = 1.0 => Loss = 0.0
    assert pytest.approx(loss.item(), abs=1e-4) == 0.0


def test_ccc_loss_inverse():
    ccc_fn = CCCLoss()
    y_true = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    y_pred = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])

    loss = ccc_fn(y_pred, y_true)
    # Inverse agreement => CCC = -1.0 => Loss = 2.0
    assert pytest.approx(loss.item(), abs=1e-4) == 2.0


def test_joint_loss():
    joint_loss_fn = JointLoss(
        ccc_weight_v=1.0,
        ccc_weight_a=1.0,
        smooth_l1_weight_v=0.5,
        smooth_l1_weight_a=0.5,
    )

    pred_v = torch.tensor([0.3, 0.5, 0.7])
    pred_a = torch.tensor([0.4, 0.6, 0.8])
    true_v = torch.tensor([0.3, 0.5, 0.7])
    true_a = torch.tensor([0.4, 0.6, 0.8])

    losses = joint_loss_fn(pred_v, pred_a, true_v, true_a)
    assert "total_loss" in losses
    assert "loss_ccc_v" in losses
    assert "loss_ccc_a" in losses
    assert "loss_smooth_l1_v" in losses
    assert "loss_smooth_l1_a" in losses
    assert pytest.approx(losses["total_loss"].item(), abs=1e-4) == 0.0


def test_evaluation_metrics():
    y_true_v = [0.1, 0.4, 0.7]
    y_pred_v = [0.1, 0.4, 0.7]
    y_true_a = [0.2, 0.5, 0.8]
    y_pred_a = [0.2, 0.5, 0.8]

    metrics = evaluate_predictions(y_true_v, y_pred_v, y_true_a, y_pred_a)
    assert pytest.approx(metrics["ccc_v"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["ccc_a"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["rmse_v"], abs=1e-4) == 0.0
