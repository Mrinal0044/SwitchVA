"""Concordance Correlation Coefficient (CCC) and Joint VA Loss module for Phase 5."""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CCCLoss(nn.Module):
    """Numerically stable Concordance Correlation Coefficient (CCC) Loss.

    Computes:
        CCC = (2 * Cov(y, y_hat)) / (Var(y) + Var(y_hat) + (mean(y) - mean(y_hat))^2 + eps)
        Loss_CCC = 1.0 - CCC

    Safeguards:
    - If N < 2 or variance is 0, gracefully falls back to MSE/Smooth L1 to prevent division by zero or NaN.
    - Epsilon prevents zero-denominator instability.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted continuous values (N,) or (N, 1) or arbitrary shape.
            target: Ground-truth continuous values matching pred shape.

        Returns:
            Scalar loss in [0, 2].
        """
        pred = pred.reshape(-1)
        target = target.reshape(-1)

        assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

        # Degenerate case: fewer than 2 elements
        if pred.numel() < 2:
            return F.smooth_l1_loss(pred, target)

        mean_pred = torch.mean(pred)
        mean_target = torch.mean(target)

        var_pred = torch.var(pred, unbiased=False)
        var_target = torch.var(target, unbiased=False)

        # Zero variance safeguard (e.g., constant predictions or constant targets)
        if var_pred < self.eps and var_target < self.eps:
            return (mean_pred - mean_target) ** 2

        covariance = torch.mean((pred - mean_pred) * (target - mean_target))
        numerator = 2.0 * covariance
        denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + self.eps

        ccc = numerator / denominator
        # Clamp CCC to [-1.0, 1.0] for absolute numerical guarantee
        ccc = torch.clamp(ccc, -1.0, 1.0)
        return 1.0 - ccc


class JointVALoss(nn.Module):
    """Joint Continuous Valence-Arousal Loss.

    Combines:
    1. Valence CCC loss
    2. Arousal CCC loss
    3. Valence Smooth L1 loss
    4. Arousal Smooth L1 loss
    5. Optional auxiliary span loss
    """

    def __init__(
        self,
        ccc_weight_v: float = 1.0,
        ccc_weight_a: float = 1.0,
        smooth_l1_weight_v: float = 0.5,
        smooth_l1_weight_a: float = 0.5,
        span_loss_weight: float = 0.3,
        smooth_l1_beta: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.ccc_weight_v = ccc_weight_v
        self.ccc_weight_a = ccc_weight_a
        self.smooth_l1_weight_v = smooth_l1_weight_v
        self.smooth_l1_weight_a = smooth_l1_weight_a
        self.span_loss_weight = span_loss_weight

        self.ccc_loss = CCCLoss(eps=eps)
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.span_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        pred_valence: torch.Tensor,
        pred_arousal: torch.Tensor,
        target_valence: torch.Tensor,
        target_arousal: torch.Tensor,
        span_logits: Optional[torch.Tensor] = None,
        span_targets: Optional[torch.Tensor] = None,
        va_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss terms and return total loss.

        Args:
            pred_valence: Predicted valence scores in [0, 1]
            pred_arousal: Predicted arousal scores in [0, 1]
            target_valence: Ground-truth valence scores in [0, 1]
            target_arousal: Ground-truth arousal scores in [0, 1]
            span_logits: Optional span grid logits from Branch 1
            span_targets: Optional gold span targets
            va_mask: Optional boolean mask for valid VA target supervision

        Returns:
            Dictionary containing individual loss components and 'total_loss'.
        """
        p_v = pred_valence.reshape(-1)
        p_a = pred_arousal.reshape(-1)
        t_v = target_valence.reshape(-1)
        t_a = target_arousal.reshape(-1)

        if va_mask is not None:
            mask = va_mask.reshape(-1).bool()
            p_v = p_v[mask]
            p_a = p_a[mask]
            t_v = t_v[mask]
            t_a = t_a[mask]

        loss_ccc_v = self.ccc_loss(p_v, t_v)
        loss_ccc_a = self.ccc_loss(p_a, t_a)
        loss_smooth_v = self.smooth_l1(p_v, t_v)
        loss_smooth_a = self.smooth_l1(p_a, t_a)

        total_loss = (
            self.ccc_weight_v * loss_ccc_v
            + self.ccc_weight_a * loss_ccc_a
            + self.smooth_l1_weight_v * loss_smooth_v
            + self.smooth_l1_weight_a * loss_smooth_a
        )

        loss_dict: Dict[str, torch.Tensor] = {
            "total_loss": total_loss,
            "loss_ccc_v": loss_ccc_v,
            "loss_ccc_a": loss_ccc_a,
            "loss_smooth_l1_v": loss_smooth_v,
            "loss_smooth_l1_a": loss_smooth_a,
            "loss_ccc_total": loss_ccc_v + loss_ccc_a,
            "loss_smooth_total": loss_smooth_v + loss_smooth_a,
        }

        if span_logits is not None and span_targets is not None and self.span_loss_weight > 0.0:
            num_classes = span_logits.size(-1)
            flat_logits = span_logits.view(-1, num_classes)
            flat_targets = span_targets.view(-1)
            loss_span = self.span_loss_fn(flat_logits, flat_targets)
            loss_dict["loss_span"] = loss_span
            loss_dict["total_loss"] = total_loss + self.span_loss_weight * loss_span

        return loss_dict
