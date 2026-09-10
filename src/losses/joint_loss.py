"""Joint Loss for NSSG-DimNet combining CCC loss, Smooth L1 loss, and auxiliary span loss."""

from typing import Dict, Optional
import torch
import torch.nn as nn
from .ccc_loss import CCCLoss


class JointLoss(nn.Module):
    """Joint Loss combining:
    1. Concordance Correlation Coefficient (CCC) loss on Valence and Arousal
    2. Smooth L1 loss on Valence and Arousal
    3. Optional auxiliary span classification loss for Biaffine span extractor
    """

    def __init__(
        self,
        ccc_weight_v: float = 1.0,
        ccc_weight_a: float = 1.0,
        smooth_l1_weight_v: float = 0.5,
        smooth_l1_weight_a: float = 0.5,
        span_loss_weight: float = 0.3,
        smooth_l1_beta: float = 0.05,
    ):
        super().__init__()
        self.ccc_weight_v = ccc_weight_v
        self.ccc_weight_a = ccc_weight_a
        self.smooth_l1_weight_v = smooth_l1_weight_v
        self.smooth_l1_weight_a = smooth_l1_weight_a
        self.span_loss_weight = span_loss_weight

        self.ccc_loss = CCCLoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta)
        self.span_ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        pred_valence: torch.Tensor,
        pred_arousal: torch.Tensor,
        target_valence: torch.Tensor,
        target_arousal: torch.Tensor,
        span_logits: Optional[torch.Tensor] = None,
        span_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components and weighted sum.

        Args:
            pred_valence: Predicted valence tensor of shape (N,) or (N, 1).
            pred_arousal: Predicted arousal tensor of shape (N,) or (N, 1).
            target_valence: Ground-truth valence tensor of shape (N,) or (N, 1).
            target_arousal: Ground-truth arousal tensor of shape (N,) or (N, 1).
            span_logits: Optional span scoring logits of shape (N, L, L, num_labels).
            span_targets: Optional span ground-truth labels of shape (N, L, L).

        Returns:
            Dictionary containing 'total_loss', 'loss_ccc_v', 'loss_ccc_a',
            'loss_smooth_l1_v', 'loss_smooth_l1_a', and optionally 'loss_span'.
        """
        loss_ccc_v = self.ccc_loss(pred_valence, target_valence)
        loss_ccc_a = self.ccc_loss(pred_arousal, target_arousal)
        loss_l1_v = self.smooth_l1(pred_valence.view(-1), target_valence.view(-1))
        loss_l1_a = self.smooth_l1(pred_arousal.view(-1), target_arousal.view(-1))

        total_loss = (
            self.ccc_weight_v * loss_ccc_v
            + self.ccc_weight_a * loss_ccc_a
            + self.smooth_l1_weight_v * loss_l1_v
            + self.smooth_l1_weight_a * loss_l1_a
        )

        loss_dict = {
            "total_loss": total_loss,
            "loss_ccc_v": loss_ccc_v,
            "loss_ccc_a": loss_ccc_a,
            "loss_smooth_l1_v": loss_l1_v,
            "loss_smooth_l1_a": loss_l1_a,
        }

        if span_logits is not None and span_targets is not None and self.span_loss_weight > 0.0:
            # Reshape for CrossEntropyLoss: (N * L * L, num_labels) vs (N * L * L)
            num_classes = span_logits.size(-1)
            flat_logits = span_logits.view(-1, num_classes)
            flat_targets = span_targets.view(-1)
            loss_span = self.span_ce_loss(flat_logits, flat_targets)
            loss_dict["loss_span"] = loss_span
            loss_dict["total_loss"] = total_loss + self.span_loss_weight * loss_span

        return loss_dict
