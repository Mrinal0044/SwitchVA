"""Concordance Correlation Coefficient (CCC) Loss for continuous Valence and Arousal regression."""

import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    """Concordance Correlation Coefficient (CCC) Loss.

    Measures the agreement between two continuous variables (e.g., predicted and ground truth
    Valence/Arousal values in [0, 1]).
    CCC = (2 * Cov(y, y_hat)) / (Var(y) + Var(y_hat) + (mean(y) - mean(y_hat))^2)
    Loss = 1 - CCC.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted continuous values of shape (N,) or (N, 1).
            target: Ground-truth continuous values of shape (N,) or (N, 1).

        Returns:
            Scalar tensor representing 1.0 - CCC.
        """
        pred = pred.view(-1)
        target = target.view(-1)

        if pred.numel() < 2:
            # For degenerate batch of size 1, return squared error
            return torch.mean((pred - target) ** 2)

        mean_pred = torch.mean(pred)
        mean_target = torch.mean(target)

        var_pred = torch.var(pred, unbiased=False)
        var_target = torch.var(target, unbiased=False)

        covariance = torch.mean((pred - mean_pred) * (target - mean_target))

        numerator = 2.0 * covariance
        denominator = var_pred + var_target + (mean_pred - mean_target) ** 2 + self.eps

        ccc = numerator / denominator
        # CCC is bounded in [-1, 1], so Loss = 1 - CCC is bounded in [0, 2]
        return 1.0 - ccc
