"""Independent Valence and Arousal Regression Heads (Linear -> GELU -> Linear -> Sigmoid in [0, 1])."""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


class ValenceRegressionHead(nn.Module):
    """Independent Valence Regression Head.

    Architecture:
    Linear(input_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, 1) -> Sigmoid -> continuous [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 2304,  # 3 * 768
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, 1),
            nn.Sigmoid(),  # Bounds output strictly in continuous [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, input_dim) or (batch_size, num_pairs, input_dim) fused representation.

        Returns:
            (batch_size, 1) or (batch_size, num_pairs, 1) continuous Valence in [0, 1].
        """
        return self.mlp(z)


class ArousalRegressionHead(nn.Module):
    """Independent Arousal Regression Head with separate parameter set.

    Architecture:
    Linear(input_dim, hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim, 1) -> Sigmoid -> continuous [0, 1]
    """

    def __init__(
        self,
        input_dim: int = 2304,  # 3 * 768
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim >= 4 else hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2 if hidden_dim >= 4 else hidden_dim, 1),
            nn.Sigmoid(),  # Bounds output strictly in continuous [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, input_dim) or (batch_size, num_pairs, input_dim) fused representation.

        Returns:
            (batch_size, 1) or (batch_size, num_pairs, 1) continuous Arousal in [0, 1].
        """
        return self.mlp(z)


# Alias for backward compatibility
DimensionRegressionHead = ValenceRegressionHead


class DualRegressionHeads(nn.Module):
    """Dual Independent Valence and Arousal Regression Heads."""

    def __init__(
        self,
        input_dim: int = 2304,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.valence_head = ValenceRegressionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.arousal_head = ArousalRegressionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch_size, input_dim) or (batch_size, num_pairs, input_dim) fused representation Z in R^(3d).

        Returns:
            Tuple of:
            - valence: (batch_size, 1) or (batch_size, num_pairs, 1) continuous Valence in [0, 1]
            - arousal: (batch_size, 1) or (batch_size, num_pairs, 1) continuous Arousal in [0, 1]
        """
        valence = self.valence_head(z)
        arousal = self.arousal_head(z)
        return valence, arousal
