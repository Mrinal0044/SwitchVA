"""Switch-Point Gated Self-Attention (SP-GSA) module."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class SPGSA(nn.Module):
    """Switch-Point Gated Self-Attention (SP-GSA).

    Approved Formulation:
    --------------------
    G = sigmoid(W_g [H || E_switch] + b_g)
    H_gated = LayerNorm(G ⊙ H + H)

    where:
    - H in R^(B x N x d): Transformer contextual token representations
    - E_switch in R^(B x N x d): Trainable switch embedding
    - W_g in R^(2d x d): Learnable gating projection
    - G in R^(B x N x d): Gating tensor
    - ⊙: Element-wise Hadamard product
    - LayerNorm with residual connection
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Representation dimension d.
            num_heads: Number of attention heads (reserved/optional).
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Gating network W_g [H || E_switch] + b_g -> R^d
        self.gate_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor,
        E_switch: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: (batch_size, seq_len, hidden_dim) transformer output.
            E_switch: (batch_size, seq_len, hidden_dim) switch embedding.
            attention_mask: Optional (batch_size, seq_len) attention mask (1=valid, 0=pad).

        Returns:
            Tuple of:
            - H_gated: (batch_size, seq_len, hidden_dim) modulated representation
            - G: (batch_size, seq_len, hidden_dim) gate values in [0, 1]
        """
        # 1. Concatenate H and E_switch along feature dimension
        concat_features = torch.cat([H, E_switch], dim=-1)  # (B, N, 2d)

        # 2. Compute Gating Tensor G = sigmoid(W_g [H || E_switch])
        gate_logits = self.gate_proj(concat_features)
        G = torch.sigmoid(gate_logits)  # (B, N, d)

        # 3. Apply element-wise gating: G ⊙ H
        gated_H = G * H

        # 4. Residual connection + LayerNorm
        residual = gated_H + H
        H_gated = self.layer_norm(self.dropout(residual))

        # 5. Mask out padding positions if attention mask provided
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            H_gated = H_gated * mask_expanded
            G = G * mask_expanded

        return H_gated, G
