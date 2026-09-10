"""Aspect-Guided Representation Fusion Module for NSSG-DimNet (Phase 5).

Fuses Branch 1 (Biaffine Span Extractor) and Branch 2 (H-NSG + RGAT) representations
into an exact 3d-dimensional representation:

    Z = [Z_span || Z_graph || Z_cross] in R^(3d)

where:
- Z_span in R^d: LayerNorm(W_s [H_aspect || H_opinion])
- Z_graph in R^d: LayerNorm(W_g [G_aspect || G_opinion])
- Z_cross in R^d: Aspect-Guided Mutual Cross-Attention context between both branches
"""

from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from .cross_attention import AspectGuidedMutualCrossAttention


class AspectGuidedFusion(nn.Module):
    """Aspect-Guided Representation Fusion for DimABSA.

    Combines span-level and graph-level representations under aspect guidance
    to produce the final fused representation Z in R^(3d).
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
            num_heads: Number of attention heads for mutual cross-attention.
            dropout: Dropout probability.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fused_dim = 3 * hidden_dim

        # Mutual Cross-Attention mechanism
        self.mutual_cross_attention = AspectGuidedMutualCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Projections for Branch 1 and Branch 2 span/opinion combinations
        self.span_fuse_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.graph_fuse_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.norm_z = nn.LayerNorm(3 * hidden_dim)

    def forward(
        self,
        aspect_span_repr: torch.Tensor,
        opinion_span_repr: Optional[torch.Tensor] = None,
        graph_aspect_repr: Optional[torch.Tensor] = None,
        graph_opinion_repr: Optional[torch.Tensor] = None,
        graph_node_features: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """Compute the fused representation Z in R^(3d).

        Args:
            aspect_span_repr: (B, d) or (B, K, d) from Branch 1 (Biaffine Span).
            opinion_span_repr: Optional (B, d) or (B, K, d) from Branch 1.
            graph_aspect_repr: Optional (B, d) or (B, K, d) from Branch 2 (RGAT).
            graph_opinion_repr: Optional (B, d) or (B, K, d) from Branch 2.
            graph_node_features: Optional (B, N, d) all graph token nodes from RGAT.
            graph_mask: Optional attention mask for graph nodes.
            span_mask: Optional attention mask for spans.
            debug: If True, include detailed intermediate representations in output.

        Returns:
            Dictionary containing:
            - 'z': (B, 3d) or (B, K, 3d) fused representation tensor
            - 'z_span': (B, d) or (B, K, d) Branch 1 component
            - 'z_graph': (B, d) or (B, K, d) Branch 2 component
            - 'z_cross': (B, d) or (B, K, d) mutual cross-attention component
            - 'attn_weights': Attention maps from mutual cross-attention
        """
        is_2d = (aspect_span_repr.dim() == 2)
        if is_2d:
            aspect_span_repr = aspect_span_repr.unsqueeze(1)
            if opinion_span_repr is not None and opinion_span_repr.dim() == 2:
                opinion_span_repr = opinion_span_repr.unsqueeze(1)
            if graph_aspect_repr is not None and graph_aspect_repr.dim() == 2:
                graph_aspect_repr = graph_aspect_repr.unsqueeze(1)
            if graph_opinion_repr is not None and graph_opinion_repr.dim() == 2:
                graph_opinion_repr = graph_opinion_repr.unsqueeze(1)

        B, num_pairs, d = aspect_span_repr.shape
        assert d == self.hidden_dim, f"Expected hidden_dim {self.hidden_dim}, got {d}"

        # Fallbacks for optional inputs
        op_span = opinion_span_repr if opinion_span_repr is not None else torch.zeros_like(aspect_span_repr)
        g_asp = graph_aspect_repr if graph_aspect_repr is not None else aspect_span_repr
        g_op = graph_opinion_repr if graph_opinion_repr is not None else op_span

        # 1. Branch 1 Representation: Z_span in R^d
        span_combined = torch.cat([aspect_span_repr, op_span], dim=-1)
        z_span = self.span_fuse_proj(span_combined)  # (B, num_pairs, d)

        # 2. Branch 2 Representation: Z_graph in R^d
        graph_combined = torch.cat([g_asp, g_op], dim=-1)
        z_graph = self.graph_fuse_proj(graph_combined)  # (B, num_pairs, d)

        # 3. Cross-Attention Representation: Z_cross in R^d
        z_cross, attn_dict = self.mutual_cross_attention(
            aspect_span_repr=aspect_span_repr,
            opinion_span_repr=op_span,
            graph_node_features=graph_node_features,
            graph_aspect_repr=g_asp,
            graph_opinion_repr=g_op,
            graph_mask=graph_mask,
            span_mask=span_mask,
        )  # (B, num_pairs, d)

        # 4. Concatenate into Fused Representation Z in R^(3d)
        Z = torch.cat([z_span, z_graph, z_cross], dim=-1)  # (B, num_pairs, 3d)
        Z = self.norm_z(Z)

        # Explicit assertion for exact 3d dimension
        assert Z.shape[-1] == 3 * self.hidden_dim, (
            f"Dimension mismatch: expected {3 * self.hidden_dim}, got {Z.shape[-1]}"
        )

        if is_2d:
            Z = Z.squeeze(1)
            z_span = z_span.squeeze(1)
            z_graph = z_graph.squeeze(1)
            z_cross = z_cross.squeeze(1)

        result: Dict[str, Any] = {
            "z": Z,
            "fused_representation": Z,  # Alias
            "z_span": z_span,
            "z_graph": z_graph,
            "z_cross": z_cross,
            "attn_weights": attn_dict,
            "cross_attn_weights": attn_dict.get("span_to_graph"),
        }

        if debug:
            result["debug_info"] = {
                "span_branch_in": aspect_span_repr,
                "graph_branch_in": g_asp,
                "mutual_attention_dict": attn_dict,
            }

        return result
