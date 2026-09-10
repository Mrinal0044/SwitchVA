"""Aspect-Guided Mutual Cross-Attention module for NSSG-DimNet (Phase 5).

Implements mutual cross-attention between:
- Branch 1: Biaffine Span representations (H_aspect, H_opinion)
- Branch 2: Heterogeneous Graph RGAT representations (G_aspect, G_opinion / H_graph)

Under explicit aspect-guidance:
- Path A: Span representations attend to Graph representations (conditioned on aspect).
- Path B: Graph representations attend to Span representations (conditioned on aspect).
Producing the cross-attended context representation Z_cross in R^d.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectGuidedMutualCrossAttention(nn.Module):
    """Aspect-Guided Mutual Cross-Attention between Span and Graph branches.

    Allows two-way bidirectional information exchange:
    1. Span branch attending to Graph branch (Path A: Q from Span, K/V from Graph)
    2. Graph branch attending to Span branch (Path B: Q from Graph, K/V from Span)

    Conditioning on Aspect:
    The aspect vector serves as the primary guidance query, ensuring cross-attention
    focuses on sentiment and syntax relevant to the specific target aspect.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_dim: Model representation dimension d.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # -------------------------------------------------------------
        # Path A: Span -> Graph Cross-Attention (Q from Span, K/V from Graph)
        # -------------------------------------------------------------
        self.q_span = nn.Linear(hidden_dim, hidden_dim)
        self.k_graph = nn.Linear(hidden_dim, hidden_dim)
        self.v_graph = nn.Linear(hidden_dim, hidden_dim)
        self.out_span_to_graph = nn.Linear(hidden_dim, hidden_dim)

        # -------------------------------------------------------------
        # Path B: Graph -> Span Cross-Attention (Q from Graph, K/V from Span)
        # -------------------------------------------------------------
        self.q_graph = nn.Linear(hidden_dim, hidden_dim)
        self.k_span = nn.Linear(hidden_dim, hidden_dim)
        self.v_span = nn.Linear(hidden_dim, hidden_dim)
        self.out_graph_to_span = nn.Linear(hidden_dim, hidden_dim)

        # Aspect Guidance Gate / Projection
        self.aspect_cond_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Mutual fusion projection: combines Path A and Path B into Z_cross in R^d
        self.fuse_cross_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_span = nn.LayerNorm(hidden_dim)
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _cross_attention_block(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-head cross-attention block."""
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        B, Q_len, _ = query.shape
        _, K_len, _ = key.shape

        q = q_proj(query).view(B, Q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_proj(key).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_proj(value).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, num_heads, Q_len, K_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = (key_padding_mask == 0).unsqueeze(1).unsqueeze(2).float() * -1e9
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_dropped = self.dropout(attn_weights)

        # Context: (B, num_heads, Q_len, head_dim) -> (B, Q_len, d)
        context = torch.matmul(attn_dropped, v).transpose(1, 2).contiguous().view(B, Q_len, self.hidden_dim)
        context = out_proj(context)
        context = self.dropout(context)
        return context, attn_weights

    def forward(
        self,
        aspect_span_repr: torch.Tensor,
        arg2: Optional[torch.Tensor] = None,
        arg3: Optional[torch.Tensor] = None,
        opinion_span_repr: Optional[torch.Tensor] = None,
        graph_node_features: Optional[torch.Tensor] = None,
        graph_aspect_repr: Optional[torch.Tensor] = None,
        graph_opinion_repr: Optional[torch.Tensor] = None,
        graph_mask: Optional[torch.Tensor] = None,
        span_mask: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        """Compute aspect-guided mutual cross-attention."""
        legacy_mode = False

        # Detect legacy positional call: cross_attn(aspect_span_repr, graph_node_features, graph_aspect_repr)
        if arg2 is not None and arg3 is not None and graph_node_features is None and graph_aspect_repr is None:
            graph_node_features = arg2
            graph_aspect_repr = arg3
            opinion_span_repr = None
            legacy_mode = True
        elif arg2 is not None and opinion_span_repr is None:
            opinion_span_repr = arg2

        is_2d = (aspect_span_repr.dim() == 2)
        if is_2d:
            aspect_span_repr = aspect_span_repr.unsqueeze(1)  # (B, 1, d)
            if opinion_span_repr is not None and opinion_span_repr.dim() == 2:
                opinion_span_repr = opinion_span_repr.unsqueeze(1)
            if graph_aspect_repr is not None and graph_aspect_repr.dim() == 2:
                graph_aspect_repr = graph_aspect_repr.unsqueeze(1)
            if graph_opinion_repr is not None and graph_opinion_repr.dim() == 2:
                graph_opinion_repr = graph_opinion_repr.unsqueeze(1)

        B, num_pairs, d = aspect_span_repr.shape

        op_span = opinion_span_repr if opinion_span_repr is not None else torch.zeros_like(aspect_span_repr)
        g_asp = graph_aspect_repr if graph_aspect_repr is not None else aspect_span_repr
        g_op = graph_opinion_repr if graph_opinion_repr is not None else op_span

        if graph_node_features is not None:
            graph_memory = graph_node_features
            if graph_memory.dim() == 2:
                graph_memory = graph_memory.unsqueeze(1)
        else:
            graph_memory = torch.cat([g_asp, g_op], dim=1)

        span_memory = torch.cat([aspect_span_repr, op_span], dim=1)

        # Aspect Guidance Conditioning
        asp_guidance_span = self.aspect_cond_proj(torch.cat([aspect_span_repr, g_asp], dim=-1))
        asp_guidance_graph = self.aspect_cond_proj(torch.cat([g_asp, aspect_span_repr], dim=-1))

        # Path A: Span Branch -> Graph Memory
        context_span_to_graph, attn_span_to_graph = self._cross_attention_block(
            query=asp_guidance_span,
            key=graph_memory,
            value=graph_memory,
            q_proj=self.q_span,
            k_proj=self.k_graph,
            v_proj=self.v_graph,
            out_proj=self.out_span_to_graph,
            key_padding_mask=graph_mask,
        )

        # Path B: Graph Branch -> Span Memory
        context_graph_to_span, attn_graph_to_span = self._cross_attention_block(
            query=asp_guidance_graph,
            key=span_memory,
            value=span_memory,
            q_proj=self.q_graph,
            k_proj=self.k_span,
            v_proj=self.v_span,
            out_proj=self.out_graph_to_span,
            key_padding_mask=span_mask,
        )

        # Fuse both mutual directions into Z_cross in R^d
        mutual_context = torch.cat([context_span_to_graph, context_graph_to_span], dim=-1)
        z_cross = self.fuse_cross_proj(mutual_context)
        z_cross = self.norm_cross(aspect_span_repr + z_cross)

        if is_2d:
            z_cross = z_cross.squeeze(1)
            aspect_span_repr = aspect_span_repr.squeeze(1)
            g_asp = g_asp.squeeze(1)

        attn_dict = {
            "span_to_graph": attn_span_to_graph,
            "graph_to_span": attn_graph_to_span,
            "cross_attn_weights": attn_span_to_graph,
        }

        if legacy_mode:
            # Construct legacy fused Z: [norm(aspect), norm(graph_aspect), z_cross]
            h_asp = self.norm_span(aspect_span_repr)
            h_g_asp = self.norm_graph(g_asp)
            Z = torch.cat([h_asp, h_g_asp, z_cross], dim=-1)
            return Z, attn_span_to_graph

        return z_cross, attn_dict
