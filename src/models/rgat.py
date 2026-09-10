"""Relational Graph Attention Network (RGAT) for heterogeneous graph message passing."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hnsg import NUM_RELATIONS, REL_SYNTACTIC, REL_SWITCH, REL_ASPECT_OPINION


def extract_graph_span_representation(
    H_graph: torch.Tensor,
    spans: Optional[torch.Tensor] = None,
    pool_mode: str = "mean",
    decoded_spans: Optional[List[List[Dict[str, Any]]]] = None,
) -> torch.Tensor:
    """Extract graph-aware pooled span representation for aspect or opinion spans.

    Args:
        H_graph: (batch_size, seq_len, hidden_dim) output of RGAT
        spans: Optional (batch_size, num_spans, 2) or (batch_size, 2) [start, end] tensor
        pool_mode: 'mean', 'max', or 'endpoints'
        decoded_spans: Optional list of decoded span dictionaries

    Returns:
        Tensor of pooled span representations. If spans is (batch_size, num_spans, 2), returns
        (batch_size, num_spans, hidden_dim) [or (batch_size, num_spans, 2*hidden_dim) for endpoints].
        If spans is (batch_size, 2), returns (batch_size, hidden_dim).
    """
    batch_size, seq_len, dim = H_graph.shape

    if spans is None:
        if decoded_spans is not None:
            span_reprs = []
            for b in range(batch_size):
                if b < len(decoded_spans) and len(decoded_spans[b]) > 0:
                    top_span = decoded_spans[b][0]
                    s_idx = max(0, min(seq_len - 1, top_span["start"]))
                    e_idx = max(s_idx, min(seq_len - 1, top_span["end"]))
                else:
                    s_idx = 0
                    e_idx = seq_len - 1
                span_slice = H_graph[b, s_idx : e_idx + 1]
                if pool_mode == "max":
                    pooled, _ = torch.max(span_slice, dim=0)
                elif pool_mode == "endpoints":
                    pooled = torch.cat([H_graph[b, s_idx], H_graph[b, e_idx]], dim=-1)
                else:
                    pooled = torch.mean(span_slice, dim=0)
                span_reprs.append(pooled)
            return torch.stack(span_reprs, dim=0)
        else:
            if pool_mode == "endpoints":
                return torch.cat([H_graph[:, 0], H_graph[:, -1]], dim=-1)
            elif pool_mode == "max":
                pooled, _ = torch.max(H_graph, dim=1)
                return pooled
            else:
                return torch.mean(H_graph, dim=1)

    is_2d = (spans.dim() == 2)
    if is_2d:
        spans = spans.unsqueeze(1)  # (B, 1, 2)

    num_spans = spans.shape[1]
    out_spans = []

    for b in range(batch_size):
        b_spans = []
        for k in range(num_spans):
            s = int(spans[b, k, 0].item())
            e = int(spans[b, k, 1].item())
            if s < 0 or e < 0 or s >= seq_len:
                s_idx = 0
                e_idx = 0
            else:
                s_idx = max(0, min(seq_len - 1, s))
                e_idx = max(s_idx, min(seq_len - 1, e))

            span_slice = H_graph[b, s_idx : e_idx + 1]
            if pool_mode == "max":
                pooled, _ = torch.max(span_slice, dim=0)
            elif pool_mode == "endpoints":
                pooled = torch.cat([H_graph[b, s_idx], H_graph[b, e_idx]], dim=-1)
            else:  # mean
                pooled = torch.mean(span_slice, dim=0)
            b_spans.append(pooled)
        out_spans.append(torch.stack(b_spans, dim=0))

    res = torch.stack(out_spans, dim=0)  # (B, num_spans, dim or 2*dim)
    if is_2d:
        res = res.squeeze(1)  # (B, dim or 2*dim)
    return res


class RGATLayer(nn.Module):
    """Single Relational Graph Attention Network (RGAT) Layer.

    Applies relation-specific linear transformations and multi-head attention
    across 3 distinct heterogeneous relations:
    - r=0: SYNTACTIC
    - r=1: SWITCH
    - r=2: ASPECT_OPINION
    """

    def __init__(
        self,
        d_model: int = 768,
        in_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        num_relations: int = NUM_RELATIONS,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
    ):
        super().__init__()
        in_dim = in_dim or d_model
        out_dim = out_dim or d_model
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_residual = use_residual

        # 1. Relation-specific linear projections: W_r for each relation type r
        self.W_r = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])
        # Alias for backward compatibility
        self.rel_proj = self.W_r

        # 2. Multi-head Query, Key, Value projections per relation
        self.q_proj = nn.ModuleList([
            nn.Linear(out_dim, out_dim) for _ in range(num_relations)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(out_dim, out_dim) for _ in range(num_relations)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(out_dim, out_dim) for _ in range(num_relations)
        ])

        # 3. Learnable relation importance gates
        self.rel_gate = nn.Parameter(torch.ones(num_relations))

        # 4. Output projection, normalization, and activation
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

        if use_residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = nn.Identity()

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: (batch_size, seq_len, in_dim)
            adj_matrices: (batch_size, num_relations, seq_len, seq_len)
            attention_mask: Optional (batch_size, seq_len) padding mask

        Returns:
            (batch_size, seq_len, out_dim) updated graph node representations.
        """
        batch_size, seq_len, _ = node_features.shape
        rel_outputs: List[torch.Tensor] = []
        rel_weights = torch.softmax(self.rel_gate, dim=0)

        for r in range(self.num_relations):
            adj_r = adj_matrices[:, r]  # (B, N, N)

            # Relation-specific node projection
            h_r = self.W_r[r](node_features)  # (B, N, out_dim)

            q = self.q_proj[r](h_r).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj[r](h_r).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj[r](h_r).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Attention scores: (B, num_heads, N, N)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            scores = F.leaky_relu(scores, negative_slope=0.2)

            # Mask non-existing edges with large negative value (-1e9)
            adj_mask = (adj_r.unsqueeze(1) == 0).float() * -1e9
            scores = scores + adj_mask

            if attention_mask is not None:
                pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2).float() * -1e9
                scores = scores + pad_mask

            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.dropout(attn)

            # Context aggregation for relation r: (B, N, out_dim)
            out_r = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.out_dim)
            rel_outputs.append(rel_weights[r] * out_r)

        # Aggregate across all 3 relations
        aggregated = sum(rel_outputs)
        aggregated = self.out_proj(aggregated)
        aggregated = self.act(aggregated)
        aggregated = self.dropout(aggregated)

        # Residual connection + LayerNorm
        res = self.res_proj(node_features)
        output = self.layer_norm(res + aggregated)

        return output


class RGAT(nn.Module):
    """Multi-layer Relational Graph Attention Network (RGAT) (Branch 2)."""

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 4,
        num_relations: int = NUM_RELATIONS,
        dropout: float = 0.2,
        use_residual: bool = True,
    ):
        super().__init__()
        dim = hidden_dim or d_model
        self.hidden_dim = dim
        self.d_model = dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            RGATLayer(
                d_model=dim,
                in_dim=dim,
                out_dim=dim,
                num_relations=num_relations,
                num_heads=num_heads,
                dropout=dropout,
                use_residual=use_residual,
            )
            for _ in range(num_layers)
        ])

        self.span_pool_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )

    def extract_graph_span_representation(
        self,
        H_graph: torch.Tensor,
        spans: Optional[torch.Tensor] = None,
        decoded_spans: Optional[List[List[Dict[str, Any]]]] = None,
        pool_mode: str = "mean",
    ) -> torch.Tensor:
        """Extract graph-aware pooled span representation for aspect or opinion spans."""
        return extract_graph_span_representation(
            H_graph=H_graph,
            spans=spans,
            pool_mode=pool_mode,
            decoded_spans=decoded_spans,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aspect_spans: Optional[torch.Tensor] = None,
        opinion_spans: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            node_features: (batch_size, seq_len, hidden_dim)
            adj_matrices: (batch_size, num_relations, seq_len, seq_len)
            attention_mask: Optional (batch_size, seq_len)
            aspect_spans: Optional aspect span indices
            opinion_spans: Optional opinion span indices
            seq_len: Optional sequence length

        Returns:
            H_graph if spans are not provided, or (H_graph, graph_aspect_repr, graph_opinion_repr)
            if aspect/opinion spans are provided.
        """
        h = node_features
        for layer in self.layers:
            h = layer(h, adj_matrices, attention_mask=attention_mask)

        if aspect_spans is not None or opinion_spans is not None:
            graph_aspect_repr = self.extract_graph_span_representation(h, spans=aspect_spans)
            graph_opinion_repr = self.extract_graph_span_representation(h, spans=opinion_spans)
            return h, graph_aspect_repr, graph_opinion_repr

        if seq_len is not None:
            graph_aspect_repr = self.extract_graph_span_representation(h)
            return h, graph_aspect_repr

        return h
