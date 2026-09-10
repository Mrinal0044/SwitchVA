"""Branch 1: Biaffine Aspect-Opinion Span Extractor for NSSG-DimNet."""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiaffineScorer(nn.Module):
    """Biaffine scoring layer for token pairs (i, j).

    Mathematical Formulation:
    -------------------------
    score(i, j) = x_i^T U y_j + W [x_i || y_j] + b

    where:
    - x_i in R^(d_span): Start boundary token representation
    - y_j in R^(d_span): End boundary token representation
    - U in R^(d_span x d_span): Bilinear weight matrix
    - W in R^(2 * d_span): Linear weight vector
    - b in R: Scalar bias
    """

    def __init__(self, span_dim: int = 256):
        super().__init__()
        self.span_dim = span_dim
        self.U = nn.Parameter(torch.Tensor(span_dim, span_dim))
        self.W = nn.Linear(2 * span_dim, 1, bias=True)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.W.bias)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, start_repr: torch.Tensor, end_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            start_repr: (batch_size, seq_len, span_dim) start token representations
            end_repr: (batch_size, seq_len, span_dim) end token representations

        Returns:
            (batch_size, seq_len, seq_len) 2D score grid where entry (b, i, j) is the span score.
        """
        batch_size, seq_len, _ = start_repr.shape

        # 1. Bilinear term: x_i^T U y_j
        # start_u: (B, L, d_span) @ (d_span, d_span) -> (B, L, d_span)
        start_u = torch.matmul(start_repr, self.U)
        # bilinear: (B, L, d_span) @ (B, d_span, L) -> (B, L, L)
        bilinear = torch.bmm(start_u, end_repr.transpose(1, 2))

        # 2. Linear term: W [x_i || y_j] + b
        start_exp = start_repr.unsqueeze(2).expand(batch_size, seq_len, seq_len, self.span_dim)
        end_exp = end_repr.unsqueeze(1).expand(batch_size, seq_len, seq_len, self.span_dim)
        pair_features = torch.cat([start_exp, end_exp], dim=-1)  # (B, L, L, 2*span_dim)
        linear = self.W(pair_features).squeeze(-1)              # (B, L, L)

        scores = bilinear + linear
        return scores


class BiaffineAspectOpinionSpanExtractor(nn.Module):
    """Biaffine Aspect-Opinion Span Extractor (Branch 1 of NSSG-DimNet).

    Maps contextual representations H_gated into 2D candidate span grids
    for Aspect Spans and Opinion Spans, extracts decoded spans, and pools
    span representations.
    """

    def __init__(
        self,
        input_dim: int = 768,
        span_dim: int = 256,
        mlp_dim: Optional[int] = None,
        num_labels: int = 3,
        max_span_length: int = 15,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_dim: Input representation dimension d from SP-GSA.
            span_dim: Dimension d_span of start/end boundary MLPs.
            mlp_dim: Alias for span_dim for backward compatibility.
            num_labels: Number of span categories (reserved/optional).
            max_span_length: Maximum allowed span length in tokens (default 15).
            dropout: Dropout probability.
        """
        super().__init__()
        effective_span_dim = mlp_dim if mlp_dim is not None else span_dim
        self.input_dim = input_dim
        self.span_dim = effective_span_dim
        self.max_span_length = max_span_length

        # 1. Aspect Start & End Projections
        self.aspect_start_mlp = nn.Sequential(
            nn.Linear(input_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.aspect_end_mlp = nn.Sequential(
            nn.Linear(input_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. Opinion Start & End Projections
        self.opinion_start_mlp = nn.Sequential(
            nn.Linear(input_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.opinion_end_mlp = nn.Sequential(
            nn.Linear(input_dim, span_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3. Dedicated Biaffine Scorers
        self.aspect_scorer = BiaffineScorer(span_dim=span_dim)
        self.opinion_scorer = BiaffineScorer(span_dim=span_dim)

        # 4. Span Representation Projection: [H_start || H_end] -> R^d
        self.span_repr_proj = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )

    def compute_valid_span_mask(
        self,
        batch_size: int,
        seq_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Create a boolean mask for valid span pairs (i <= j, length <= max_span_length, non-pad).

        Returns:
            (batch_size, seq_len, seq_len) boolean tensor (True = valid span candidate).
        """
        if device is None and attention_mask is not None:
            device = attention_mask.device
        elif device is None:
            device = torch.device("cpu")

        # Range grid: i_grid[i, j] = i, j_grid[i, j] = j
        i_grid = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, seq_len)
        j_grid = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq_len, seq_len)

        # Start <= End
        triu_mask = i_grid <= j_grid
        # Length <= max_span_length
        len_mask = (j_grid - i_grid + 1) <= self.max_span_length
        base_mask = triu_mask & len_mask  # (seq_len, seq_len)

        valid_mask = base_mask.unsqueeze(0).expand(batch_size, seq_len, seq_len).clone()

        if attention_mask is not None:
            # Token i and token j must both be valid non-padding tokens
            pad_i = attention_mask.unsqueeze(2).expand(batch_size, seq_len, seq_len).bool()
            pad_j = attention_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len).bool()
            valid_mask = valid_mask & pad_i & pad_j

        return valid_mask

    def decode_spans(
        self,
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
        threshold: Optional[float] = None,
        top_k: Optional[int] = 5,
        score_floor: Optional[float] = None,
        strategy: str = "ranked",
    ) -> List[List[Dict[str, Any]]]:
        """Decode candidate spans from a 2D score grid for each batch element.

        Supports two decoding strategies:
        - "ranked" (Step 7 default): Identify valid candidate spans, rank descending by score,
          select top-k candidates, and optionally apply a score floor after ranking.
        - "threshold" (legacy): Filter candidates where score >= threshold, then sort and top-k.

        Args:
            scores: (batch_size, seq_len, seq_len) raw logits
            valid_mask: (batch_size, seq_len, seq_len) boolean mask
            threshold: Logit score threshold for threshold strategy (default 0.0)
            top_k: Optional maximum number of spans to return per sequence (default 5)
            score_floor: Optional minimum score floor applied after candidate ranking
            strategy: "ranked" or "threshold"

        Returns:
            List of decoded span lists for each sequence in batch:
            [[{"start": i, "end": j, "score": s, "probability": p, "length": l}, ...], ...]
        """
        batch_size, seq_len, _ = scores.shape
        decoded_batch: List[List[Dict[str, Any]]] = []

        probs = torch.sigmoid(scores)

        for b in range(batch_size):
            spans = []
            mask_b = valid_mask[b]
            scores_b = scores[b]
            probs_b = probs[b]

            valid_indices = torch.where(mask_b)
            for i, j in zip(valid_indices[0].tolist(), valid_indices[1].tolist()):
                s_val = scores_b[i, j].item()

                if strategy == "threshold":
                    th = threshold if threshold is not None else 0.0
                    if s_val < th:
                        continue

                spans.append({
                    "start": i,
                    "end": j,
                    "score": s_val,
                    "probability": probs_b[i, j].item(),
                    "length": j - i + 1,
                })

            # Sort candidate spans by score descending
            spans.sort(key=lambda x: x["score"], reverse=True)

            # Apply top_k selection
            if top_k is not None and len(spans) > top_k:
                spans = spans[:top_k]

            # Apply score floor after ranking if specified
            if score_floor is not None:
                spans = [sp for sp in spans if sp["score"] >= score_floor]

            decoded_batch.append(spans)

        return decoded_batch

    def extract_span_representation(
        self,
        H_gated: torch.Tensor,
        spans: Optional[torch.Tensor] = None,
        decoded_spans: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> torch.Tensor:
        """Extract boundary-pooled span representation vectors for aspect or opinion spans.

        Args:
            H_gated: (batch_size, seq_len, input_dim)
            spans: Optional (batch_size, 2) explicit [start, end] indices
            decoded_spans: Optional decoded spans list

        Returns:
            (batch_size, input_dim) pooled span representation vector.
        """
        batch_size, seq_len, dim = H_gated.shape
        device = H_gated.device

        # If spans is provided as 3D (B, K, 2), handle each span per item
        if spans is not None and spans.dim() == 3:
            num_spans = spans.shape[1]
            out_spans = []
            for b in range(batch_size):
                b_spans = []
                for k in range(num_spans):
                    s = int(spans[b, k, 0].item())
                    e = int(spans[b, k, 1].item())
                    if s < 0 or e < 0 or s >= seq_len:
                        s_idx, e_idx = 0, 0
                    else:
                        s_idx = max(0, min(seq_len - 1, s))
                        e_idx = max(s_idx, min(seq_len - 1, e))
                    start_vec = H_gated[b, s_idx]
                    end_vec = H_gated[b, e_idx]
                    b_cat = torch.cat([start_vec, end_vec], dim=-1)
                    b_spans.append(self.span_repr_proj(b_cat))
                out_spans.append(torch.stack(b_spans, dim=0))
            return torch.stack(out_spans, dim=0)  # (B, K, dim)

        span_reprs = []
        for b in range(batch_size):
            if spans is not None and b < len(spans) and spans.dim() == 2 and spans[b, 0].item() >= 0:
                s_idx = max(0, min(seq_len - 1, int(spans[b, 0].item())))
                e_idx = max(s_idx, min(seq_len - 1, int(spans[b, 1].item())))
            elif decoded_spans is not None and b < len(decoded_spans) and len(decoded_spans[b]) > 0:
                top_span = decoded_spans[b][0]
                s_idx = top_span["start"]
                e_idx = top_span["end"]
            else:
                # Default to sequence mean representation
                s_idx = 0
                e_idx = seq_len - 1

            start_vec = H_gated[b, s_idx]
            end_vec = H_gated[b, e_idx]

            # Boundary concatenation [H_start || H_end]
            boundary_cat = torch.cat([start_vec, end_vec], dim=-1)
            span_vec = self.span_repr_proj(boundary_cat)
            span_reprs.append(span_vec)

        return torch.stack(span_reprs, dim=0)

    def forward(
        self,
        H_gated: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aspect_spans: Optional[torch.Tensor] = None,
        opinion_spans: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = 5,
        top_k_aspect: Optional[int] = None,
        top_k_opinion: Optional[int] = None,
        score_floor: Optional[float] = None,
        strategy: str = "ranked",
    ) -> Dict[str, Any]:
        """
        Args:
            H_gated: (batch_size, seq_len, input_dim) from SP-GSA
            attention_mask: Optional (batch_size, seq_len)
            aspect_spans: Optional (batch_size, 2) gold/target aspect bounds
            opinion_spans: Optional (batch_size, 2) gold/target opinion bounds
            threshold: Logit score threshold for span decoding (used if strategy='threshold')
            top_k: Default max spans to decode per sequence
            top_k_aspect: Optional separate top-k for aspect spans
            top_k_opinion: Optional separate top-k for opinion spans
            score_floor: Optional minimum score floor applied after ranking
            strategy: Decoding strategy ("ranked" or "threshold")

        Returns:
            Dictionary containing aspect/opinion scores, valid_span_mask, decoded spans, and representations.
        """
        batch_size, seq_len, _ = H_gated.shape

        # 1. Compute Boundary Projections
        asp_start = self.aspect_start_mlp(H_gated)  # (B, N, span_dim)
        asp_end = self.aspect_end_mlp(H_gated)      # (B, N, span_dim)

        op_start = self.opinion_start_mlp(H_gated)  # (B, N, span_dim)
        op_end = self.opinion_end_mlp(H_gated)      # (B, N, span_dim)

        # 2. Compute 2D Biaffine Score Grids
        aspect_scores = self.aspect_scorer(asp_start, asp_end)    # (B, N, N)
        opinion_scores = self.opinion_scorer(op_start, op_end)    # (B, N, N)

        # 3. Create Valid Span Mask (start <= end, max length, padding)
        valid_mask = self.compute_valid_span_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            attention_mask=attention_mask,
            device=H_gated.device,
        )

        # Apply large negative value to invalid candidate positions for safe softmax/decoding
        masked_aspect_scores = torch.where(valid_mask, aspect_scores, torch.tensor(-1e9, device=H_gated.device))
        masked_opinion_scores = torch.where(valid_mask, opinion_scores, torch.tensor(-1e9, device=H_gated.device))

        # 4. Decode Spans using configured strategy (ranked top-k by default)
        k_asp = top_k_aspect if top_k_aspect is not None else top_k
        k_op = top_k_opinion if top_k_opinion is not None else top_k

        decoded_aspects = self.decode_spans(
            masked_aspect_scores,
            valid_mask,
            threshold=threshold,
            top_k=k_asp,
            score_floor=score_floor,
            strategy=strategy,
        )
        decoded_opinions = self.decode_spans(
            masked_opinion_scores,
            valid_mask,
            threshold=threshold,
            top_k=k_op,
            score_floor=score_floor,
            strategy=strategy,
        )

        # 5. Extract Span-Level Representations
        aspect_repr = self.extract_span_representation(
            H_gated,
            spans=aspect_spans,
            decoded_spans=decoded_aspects,
        )
        opinion_repr = self.extract_span_representation(
            H_gated,
            spans=opinion_spans,
            decoded_spans=decoded_opinions,
        )

        return {
            "aspect_scores": aspect_scores,
            "opinion_scores": opinion_scores,
            "masked_aspect_scores": masked_aspect_scores,
            "masked_opinion_scores": masked_opinion_scores,
            "valid_span_mask": valid_mask,
            "decoded_aspect_spans": decoded_aspects,
            "decoded_opinion_spans": decoded_opinions,
            "aspect_repr": aspect_repr,
            "opinion_repr": opinion_repr,
            # Backward-compatibility aliases:
            "aspect_representations": aspect_repr,
            "opinion_representations": opinion_repr,
            "span_logits": aspect_scores,
        }


# Alias for backward compatibility
BiaffineSpanExtractor = BiaffineAspectOpinionSpanExtractor
