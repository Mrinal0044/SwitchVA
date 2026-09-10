"""Span-level loss functions for Biaffine Aspect-Opinion Span Extractor (Phase 3)."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_gold_span_grid(
    batch_size: int,
    seq_len: int,
    gold_spans: List[List[Dict[str, Any]]],
    valid_mask: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct a 2D binary target grid (B, N, N) and supervision mask for gold spans.

    Args:
        batch_size: Batch size B
        seq_len: Sequence length N
        gold_spans: List of gold span dicts per sequence:
                    [[{"start": s, "end": e, "is_training_safe": bool}, ...], ...]
        valid_mask: (B, N, N) boolean mask for valid span positions (i <= j, length <= max_len, non-pad)
        device: Target compute device

    Returns:
        Tuple of:
        - gold_grid: (B, N, N) float tensor with 1.0 at gold span positions and 0.0 elsewhere
        - supervision_mask: (B, N, N) boolean tensor indicating positions included in loss
    """
    if device is None:
        device = valid_mask.device

    gold_grid = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=device)
    supervision_mask = valid_mask.clone()

    for b in range(batch_size):
        if b >= len(gold_spans):
            continue

        for span in gold_spans[b]:
            s = span.get("start", -1)
            e = span.get("end", -1)
            is_safe = span.get("is_training_safe", True)

            if s >= 0 and e >= 0 and s < seq_len and e < seq_len and s <= e:
                if is_safe:
                    gold_grid[b, s, e] = 1.0
                else:
                    # If span is unsafe (approximate/unmatched), exclude this cell from negative loss
                    supervision_mask[b, s, e] = False

    return gold_grid, supervision_mask


class BiaffineSpanLoss(nn.Module):
    """Binary cross-entropy span loss over valid 2D span candidate grids."""

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        aspect_pos_weight: float = 50.0,
        opinion_pos_weight: float = 50.0,
        aspect_weight: float = 1.0,
        opinion_weight: float = 1.0,
    ):
        """
        Args:
            pos_weight: Legacy parameter. If provided, overrides aspect_pos_weight and opinion_pos_weight.
            aspect_pos_weight: Multiplier for positive gold aspect span cells.
            opinion_pos_weight: Multiplier for positive gold opinion span cells.
            aspect_weight: Multiplier for aspect span loss component.
            opinion_weight: Multiplier for opinion span loss component.
        """
        super().__init__()
        if pos_weight is not None:
            aspect_pos_weight = pos_weight
            opinion_pos_weight = pos_weight

        self.pos_weight = pos_weight if pos_weight is not None else (aspect_pos_weight + opinion_pos_weight) / 2.0
        self.aspect_pos_weight = aspect_pos_weight
        self.opinion_pos_weight = opinion_pos_weight
        self.aspect_weight = aspect_weight
        self.opinion_weight = opinion_weight

    def forward(
        self,
        aspect_scores: torch.Tensor,
        opinion_scores: torch.Tensor,
        valid_span_mask: torch.Tensor,
        gold_aspect_spans: Union[torch.Tensor, List[List[Dict[str, Any]]]],
        gold_opinion_spans: Union[torch.Tensor, List[List[Dict[str, Any]]]],
        aspect_supervision_mask: Optional[torch.Tensor] = None,
        opinion_supervision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute masked BCE loss for aspect and opinion span grids.

        Args:
            aspect_scores: (B, N, N) raw aspect logits
            opinion_scores: (B, N, N) raw opinion logits
            valid_span_mask: (B, N, N) boolean mask
            gold_aspect_spans: Either (B, N, N) tensor target grid OR list of span dicts per batch item
            gold_opinion_spans: Either (B, N, N) tensor target grid OR list of span dicts per batch item
            aspect_supervision_mask: Optional (B, N, N) boolean mask
            opinion_supervision_mask: Optional (B, N, N) boolean mask

        Returns:
            Dictionary containing 'aspect_span_loss', 'opinion_span_loss', 'total_span_loss'.
        """
        batch_size, seq_len, _ = aspect_scores.shape
        device = aspect_scores.device

        # 1. Aspect Targets & Supervision Mask
        if isinstance(gold_aspect_spans, torch.Tensor):
            gold_asp_grid = gold_aspect_spans.to(device)
            asp_sup_mask = (aspect_supervision_mask.to(device) if aspect_supervision_mask is not None else valid_span_mask.to(device))
        else:
            gold_asp_grid, asp_sup_mask = construct_gold_span_grid(
                batch_size, seq_len, gold_aspect_spans, valid_span_mask, device=device
            )

        # 2. Opinion Targets & Supervision Mask
        if isinstance(gold_opinion_spans, torch.Tensor):
            gold_op_grid = gold_opinion_spans.to(device)
            op_sup_mask = (opinion_supervision_mask.to(device) if opinion_supervision_mask is not None else valid_span_mask.to(device))
        else:
            gold_op_grid, op_sup_mask = construct_gold_span_grid(
                batch_size, seq_len, gold_opinion_spans, valid_span_mask, device=device
            )

        # 3. Class Weighting per task
        asp_pw = torch.tensor([self.aspect_pos_weight], device=device, dtype=torch.float)
        op_pw = torch.tensor([self.opinion_pos_weight], device=device, dtype=torch.float)

        asp_bce = nn.BCEWithLogitsLoss(pos_weight=asp_pw, reduction="mean")
        op_bce = nn.BCEWithLogitsLoss(pos_weight=op_pw, reduction="mean")

        # 4. Compute Aspect Span Loss
        if asp_sup_mask.sum() > 0:
            asp_pred_flat = aspect_scores[asp_sup_mask]
            asp_gold_flat = gold_asp_grid[asp_sup_mask]
            loss_asp = asp_bce(asp_pred_flat, asp_gold_flat)
        else:
            loss_asp = torch.tensor(0.0, device=device, requires_grad=True)

        # 5. Compute Opinion Span Loss
        if op_sup_mask.sum() > 0:
            op_pred_flat = opinion_scores[op_sup_mask]
            op_gold_flat = gold_op_grid[op_sup_mask]
            loss_op = op_bce(op_pred_flat, op_gold_flat)
        else:
            loss_op = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = self.aspect_weight * loss_asp + self.opinion_weight * loss_op

        return {
            "aspect_span_loss": loss_asp,
            "opinion_span_loss": loss_op,
            "total_span_loss": total_loss,
            "total_loss": total_loss,
            "gold_aspect_grid": gold_asp_grid,
            "gold_opinion_grid": gold_op_grid,
        }
