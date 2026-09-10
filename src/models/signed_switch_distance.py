"""Signed Switch-Distance computation for code-switched sequences."""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class SignedSwitchDistance(nn.Module):
    """Computes directional signed distance from each token to the nearest code-switch boundary.

    Mathematical Definition:
    -------------------------
    Let S = {s_1, s_2, ..., s_M} be the set of switch-point token indices where
    language_id(s_k) != language_id(s_k - 1) across consecutive valid tokens.

    For any token at index i:
        s*(i) = argmin_{s in S} |i - s|
        signed_distance(i) = i - s*(i)

    Sign Convention:
    - Distance < 0: Token occurs BEFORE the switch point (e.g., -2, -1)
    - Distance = 0: Token is AT the switch point (start of the new language)
    - Distance > 0: Token occurs AFTER the switch point (e.g., +1, +2)

    Boundary and Edge Handling:
    - Distance is clamped to [-D_max, +D_max].
    - Shifted index for embedding lookup: index(i) = clamped_dist(i) + D_max + 1
    - No switch in sequence: assigned special distance index 0 (NO_SWITCH).
    - Padding tokens: assigned index 2 * D_max + 2 (PAD_INDEX).
    """

    def __init__(
        self,
        max_distance: int = 32,
        special_lang_id: int = 3,
        pad_lang_id: int = 4,
    ):
        """
        Args:
            max_distance: Maximum absolute distance for clipping (D_max).
            special_lang_id: Language ID assigned to special tokens ([CLS], [SEP]).
            pad_lang_id: Language ID assigned to padding tokens.
        """
        super().__init__()
        self.max_distance = max_distance
        self.special_lang_id = special_lang_id
        self.pad_lang_id = pad_lang_id

        # Vocabulary size for embedding:
        # Index 0: NO_SWITCH
        # Index 1 to 2 * max_distance + 1: Distances from -max_distance to +max_distance
        # Index 2 * max_distance + 2: PAD
        self.num_embeddings = 2 * max_distance + 3
        self.no_switch_idx = 0
        self.pad_idx = 2 * max_distance + 2

    def forward(
        self,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute signed distance and embedding lookup indices for a batch.

        Args:
            lang_ids: (batch_size, seq_len) numeric language IDs.
            attention_mask: Optional (batch_size, seq_len) attention mask (1=valid, 0=pad).

        Returns:
            Tuple of:
            - raw_signed_distances: (batch_size, seq_len) float/int tensor with distances in [-D_max, D_max]
            - distance_indices: (batch_size, seq_len) long tensor for SwitchEmbedding lookup
        """
        batch_size, seq_len = lang_ids.shape
        device = lang_ids.device

        raw_signed_distances = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        distance_indices = torch.full((batch_size, seq_len), self.no_switch_idx, dtype=torch.long, device=device)

        for b in range(batch_size):
            seq = lang_ids[b]
            mask = attention_mask[b] if attention_mask is not None else (seq != self.pad_lang_id)

            # Valid tokens: not padding and not special
            valid_mask = mask.bool() & (seq != self.pad_lang_id) & (seq != self.special_lang_id)
            valid_indices = torch.where(valid_mask)[0].tolist()

            # Set padding positions
            pad_positions = torch.where(~mask.bool() | (seq == self.pad_lang_id))[0]
            distance_indices[b, pad_positions] = self.pad_idx
            raw_signed_distances[b, pad_positions] = 0.0

            if len(valid_indices) <= 1:
                # No possible switch with 0 or 1 valid tokens
                continue

            # Identify code-switch points: positions where language changes relative to previous valid token
            switch_points: List[int] = []
            for k in range(1, len(valid_indices)):
                prev_idx = valid_indices[k - 1]
                curr_idx = valid_indices[k]
                if seq[curr_idx] != seq[prev_idx]:
                    switch_points.append(curr_idx)

            if not switch_points:
                # Monolingual / no switch in sequence
                continue

            # Compute signed distance to nearest switch point for all tokens (including special tokens)
            sw_tensor = torch.tensor(switch_points, dtype=torch.float32, device=device)

            all_active_indices = torch.where(mask.bool() & (seq != self.pad_lang_id))[0]
            for pos in all_active_indices:
                pos_val = float(pos.item())
                # Signed distance = current_pos - switch_point
                diffs = pos_val - sw_tensor
                # Find nearest switch point by minimum absolute distance
                min_idx = torch.argmin(torch.abs(diffs))
                signed_dist = diffs[min_idx].item()

                # Clamp to [-max_distance, max_distance]
                clamped_dist = max(-float(self.max_distance), min(float(self.max_distance), signed_dist))
                raw_signed_distances[b, pos] = clamped_dist

                # Shifted index: 1 + (clamped_dist + max_distance) -> in [1, 2*max_dist + 1]
                idx_val = int(clamped_dist + self.max_distance) + 1
                distance_indices[b, pos] = idx_val

        return raw_signed_distances, distance_indices
