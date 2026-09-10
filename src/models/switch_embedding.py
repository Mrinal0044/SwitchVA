"""Switch Embedding module fusing token-level language IDs and signed switch distances."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .signed_switch_distance import SignedSwitchDistance


def compute_signed_switch_distance(
    lang_ids: torch.Tensor,
    pad_id: int = 4,
    max_dist: int = 32,
) -> torch.Tensor:
    """Convenience wrapper around SignedSwitchDistance for shifted distance index lookup."""
    calc = SignedSwitchDistance(max_distance=max_dist, pad_lang_id=pad_id)
    _, dist_indices = calc(lang_ids)
    return dist_indices


class SwitchEmbedding(nn.Module):
    """Trainable Switch Embedding module fusing language identity and signed switch distance."""

    def __init__(
        self,
        num_languages: int = 5,  # HI=0, EN=1, OTHER=2, SPECIAL=3, PAD=4
        lang_embed_dim: int = 64,
        max_signed_distance: int = 32,
        dist_embed_dim: int = 64,
        output_dim: int = 768,
        dropout: float = 0.1,
        pad_lang_id: int = 4,
    ):
        """
        Args:
            num_languages: Number of distinct language categories.
            lang_embed_dim: Dimension of language identification embedding.
            max_signed_distance: Maximum distance for signed distance embedding (D_max).
            dist_embed_dim: Dimension of switch distance embedding.
            output_dim: Output projection dimension (matches backbone hidden_dim d).
            dropout: Dropout probability.
            pad_lang_id: Language ID for padding tokens.
        """
        super().__init__()
        self.max_signed_distance = max_signed_distance
        self.output_dim = output_dim
        self.pad_lang_id = pad_lang_id

        self.distance_calculator = SignedSwitchDistance(
            max_distance=max_signed_distance,
            special_lang_id=3,
            pad_lang_id=pad_lang_id,
        )

        num_dist_embeddings = self.distance_calculator.num_embeddings
        dist_pad_idx = self.distance_calculator.pad_idx

        num_lang_embeds = max(num_languages, (pad_lang_id + 1) if pad_lang_id is not None else 0)
        self.lang_embedding = nn.Embedding(
            num_lang_embeds,
            lang_embed_dim,
            padding_idx=pad_lang_id if pad_lang_id is not None and pad_lang_id < num_lang_embeds else None,
        )
        self.dist_embedding = nn.Embedding(
            num_dist_embeddings,
            dist_embed_dim,
            padding_idx=dist_pad_idx,
        )

        combined_dim = lang_embed_dim + dist_embed_dim
        self.proj = nn.Linear(combined_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lang_ids: torch.Tensor,
        distance_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lang_ids: (batch_size, seq_len) numeric language IDs.
            distance_indices: Optional precomputed switch distance indices.
            attention_mask: Optional attention mask (1=valid, 0=pad).

        Returns:
            (batch_size, seq_len, output_dim) switch embeddings E_switch.
        """
        if distance_indices is None:
            _, distance_indices = self.distance_calculator(lang_ids, attention_mask=attention_mask)

        e_lang = self.lang_embedding(lang_ids)
        e_dist = self.dist_embedding(distance_indices)

        e_combined = torch.cat([e_lang, e_dist], dim=-1)
        e_switch = self.proj(e_combined)
        e_switch = self.layer_norm(e_switch)
        e_switch = self.dropout(e_switch)

        # Zero out padding positions if attention mask provided
        if attention_mask is not None:
            e_switch = e_switch * attention_mask.unsqueeze(-1).float()

        return e_switch
