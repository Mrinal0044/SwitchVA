"""Integrated Switch-Aware Encoder and diagnostic inspection utilities (Phase 2)."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .signed_switch_distance import SignedSwitchDistance
from .switch_embedding import SwitchEmbedding
from .multilingual_backbone import MultilingualBackbone
from .sp_gsa import SPGSA


def inspect_switch_gates(
    tokens: List[str],
    language_ids: List[str],
    signed_distances: Union[List[float], torch.Tensor],
    gate_values: Union[torch.Tensor, List[float]],
) -> str:
    """Format a diagnostic table of token-level language, switch distance, and gate magnitudes.

    Args:
        tokens: Subwords or word tokens.
        language_ids: Language ID strings matching tokens.
        signed_distances: Signed switch distance values.
        gate_values: Gate tensor of shape (seq_len, d) or 1D mean gate values.

    Returns:
        Formatted string diagnostic table.
    """
    if isinstance(signed_distances, torch.Tensor):
        signed_distances = signed_distances.detach().cpu().view(-1).tolist()

    if isinstance(gate_values, torch.Tensor):
        if gate_values.dim() == 2:
            # Mean gate magnitude across hidden dimension d
            gate_mags = gate_values.mean(dim=-1).detach().cpu().view(-1).tolist()
        else:
            gate_mags = gate_values.detach().cpu().view(-1).tolist()
    else:
        gate_mags = list(gate_values)

    lines = [
        f"{'Token':<20} {'Language':<12} {'SwitchDist':<14} {'MeanGate':<10}",
        "-" * 58,
    ]

    for tok, lang, dist, gate in zip(tokens, language_ids, signed_distances, gate_mags):
        dist_str = f"{int(dist):+d}" if not isinstance(dist, str) else str(dist)
        lines.append(f"{tok:<20} {lang:<12} {dist_str:<14} {gate:<10.4f}")

    return "\n".join(lines)


class SwitchAwareEncoder(nn.Module):
    """Switch-Aware Encoder portion of NSSG-DimNet (Phase 2).

    Encapsulates:
    1. SignedSwitchDistance
    2. SwitchEmbedding
    3. MultilingualBackbone (HingRoBERTa / mDeBERTa)
    4. Switch-Point Gated Self-Attention (SP-GSA)
    """

    def __init__(
        self,
        backbone_name: str = "l3cube-pune/hing-roberta",
        hidden_dim: int = 768,
        max_signed_distance: int = 32,
        num_languages: int = 5,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_mock_backbone: bool = False,
    ):
        """
        Args:
            backbone_name: HuggingFace model name.
            hidden_dim: Transformer and switch embedding hidden dimension d.
            max_signed_distance: Maximum distance for signed distance clipping.
            num_languages: Number of language categories (HI, EN, OTHER, SPECIAL, PAD).
            dropout: Dropout probability.
            freeze_backbone: Whether to freeze transformer layers.
            use_mock_backbone: If True, uses local mock transformer for offline testing.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_signed_distance = max_signed_distance

        # 1. Multilingual Backbone
        self.backbone = MultilingualBackbone(
            model_name=backbone_name,
            hidden_dim=hidden_dim,
            freeze_layers=freeze_backbone,
            use_mock=use_mock_backbone,
        )

        # 2. Switch Distance Calculator & Trainable Embedding
        self.distance_calculator = SignedSwitchDistance(
            max_distance=max_signed_distance,
            special_lang_id=3,
            pad_lang_id=4,
        )
        self.switch_embedding = SwitchEmbedding(
            num_languages=num_languages,
            lang_embed_dim=64,
            max_signed_distance=max_signed_distance,
            dist_embed_dim=64,
            output_dim=hidden_dim,
            dropout=dropout,
            pad_lang_id=4,
        )

        # 3. SP-GSA
        self.sp_gsa = SPGSA(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        lang_ids_numeric: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            lang_ids_numeric: (batch_size, seq_len) language IDs.
            attention_mask: (batch_size, seq_len) attention mask.

        Returns:
            Dictionary containing:
            - H_gated: (batch_size, seq_len, hidden_dim) output of SP-GSA
            - H: (batch_size, seq_len, hidden_dim) raw backbone output
            - E_switch: (batch_size, seq_len, hidden_dim) switch embeddings
            - gate_values: (batch_size, seq_len, hidden_dim) G tensor in [0, 1]
            - signed_distances: (batch_size, seq_len) raw distance values in [-D_max, D_max]
            - distance_indices: (batch_size, seq_len) shifted lookup indices
        """
        # 1. Compute Signed Switch Distance
        signed_dists, dist_indices = self.distance_calculator(
            lang_ids_numeric,
            attention_mask=attention_mask,
        )

        # 2. Compute Switch Embedding E_switch
        E_switch = self.switch_embedding(
            lang_ids_numeric,
            distance_indices=dist_indices,
            attention_mask=attention_mask,
        )

        # 3. Compute Transformer Backbone Output H
        H = self.backbone(input_ids, attention_mask=attention_mask)

        # 4. Compute SP-GSA
        H_gated, G = self.sp_gsa(H, E_switch, attention_mask=attention_mask)

        return {
            "H_gated": H_gated,
            "H": H,
            "E_switch": E_switch,
            "gate_values": G,
            "signed_distances": signed_dists,
            "distance_indices": dist_indices,
        }
