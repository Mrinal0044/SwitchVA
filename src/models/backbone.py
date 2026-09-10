"""Multilingual Transformer Backbone wrapper (HingRoBERTa / mDeBERTa)."""

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class TransformerBackbone(nn.Module):
    """Wraps multilingual pretrained transformer models (HingRoBERTa / mDeBERTa)
    for token-level representations."""

    def __init__(
        self,
        pretrained_model_name: str = "l3cube-pune/hing-roberta",
        hidden_dim: int = 768,
        freeze_backbone: bool = False,
        use_mock: bool = False,
    ):
        """
        Args:
            pretrained_model_name: HuggingFace model identifier.
            hidden_dim: Expected output hidden dimension.
            freeze_backbone: Whether to freeze backbone parameters.
            use_mock: If True, uses a lightweight mock transformer for offline testing.
        """
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.hidden_dim = hidden_dim
        self.use_mock = use_mock

        if use_mock:
            self.model = None
            self.mock_embed = nn.Embedding(50000, hidden_dim)
            n_heads = 8 if hidden_dim % 8 == 0 else (4 if hidden_dim % 4 == 0 else 2)
            self.mock_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True),
                num_layers=2,
                enable_nested_tensor=False,
            )
        else:
            try:
                self.model = AutoModel.from_pretrained(pretrained_model_name)
                backbone_hidden = self.model.config.hidden_size
            except Exception as e:
                logger.warning(
                    "Could not load '%s' from remote (%s). Falling back to randomized config.",
                    pretrained_model_name,
                    e,
                )
                config = AutoConfig.for_model("roberta", hidden_size=hidden_dim)
                self.model = AutoModel.from_config(config)
                backbone_hidden = hidden_dim

            if backbone_hidden != hidden_dim:
                self.proj = nn.Linear(backbone_hidden, hidden_dim)
            else:
                self.proj = nn.Identity()

            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("Transformer backbone parameters frozen.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask.
            token_type_ids: Optional segment IDs.

        Returns:
            (batch_size, seq_len, hidden_dim) token-level representation tensor.
        """
        if self.use_mock:
            x = self.mock_embed(input_ids)
            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.mock_encoder(x, src_key_padding_mask=src_key_padding_mask)
            return hidden

        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None and "mdeberta" not in self.pretrained_model_name.lower():
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state
        hidden = self.proj(hidden)
        return hidden
