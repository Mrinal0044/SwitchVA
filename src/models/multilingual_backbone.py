"""Multilingual Transformer Backbone wrapper supporting HingRoBERTa and mDeBERTa."""

import logging
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger(__name__)


class MultilingualBackbone(nn.Module):
    """Wraps multilingual transformer backbones (HingRoBERTa / mDeBERTa).

    Outputs contextual token representations:
        H in R^(B x N x d)
    """

    def __init__(
        self,
        model_name: str = "l3cube-pune/hing-roberta",
        hidden_dim: int = 768,
        freeze_layers: bool = False,
        use_mock: bool = False,
        vocab_size: int = 50265,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            hidden_dim: Target representation dimension d.
            freeze_layers: Whether to freeze transformer parameters.
            use_mock: If True, uses a lightweight local transformer for fast offline testing.
            vocab_size: Vocabulary size for mock embedding.
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.use_mock = use_mock
        self.freeze_layers = freeze_layers

        if use_mock:
            self.model = None
            self.mock_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=1)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dropout=0.1,
            )
            self.mock_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.proj = nn.Identity()
        else:
            try:
                self.model = AutoModel.from_pretrained(model_name)
                backbone_hidden = self.model.config.hidden_size
            except Exception as e:
                logger.warning(
                    "Could not load '%s' from remote (%s). Initializing from local random configuration.",
                    model_name,
                    e,
                )
                config = AutoConfig.for_model("roberta", hidden_size=hidden_dim, vocab_size=vocab_size)
                self.model = AutoModel.from_config(config)
                backbone_hidden = hidden_dim

            if backbone_hidden != hidden_dim:
                self.proj = nn.Linear(backbone_hidden, hidden_dim)
            else:
                self.proj = nn.Identity()

            if freeze_layers:
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("Frozen backbone parameters for '%s'.", model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) attention mask (1=valid, 0=pad).
            token_type_ids: Optional token type IDs.

        Returns:
            (batch_size, seq_len, hidden_dim) token representations H.
        """
        if self.use_mock:
            x = self.mock_embed(input_ids)
            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            hidden = self.mock_encoder(x, src_key_padding_mask=src_key_padding_mask)
            return hidden

        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if token_type_ids is not None and "mdeberta" not in self.model_name.lower():
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state
        hidden = self.proj(hidden)
        return hidden
