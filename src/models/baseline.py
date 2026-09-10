"""Switch-Unaware Baseline Model for Hinglish DimABSA.

Architecture:
- Multilingual Transformer Backbone (HingRoBERTa / mDeBERTa)
- Standard contextual representation H_ctx (WITHOUT signed switch distance,
  WITHOUT switch embeddings, WITHOUT SP-GSA, and WITHOUT switch graph edges)
- Biaffine aspect/opinion span extraction directly on H_ctx
- Dual Valence and Arousal MLP regression heads on concatenated span representations

This isolates and benchmarks the exact contribution of switch-aware modeling.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .backbone import TransformerBackbone
from .biaffine_span import BiaffineSpanExtractor
from .regression import DualRegressionHeads
from .va_regression import DimensionalVARegression


class SwitchUnawareBaseline(nn.Module):
    """Switch-Unaware Baseline Model for DimABSA."""

    def __init__(
        self,
        pretrained_backbone: str = "l3cube-pune/hing-roberta",
        hidden_dim: int = 768,
        span_mlp_dim: int = 256,
        regression_hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_mock_backbone: bool = False,
        config: Optional[Any] = None,
    ):
        super().__init__()

        if config is not None:
            model_cfg = getattr(config, "model", config)
            bb_cfg = getattr(model_cfg, "backbone", {})
            span_cfg = getattr(model_cfg, "biaffine_span", {})
            reg_cfg = getattr(model_cfg, "regression_heads", {})

            pretrained_backbone = getattr(bb_cfg, "pretrained_model_name", pretrained_backbone)
            hidden_dim = getattr(bb_cfg, "hidden_dim", hidden_dim)
            freeze_backbone = getattr(bb_cfg, "freeze_backbone", freeze_backbone)
            span_mlp_dim = getattr(span_cfg, "mlp_dim", span_mlp_dim)
            regression_hidden_dim = getattr(reg_cfg, "hidden_dim", regression_hidden_dim)

        self.hidden_dim = hidden_dim

        # 1. Standard Multilingual Transformer Backbone (no switch embeddings added)
        self.backbone = TransformerBackbone(
            pretrained_model_name=pretrained_backbone,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_mock=use_mock_backbone,
        )

        # 2. Standard Biaffine Span Extractor directly on contextual representations
        self.biaffine_span = BiaffineSpanExtractor(
            input_dim=hidden_dim,
            mlp_dim=span_mlp_dim,
            num_labels=3,
            dropout=dropout,
        )

        # 3. Span representation fusion: concatenates aspect & opinion representations
        # Input dimension: 2 * hidden_dim
        self.span_fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LayerNorm(2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 4. Continuous Valence and Arousal Regression Heads
        self.va_regression = DimensionalVARegression(
            input_dim=2 * hidden_dim,
            hidden_dim=regression_hidden_dim,
            dropout=dropout,
        )
        self.regression_heads = self.va_regression.dual_heads

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        aspect_spans: Optional[torch.Tensor] = None,
        opinion_spans: Optional[torch.Tensor] = None,
        tokens_batch: Optional[List[List[str]]] = None,
        aspect_texts_batch: Optional[List[List[str]]] = None,
        opinion_texts_batch: Optional[List[List[str]]] = None,
        sentence_ids: Optional[List[Union[str, int]]] = None,
        sentences: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = 5,
        top_k_aspect: Optional[int] = None,
        top_k_opinion: Optional[int] = None,
        score_floor: Optional[float] = None,
        strategy: str = "ranked",
        **kwargs: Any,  # Safely ignore any switch-related arguments (lang_ids, switch_dist, etc.)
    ) -> Dict[str, Any]:
        """Forward pass of the switch-unaware baseline.

        Note: kwargs like lang_ids and switch_dist are explicitly ignored to ensure
        the baseline remains switch-unaware.
        """
        # 1. Standard Transformer contextual encoding
        ctx_hidden = self.backbone(input_ids, attention_mask=attention_mask)

        # 2. Biaffine aspect/opinion span extraction
        span_outputs = self.biaffine_span(
            ctx_hidden,
            attention_mask=attention_mask,
            aspect_spans=aspect_spans,
            opinion_spans=opinion_spans,
            threshold=threshold,
            top_k=top_k,
            top_k_aspect=top_k_aspect,
            top_k_opinion=top_k_opinion,
            score_floor=score_floor,
            strategy=strategy,
        )
        aspect_span_repr = span_outputs["aspect_repr"]
        opinion_span_repr = span_outputs["opinion_repr"]
        span_logits = span_outputs["span_logits"]

        # 3. Concatenate aspect and opinion vectors (2 * hidden_dim)
        combined_repr = torch.cat([aspect_span_repr, opinion_span_repr], dim=-1)
        z_base = self.span_fuse(combined_repr)

        # 4. Continuous Valence and Arousal Regression
        va_out = self.va_regression(z_base)
        valence = va_out["valence"]
        arousal = va_out["arousal"]

        out_dict: Dict[str, Any] = {
            "valence": valence,
            "arousal": arousal,
            "fused_representation": z_base,
            "z": z_base,
            "span_logits": span_logits,
            "aspect_span_repr": aspect_span_repr,
            "opinion_span_repr": opinion_span_repr,
            "contextual_hidden": ctx_hidden,
            "aspect_scores": span_outputs.get("aspect_scores"),
            "opinion_scores": span_outputs.get("opinion_scores"),
            "masked_aspect_scores": span_outputs.get("masked_aspect_scores"),
            "masked_opinion_scores": span_outputs.get("masked_opinion_scores"),
            "valid_span_mask": span_outputs.get("valid_span_mask"),
            "decoded_aspect_spans": span_outputs.get("decoded_aspect_spans", []),
            "decoded_opinion_spans": span_outputs.get("decoded_opinion_spans", []),
        }

        # Format structured predictions if sentences provided
        if sentences is not None:
            asp_s_list = aspect_spans.tolist() if (aspect_spans is not None and isinstance(aspect_spans, torch.Tensor)) else None
            op_s_list = opinion_spans.tolist() if (opinion_spans is not None and isinstance(opinion_spans, torch.Tensor)) else None
            out_dict["structured_predictions"] = DimensionalVARegression.format_structured_predictions(
                sentence_ids=sentence_ids or [f"sent_{i}" for i in range(len(sentences))],
                sentences=sentences,
                valence_preds=valence,
                arousal_preds=arousal,
                aspect_spans_batch=asp_s_list,
                opinion_spans_batch=op_s_list,
                aspect_texts_batch=aspect_texts_batch,
                opinion_texts_batch=opinion_texts_batch,
            )

        return out_dict
