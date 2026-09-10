"""Dimensional Valence and Arousal (VA) Regression and Structured Output Formatter (Phase 5)."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .regression import DualRegressionHeads, ValenceRegressionHead, ArousalRegressionHead


class DimensionalVARegression(nn.Module):
    """Dimensional Valence and Arousal Regression Module.

    Takes the fused representation Z in R^(3d) and predicts continuous Valence
    and Arousal values in [0, 1] using independent parameter heads.
    """

    def __init__(
        self,
        input_dim: int = 2304,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dual_heads = DualRegressionHeads(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute continuous Valence and Arousal predictions from Z.

        Args:
            z: (B, 3d) or (B, K, 3d) fused representation tensor.

        Returns:
            Dictionary with:
            - 'valence': Continuous valence score(s) in [0, 1]
            - 'arousal': Continuous arousal score(s) in [0, 1]
        """
        valence, arousal = self.dual_heads(z)
        return {
            "valence": valence,
            "arousal": arousal,
        }

    @staticmethod
    def format_structured_predictions(
        sentence_ids: List[Union[str, int]],
        sentences: List[str],
        valence_preds: torch.Tensor,
        arousal_preds: torch.Tensor,
        aspect_spans_batch: Optional[List[List[Tuple[int, int]]]] = None,
        opinion_spans_batch: Optional[List[List[Tuple[int, int]]]] = None,
        aspect_texts_batch: Optional[List[List[str]]] = None,
        opinion_texts_batch: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Format model predictions into the standardized DimABSA prediction schema.

        Schema:
        {
            "sentence_id": ...,
            "sentence": "...",
            "predictions": [
                {
                    "aspect": "...",
                    "opinion": "...",
                    "aspect_span": [s, e],
                    "opinion_span": [s, e],
                    "valence": 0.xxx,
                    "arousal": 0.xxx
                }
            ]
        }
        """
        formatted_outputs = []
        batch_size = len(sentences)

        v_cpu = valence_preds.detach().cpu()
        a_cpu = arousal_preds.detach().cpu()

        for b in range(batch_size):
            s_id = sentence_ids[b] if b < len(sentence_ids) else f"sent_{b}"
            s_text = sentences[b]

            preds_list = []
            v_b = v_cpu[b]
            a_b = a_cpu[b]

            # Determine number of pairs for sentence b
            if v_b.dim() == 0 or (v_b.dim() == 1 and v_b.size(0) == 1):
                num_p = 1
                v_vals = [float(v_b.item())]
                a_vals = [float(a_b.item())]
            else:
                num_p = v_b.size(0)
                v_vals = [float(val.item()) for val in v_b.view(-1)]
                a_vals = [float(val.item()) for val in a_b.view(-1)]

            asp_spans = aspect_spans_batch[b] if (aspect_spans_batch and b < len(aspect_spans_batch)) else []
            op_spans = opinion_spans_batch[b] if (opinion_spans_batch and b < len(opinion_spans_batch)) else []
            asp_texts = aspect_texts_batch[b] if (aspect_texts_batch and b < len(aspect_texts_batch)) else []
            op_texts = opinion_texts_batch[b] if (opinion_texts_batch and b < len(opinion_texts_batch)) else []

            for k in range(num_p):
                a_val = float(a_vals[k]) if k < len(a_vals) else 0.5
                v_val = float(v_vals[k]) if k < len(v_vals) else 0.5

                # Handle 1D/2D/3D span list nesting
                if isinstance(asp_spans, list) and len(asp_spans) > 0:
                    if isinstance(asp_spans[0], (int, float)):
                        asp_span = [int(asp_spans[0]), int(asp_spans[1])] if len(asp_spans) >= 2 else [0, 0]
                    elif k < len(asp_spans) and isinstance(asp_spans[k], (list, tuple)):
                        asp_span = [int(asp_spans[k][0]), int(asp_spans[k][1])] if len(asp_spans[k]) >= 2 else [0, 0]
                    else:
                        asp_span = [0, 0]
                else:
                    asp_span = [0, 0]

                if isinstance(op_spans, list) and len(op_spans) > 0:
                    if isinstance(op_spans[0], (int, float)):
                        op_span = [int(op_spans[0]), int(op_spans[1])] if len(op_spans) >= 2 else [0, 0]
                    elif k < len(op_spans) and isinstance(op_spans[k], (list, tuple)):
                        op_span = [int(op_spans[k][0]), int(op_spans[k][1])] if len(op_spans[k]) >= 2 else [0, 0]
                    else:
                        op_span = [0, 0]
                else:
                    op_span = [0, 0]

                asp_text = asp_texts[k] if (isinstance(asp_texts, list) and k < len(asp_texts)) else (asp_texts if isinstance(asp_texts, str) else "")
                op_text = op_texts[k] if (isinstance(op_texts, list) and k < len(op_texts)) else (op_texts if isinstance(op_texts, str) else "")

                preds_list.append({
                    "aspect": str(asp_text),
                    "opinion": str(op_text),
                    "aspect_span": asp_span,
                    "opinion_span": op_span,
                    "valence": round(v_val, 4),
                    "arousal": round(a_val, 4),
                })

            formatted_outputs.append({
                "sentence_id": s_id,
                "sentence": s_text,
                "predictions": preds_list,
            })

        return formatted_outputs
