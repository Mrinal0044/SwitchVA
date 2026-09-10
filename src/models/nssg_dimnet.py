"""NSSG-DimNet: Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA.

Full approved end-to-end architecture (Phases 1-5):
1. Input: Token IDs, Language IDs, Signed Switch-Distance
2. Switch Embedding + Multilingual Transformer Backbone (HingRoBERTa / mDeBERTa)
3. Switch-Point Gated Self-Attention (SP-GSA) -> H_gated in R^(B x N x d)
4. Dual Branch:
   - Branch 1: Biaffine Aspect-Opinion Span Extractor -> (H_aspect, H_opinion, S(i,j))
   - Branch 2: Heterogeneous Neuro-Symbolic Graph (H-NSG) + RGAT with NRC-VAD priors -> (G_aspect, G_opinion, H_graph)
5. Aspect-Guided Mutual Cross-Attention & Fusion -> Fused Span Representation Z in R^(B x 3d)
6. Independent Valence MLP and Arousal MLP regression heads -> (V_hat, A_hat) in [0, 1]
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .backbone import TransformerBackbone
from .switch_embedding import SwitchEmbedding
from .sp_gsa import SPGSA
from .biaffine_span import BiaffineSpanExtractor, BiaffineAspectOpinionSpanExtractor
from .hnsg import HeterogeneousNeuroSymbolicGraph, HNSGBuilder
from .rgat import RGAT, extract_graph_span_representation
from .cross_attention import AspectGuidedMutualCrossAttention
from .fusion import AspectGuidedFusion
from .regression import DualRegressionHeads, ValenceRegressionHead, ArousalRegressionHead
from .va_regression import DimensionalVARegression


class NSSGDimNet(nn.Module):
    """Neuro-Symbolic Switch-Gated Dual-Graph Network (NSSG-DimNet) for Hinglish DimABSA."""

    def __init__(
        self,
        config: Optional[Any] = None,
        pretrained_backbone: str = "l3cube-pune/hing-roberta",
        hidden_dim: int = 768,
        num_languages: int = 4,
        max_signed_distance: int = 64,
        num_spgsa_heads: int = 8,
        span_mlp_dim: int = 256,
        nrc_vad_dim: int = 2,
        num_relations: int = 3,
        rgat_layers: int = 2,
        rgat_heads: int = 4,
        cross_attn_heads: int = 8,
        regression_hidden_dim: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_mock_backbone: bool = False,
    ):
        super().__init__()

        # If config is provided, extract hyperparameters from it
        if config is not None:
            model_cfg = getattr(config, "model", config)
            bb_cfg = getattr(model_cfg, "backbone", {})
            sw_cfg = getattr(model_cfg, "switch_embedding", {})
            sp_cfg = getattr(model_cfg, "sp_gsa", {})
            bi_cfg = getattr(model_cfg, "biaffine_span", {})
            hnsg_cfg = getattr(model_cfg, "hnsg", {})
            rgat_cfg = getattr(hnsg_cfg, "rgat", {})
            ca_cfg = getattr(model_cfg, "cross_attention", {})
            reg_cfg = getattr(model_cfg, "regression_heads", {})

            pretrained_backbone = getattr(bb_cfg, "pretrained_model_name", pretrained_backbone)
            hidden_dim = getattr(bb_cfg, "hidden_dim", hidden_dim)
            freeze_backbone = getattr(bb_cfg, "freeze_backbone", freeze_backbone)

            num_languages = getattr(sw_cfg, "num_languages", num_languages)
            max_signed_distance = getattr(sw_cfg, "max_signed_distance", max_signed_distance)

            num_spgsa_heads = getattr(sp_cfg, "num_heads", num_spgsa_heads)
            span_mlp_dim = getattr(bi_cfg, "mlp_dim", span_mlp_dim)

            nrc_vad_dim = getattr(hnsg_cfg, "nrc_vad_dim", nrc_vad_dim)
            num_relations = getattr(hnsg_cfg, "num_relations", num_relations)
            rgat_layers = getattr(rgat_cfg, "num_layers", rgat_layers)
            rgat_heads = getattr(rgat_cfg, "num_heads", rgat_heads)

            cross_attn_heads = getattr(ca_cfg, "num_heads", cross_attn_heads)
            regression_hidden_dim = getattr(reg_cfg, "hidden_dim", regression_hidden_dim)

        self.hidden_dim = hidden_dim

        # 1. Multilingual Transformer Backbone
        self.backbone = TransformerBackbone(
            pretrained_model_name=pretrained_backbone,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            use_mock=use_mock_backbone,
        )

        # 2. Switch Embedding (Language ID + Signed Switch-Distance)
        self.switch_embedding = SwitchEmbedding(
            num_languages=num_languages,
            lang_embed_dim=64,
            max_signed_distance=max_signed_distance,
            dist_embed_dim=64,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # 3. Switch-Point Gated Self-Attention (SP-GSA)
        self.sp_gsa = SPGSA(
            hidden_dim=hidden_dim,
            num_heads=num_spgsa_heads,
            dropout=dropout,
        )

        # 4. Branch 1: Biaffine Aspect-Opinion Span Extractor
        self.biaffine_span = BiaffineSpanExtractor(
            input_dim=hidden_dim,
            mlp_dim=span_mlp_dim,
            num_labels=3,
            dropout=dropout,
        )

        # 5. Branch 2: Heterogeneous Neuro-Symbolic Graph + RGAT
        self.hnsg_builder = HeterogeneousNeuroSymbolicGraph(
            node_dim=hidden_dim,
            vad_dim=nrc_vad_dim,
            num_relations=num_relations,
        )
        self.rgat = RGAT(
            hidden_dim=hidden_dim,
            num_layers=rgat_layers,
            num_heads=rgat_heads,
            num_relations=num_relations,
            dropout=dropout,
            use_residual=True,
        )

        # 6. Aspect-Guided Representation Fusion & Mutual Cross-Attention (Z in R^(3d))
        self.fusion = AspectGuidedFusion(
            hidden_dim=hidden_dim,
            num_heads=cross_attn_heads,
            dropout=dropout,
        )
        # Retain direct access to cross_attention for backward compatibility
        self.cross_attention = self.fusion.mutual_cross_attention

        # 7. Independent Valence and Arousal MLP Regression Heads
        fused_dim = 3 * hidden_dim
        self.va_regression = DimensionalVARegression(
            input_dim=fused_dim,
            hidden_dim=regression_hidden_dim,
            dropout=dropout,
        )
        self.regression_heads = self.va_regression.dual_heads
        self.valence_head = self.va_regression.dual_heads.valence_head
        self.arousal_head = self.va_regression.dual_heads.arousal_head

    def forward(
        self,
        input_ids: torch.Tensor,
        lang_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        switch_dist: Optional[torch.Tensor] = None,
        aspect_spans: Optional[torch.Tensor] = None,
        opinion_spans: Optional[torch.Tensor] = None,
        nrc_vad_priors: Optional[torch.Tensor] = None,
        adj_matrices: Optional[torch.Tensor] = None,
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
    ) -> Dict[str, Any]:
        """Complete forward pass of NSSG-DimNet.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            lang_ids: (batch_size, seq_len) language tags.
            attention_mask: (batch_size, seq_len) attention mask.
            switch_dist: Optional (batch_size, seq_len) precomputed switch distance indices.
            aspect_spans: Optional (batch_size, num_pairs, 2) or (batch_size, 2) aspect indices.
            opinion_spans: Optional (batch_size, num_pairs, 2) or (batch_size, 2) opinion indices.
            nrc_vad_priors: Optional (batch_size, seq_len, 2) affective priors.
            adj_matrices: Optional (batch_size, 3, seq_len, seq_len) precomputed graph edges.
            tokens_batch: Optional list of token strings.
            aspect_texts_batch: Optional list of aspect strings.
            opinion_texts_batch: Optional list of opinion strings.
            sentence_ids: Optional list of sentence IDs.
            sentences: Optional list of sentence texts.
            threshold: Optional logit threshold for threshold-based span decoding.
            top_k: Default max candidate spans to decode per sequence.
            top_k_aspect: Optional separate top-k for aspect spans.
            top_k_opinion: Optional separate top-k for opinion spans.
            score_floor: Optional score floor applied after ranking.
            strategy: Decoding strategy ("ranked" or "threshold").

        Returns:
            Dictionary containing valence, arousal, representations, span logits, and decoded spans.
        """
        batch_size, seq_len = input_ids.shape

        # 1. Switch Embedding
        switch_embeds = self.switch_embedding(lang_ids, distance_indices=switch_dist, attention_mask=attention_mask)

        # 2. Multilingual Transformer Backbone
        backbone_hidden = self.backbone(input_ids, attention_mask=attention_mask)

        # 3. Switch-Point Gated Self-Attention (SP-GSA)
        spgsa_hidden, G = self.sp_gsa(backbone_hidden, switch_embeds, attention_mask=attention_mask)

        # 4. Branch 1: Biaffine Span Extractor
        span_outputs = self.biaffine_span(
            spgsa_hidden,
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

        # 5. Branch 2: Heterogeneous Neuro-Symbolic Graph + RGAT
        hnsg_out = self.hnsg_builder(
            H_gated=spgsa_hidden,
            tokens_batch=tokens_batch,
            lang_ids_batch=lang_ids.tolist() if isinstance(lang_ids, torch.Tensor) else lang_ids,
            attention_mask=attention_mask,
            nrc_vad_priors=nrc_vad_priors,
            adj_matrices=adj_matrices,
        )
        node_features = hnsg_out["node_features"] if isinstance(hnsg_out, dict) else hnsg_out[0]
        graph_adj = hnsg_out["relational_adj"] if isinstance(hnsg_out, dict) else (hnsg_out.relational_adj if hasattr(hnsg_out, "relational_adj") else hnsg_out[1])

        rgat_out = self.rgat(
            node_features=node_features,
            adj_matrices=graph_adj,
            attention_mask=attention_mask,
            aspect_spans=aspect_spans,
            opinion_spans=opinion_spans,
            seq_len=seq_len,
        )
        if isinstance(rgat_out, tuple):
            if len(rgat_out) == 3:
                graph_nodes, graph_aspect_repr, graph_opinion_repr = rgat_out
            elif len(rgat_out) == 2:
                graph_nodes, graph_aspect_repr = rgat_out
                graph_opinion_repr = graph_aspect_repr
            else:
                graph_nodes = rgat_out[0]
                graph_aspect_repr = aspect_span_repr
                graph_opinion_repr = opinion_span_repr
        else:
            graph_nodes = rgat_out
            graph_aspect_repr = aspect_span_repr
            graph_opinion_repr = opinion_span_repr

        # 6. Aspect-Guided Representation Fusion (Z in R^(3d))
        fusion_out = self.fusion(
            aspect_span_repr=aspect_span_repr,
            opinion_span_repr=opinion_span_repr,
            graph_aspect_repr=graph_aspect_repr,
            graph_opinion_repr=graph_opinion_repr,
            graph_node_features=graph_nodes,
            graph_mask=attention_mask,
        )
        Z = fusion_out["z"]

        # 7. Independent Valence and Arousal MLP Regression Heads
        va_out = self.va_regression(Z)
        valence = va_out["valence"]
        arousal = va_out["arousal"]

        out_dict: Dict[str, Any] = {
            "valence": valence,
            "arousal": arousal,
            "fused_representation": Z,
            "z": Z,
            "z_span": fusion_out["z_span"],
            "z_graph": fusion_out["z_graph"],
            "z_cross": fusion_out["z_cross"],
            "span_logits": span_logits,
            "cross_attn_weights": fusion_out["cross_attn_weights"],
            "aspect_span_repr": aspect_span_repr,
            "opinion_span_repr": opinion_span_repr,
            "graph_aspect_repr": graph_aspect_repr,
            "graph_opinion_repr": graph_opinion_repr,
            "spgsa_hidden": spgsa_hidden,
            "gate_values": G,
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
