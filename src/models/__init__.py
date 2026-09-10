"""Model architecture modules for NSSG-DimNet."""

from .backbone import TransformerBackbone
from .multilingual_backbone import MultilingualBackbone
from .signed_switch_distance import SignedSwitchDistance
from .switch_embedding import SwitchEmbedding
from .sp_gsa import SPGSA
from .switch_encoder import SwitchAwareEncoder, inspect_switch_gates
from .biaffine_span import (
    BiaffineScorer,
    BiaffineAspectOpinionSpanExtractor,
    BiaffineSpanExtractor,
)
from .hnsg import (
    HeterogeneousNeuroSymbolicGraph,
    HNSGBuilder,
    REL_SYNTACTIC,
    REL_SWITCH,
    REL_ASPECT_OPINION,
    NUM_RELATIONS,
    RELATION_NAMES,
)
from .rgat import RGAT, RGATLayer, extract_graph_span_representation
from .cross_attention import AspectGuidedMutualCrossAttention
from .fusion import AspectGuidedFusion
from .regression import (
    DimensionRegressionHead,
    DualRegressionHeads,
    ValenceRegressionHead,
    ArousalRegressionHead,
)
from .va_regression import DimensionalVARegression
from .nssg_dimnet import NSSGDimNet
from .baseline import SwitchUnawareBaseline

__all__ = [
    "TransformerBackbone",
    "MultilingualBackbone",
    "SignedSwitchDistance",
    "SwitchEmbedding",
    "SPGSA",
    "SwitchAwareEncoder",
    "inspect_switch_gates",
    "BiaffineScorer",
    "BiaffineAspectOpinionSpanExtractor",
    "BiaffineSpanExtractor",
    "HeterogeneousNeuroSymbolicGraph",
    "HNSGBuilder",
    "REL_SYNTACTIC",
    "REL_SWITCH",
    "REL_ASPECT_OPINION",
    "NUM_RELATIONS",
    "RELATION_NAMES",
    "RGAT",
    "RGATLayer",
    "extract_graph_span_representation",
    "AspectGuidedMutualCrossAttention",
    "AspectGuidedFusion",
    "DimensionRegressionHead",
    "DualRegressionHeads",
    "ValenceRegressionHead",
    "ArousalRegressionHead",
    "DimensionalVARegression",
    "NSSGDimNet",
    "SwitchUnawareBaseline",
]
