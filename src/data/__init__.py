"""Data loading, parsing, validation, alignment, and dataset classes for DimABSA."""

from .loader import load_raw_dataset, EXPECTED_COLUMNS
from .parser import (
    parse_brace_list,
    parse_code_switch,
    encode_language_ids,
    decode_language_ids,
    parse_sentence_record,
    DEFAULT_LANG_MAP,
)
from .validator import (
    validate_sentence_record,
    check_token_consistency,
    ValidationError,
)
from .aligner import (
    align_span_to_tokens,
    normalize_token,
)
from .splitter import split_dataset
from .dataset import DimABSADataset, dimabsa_collate_fn
from .tokenizer_alignment import (
    SubwordAligner,
    SUBWORD_LANG_MAP,
    REVERSE_SUBWORD_LANG_MAP,
    map_gold_spans_to_subwords,
)
from .nrc_vad import NRCVADLexicon, EMBEDDED_NRC_VAD
from .dependency_parser import (
    DependencyParserInterface,
    HeuristicDependencyBuilder,
)

__all__ = [
    "load_raw_dataset",
    "EXPECTED_COLUMNS",
    "parse_brace_list",
    "parse_code_switch",
    "encode_language_ids",
    "decode_language_ids",
    "parse_sentence_record",
    "DEFAULT_LANG_MAP",
    "validate_sentence_record",
    "check_token_consistency",
    "ValidationError",
    "align_span_to_tokens",
    "normalize_token",
    "split_dataset",
    "DimABSADataset",
    "dimabsa_collate_fn",
    "SubwordAligner",
    "SUBWORD_LANG_MAP",
    "REVERSE_SUBWORD_LANG_MAP",
    "map_gold_spans_to_subwords",
    "NRCVADLexicon",
    "EMBEDDED_NRC_VAD",
    "DependencyParserInterface",
    "HeuristicDependencyBuilder",
]
