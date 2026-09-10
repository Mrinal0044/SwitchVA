"""Annotation and code-switch parsing utilities for DimABSA."""

from typing import Any, Dict, List, Optional, Tuple

DEFAULT_LANG_MAP = {
    "HI": 0,
    "EN": 1,
    "OTHER": 2,
    "PAD": 3,
}

REVERSE_LANG_MAP = {v: k for k, v in DEFAULT_LANG_MAP.items()}


def parse_brace_list(s: Any) -> List[str]:
    """Parse brace-enclosed semicolon-separated list (e.g. '{a; b; c}').

    Args:
        s: Raw string from CSV annotation column.

    Returns:
        List of stripped string values.
    """
    if s is None:
        return []

    text = str(s).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()

    if not text:
        return []

    return [item.strip() for item in text.split(";") if item.strip()]


def parse_code_switch(cs_str: Any) -> Tuple[List[str], List[str]]:
    """Parse code_switch column containing space-separated token:LANG pairs.

    Args:
        cs_str: e.g. 'sanatan:HI 0809:HI tiding:EN luna:EN'

    Returns:
        Tuple of (tokens, language_ids).
    """
    if cs_str is None:
        return [], []

    tokens: List[str] = []
    lang_ids: List[str] = []

    parts = str(cs_str).strip().split()
    for part in parts:
        if ":" in part:
            tok, lang = part.rsplit(":", 1)
            tokens.append(tok)
            lang_ids.append(lang.upper())
        else:
            tokens.append(part)
            lang_ids.append("OTHER")

    return tokens, lang_ids


def encode_language_ids(
    lang_ids: List[str],
    mapping: Optional[Dict[str, int]] = None,
) -> List[int]:
    """Convert string language identifiers (HI, EN, etc.) to numeric IDs.

    Args:
        lang_ids: List of language tag strings.
        mapping: Optional custom mapping dictionary.

    Returns:
        List of integer language IDs.
    """
    map_dict = mapping or DEFAULT_LANG_MAP
    other_val = map_dict.get("OTHER", 2)
    return [map_dict.get(lang.upper(), other_val) for lang in lang_ids]


def decode_language_ids(
    numeric_ids: List[int],
    reverse_mapping: Optional[Dict[int, str]] = None,
) -> List[str]:
    """Convert numeric language IDs back to string labels.

    Args:
        numeric_ids: List of integer language IDs.
        reverse_mapping: Optional custom reverse mapping dictionary.

    Returns:
        List of string language tags.
    """
    rev_map = reverse_mapping or REVERSE_LANG_MAP
    return [rev_map.get(num, "OTHER") for num in numeric_ids]


def parse_sentence_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single raw CSV row into structured object format.

    Args:
        row: Dictionary representation of a DataFrame row.

    Returns:
        Dictionary containing parsed and raw fields.
    """
    aspects = parse_brace_list(row.get("all_aspects", ""))
    opinions = parse_brace_list(row.get("all_opinions", ""))
    val_strs = parse_brace_list(row.get("valence_scores", ""))
    aro_strs = parse_brace_list(row.get("arousal_scores", ""))

    valence_scores = [float(v) for v in val_strs]
    arousal_scores = [float(a) for a in aro_strs]

    tokens, lang_ids = parse_code_switch(row.get("code_switch", ""))
    numeric_lang_ids = encode_language_ids(lang_ids)

    return {
        "sentence_id": int(row["sentence_id"]),
        "text": str(row["sentence"]).strip(),
        "tokens": tokens,
        "language_ids": lang_ids,
        "language_ids_numeric": numeric_lang_ids,
        "raw_code_switch": str(row.get("code_switch", "")),
        "raw_aspects": str(row.get("all_aspects", "")),
        "raw_opinions": str(row.get("all_opinions", "")),
        "raw_valence_scores": str(row.get("valence_scores", "")),
        "raw_arousal_scores": str(row.get("arousal_scores", "")),
        "aspects": aspects,
        "opinions": opinions,
        "valence_scores": valence_scores,
        "arousal_scores": arousal_scores,
    }
