"""Validation utilities for DimABSA dataset structure and consistency."""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when a critical validation rule fails in STRICT mode."""
    pass


def validate_sentence_record(
    record: Dict[str, Any],
    mode: str = "REPORT",
) -> List[Dict[str, Any]]:
    """Validate structural constraints on a single parsed sentence record.

    Rules checked:
    1. Equal length across aspects, opinions, valence_scores, arousal_scores.
    2. Valence values bounded in [0.0, 1.0].
    3. Arousal values bounded in [0.0, 1.0].

    Args:
        record: Parsed sentence dictionary.
        mode: "REPORT" (default) or "STRICT".

    Returns:
        List of validation error dictionaries (if any).
    """
    errors: List[Dict[str, Any]] = []
    sid = record["sentence_id"]

    num_asp = len(record.get("aspects", []))
    num_op = len(record.get("opinions", []))
    num_val = len(record.get("valence_scores", []))
    num_aro = len(record.get("arousal_scores", []))

    # 1. Structural count alignment check
    if not (num_asp == num_op == num_val == num_aro):
        err = {
            "sentence_id": sid,
            "field": "annotations_length",
            "observed_value": f"aspects={num_asp}, opinions={num_op}, valence={num_val}, arousal={num_aro}",
            "expected_value": "All annotation lists must have identical lengths",
            "error_type": "LENGTH_MISMATCH",
            "error_message": f"Sentence {sid} has misaligned annotation counts.",
        }
        errors.append(err)
        if mode == "STRICT":
            raise ValidationError(err["error_message"])

    # 2. Valence range check [0, 1]
    for idx, v in enumerate(record.get("valence_scores", [])):
        if not (0.0 <= v <= 1.0):
            err = {
                "sentence_id": sid,
                "field": f"valence_scores[{idx}]",
                "observed_value": v,
                "expected_value": "0.0 <= valence <= 1.0",
                "error_type": "VALUE_OUT_OF_RANGE",
                "error_message": f"Valence score {v} at index {idx} in sentence {sid} is outside [0, 1].",
            }
            errors.append(err)
            if mode == "STRICT":
                raise ValidationError(err["error_message"])

    # 3. Arousal range check [0, 1]
    for idx, a in enumerate(record.get("arousal_scores", [])):
        if not (0.0 <= a <= 1.0):
            err = {
                "sentence_id": sid,
                "field": f"arousal_scores[{idx}]",
                "observed_value": a,
                "expected_value": "0.0 <= arousal <= 1.0",
                "error_type": "VALUE_OUT_OF_RANGE",
                "error_message": f"Arousal score {a} at index {idx} in sentence {sid} is outside [0, 1].",
            }
            errors.append(err)
            if mode == "STRICT":
                raise ValidationError(err["error_message"])

    return errors


def check_token_consistency(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compare parsed code-switch tokens against whitespace tokenization of raw sentence.

    Args:
        record: Parsed sentence dictionary.

    Returns:
        Mismatch report dictionary if counts differ, else None.
    """
    sid = record["sentence_id"]
    tokens = record.get("tokens", [])
    sentence_text = record.get("text", "")
    whitespace_tokens = sentence_text.split()

    if len(tokens) != len(whitespace_tokens):
        return {
            "sentence_id": sid,
            "original_sentence": sentence_text,
            "parsed_token_count": len(tokens),
            "whitespace_token_count": len(whitespace_tokens),
            "code_switch_string": record.get("raw_code_switch", ""),
            "status": "MISMATCH",
            "reason": "Token count from code_switch does not match whitespace split count",
        }
    return None
