"""Aspect and Opinion span alignment module for token index mapping."""

import re
from typing import Any, Dict, List, Optional, Tuple


def normalize_token(text: str) -> str:
    """Strip punctuation and convert to lowercase for normalized matching."""
    return re.sub(r"[^\w\s]", "", text).lower().strip()


def align_span_to_tokens(
    tokens: List[str],
    query_text: str,
) -> Dict[str, Any]:
    """Map aspect/opinion raw text to start and end token indices.

    Alignment hierarchy:
    1. Exact contiguous token match
    2. Case-insensitive contiguous token match
    3. Punctuation-normalized token match
    4. String substring token-boundary search
    5. Controlled approximate subsequence matching
    6. Unmatched fallback

    Args:
        tokens: Token list from parsed code_switch/sentence.
        query_text: Aspect or opinion string from annotation.

    Returns:
        Dictionary containing:
        - text: query_text
        - start_token: int (0-indexed inclusive, -1 if unmatched)
        - end_token: int (0-indexed inclusive, -1 if unmatched)
        - alignment_status: "exact" | "normalized" | "approximate" | "unmatched"
        - is_training_safe: bool (True for exact and normalized)
        - candidate_span: string of matched tokens (or "")
    """
    clean_query = query_text.strip()
    if not clean_query or not tokens:
        return {
            "text": query_text,
            "start_token": -1,
            "end_token": -1,
            "alignment_status": "unmatched",
            "is_training_safe": False,
            "candidate_span": "",
        }

    q_words = clean_query.split()
    L = len(tokens)
    K = len(q_words)

    # 1. Exact contiguous match
    for i in range(L - K + 1):
        if tokens[i : i + K] == q_words:
            return {
                "text": query_text,
                "start_token": i,
                "end_token": i + K - 1,
                "alignment_status": "exact",
                "is_training_safe": True,
                "candidate_span": " ".join(tokens[i : i + K]),
            }

    # 2. Case-insensitive contiguous match
    tokens_lower = [t.lower() for t in tokens]
    q_words_lower = [w.lower() for w in q_words]
    for i in range(L - K + 1):
        if tokens_lower[i : i + K] == q_words_lower:
            return {
                "text": query_text,
                "start_token": i,
                "end_token": i + K - 1,
                "alignment_status": "exact",
                "is_training_safe": True,
                "candidate_span": " ".join(tokens[i : i + K]),
            }

    # 3. Punctuation-normalized token match
    tokens_norm = [normalize_token(t) for t in tokens]
    q_words_norm = [normalize_token(w) for w in q_words if normalize_token(w)]
    K_norm = len(q_words_norm)

    if K_norm > 0:
        for i in range(L - K_norm + 1):
            if tokens_norm[i : i + K_norm] == q_words_norm:
                return {
                    "text": query_text,
                    "start_token": i,
                    "end_token": i + K_norm - 1,
                    "alignment_status": "normalized",
                    "is_training_safe": True,
                    "candidate_span": " ".join(tokens[i : i + K_norm]),
                }

    # 4. String substring token-boundary search
    full_str = " ".join(tokens_lower)
    q_str = " ".join(q_words_lower)
    if q_str in full_str:
        char_start = full_str.index(q_str)
        prefix = full_str[:char_start].strip()
        start_tok = len(prefix.split()) if prefix else 0
        end_tok = min(L - 1, start_tok + K - 1)
        return {
            "text": query_text,
            "start_token": start_tok,
            "end_token": end_tok,
            "alignment_status": "normalized",
            "is_training_safe": True,
            "candidate_span": " ".join(tokens[start_tok : end_tok + 1]),
        }

    # 5. Controlled approximate subsequence matching (e.g. slight elongation / word subset)
    # Search for best token window containing majority of query words
    best_iou = 0.0
    best_span = (-1, -1)
    q_set = set(q_words_norm)

    if q_set:
        for i in range(L):
            for j in range(i, min(L, i + K + 3)):
                cand_set = set(tokens_norm[i : j + 1])
                inter = len(q_set.intersection(cand_set))
                union = len(q_set.union(cand_set))
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou and inter >= max(1, len(q_set) // 2):
                    best_iou = iou
                    best_span = (i, j)

    if best_iou >= 0.5:
        s_i, e_i = best_span
        return {
            "text": query_text,
            "start_token": s_i,
            "end_token": e_i,
            "alignment_status": "approximate",
            "is_training_safe": False,
            "candidate_span": " ".join(tokens[s_i : e_i + 1]),
        }

    # 6. Unmatched fallback
    return {
        "text": query_text,
        "start_token": -1,
        "end_token": -1,
        "alignment_status": "unmatched",
        "is_training_safe": False,
        "candidate_span": "",
    }
