"""Word-to-Subword Alignment for multilingual transformers preserving language IDs."""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Extended language ID mapping including SPECIAL tokens and PAD
SUBWORD_LANG_MAP = {
    "HI": 0,
    "EN": 1,
    "OTHER": 2,
    "SPECIAL": 3,
    "PAD": 4,
}

REVERSE_SUBWORD_LANG_MAP = {v: k for k, v in SUBWORD_LANG_MAP.items()}


class SubwordAligner:
    """Aligns word-level language annotations to subword transformer tokens.

    Preserves exact word-level language IDs across constituent subword pieces
    and computes bidirectional index mappings:
    - word_to_subword: Maps each word index to its list of subword token indices.
    - subword_to_word: Maps each subword token index to its source word index (-1 for special/pad).
    """

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        max_seq_length: int = 128,
        lang_map: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            tokenizer: Pretrained HuggingFace tokenizer (or None for rule-based mock).
            max_seq_length: Maximum subword sequence length.
            lang_map: Language ID mapping dictionary.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.lang_map = lang_map or SUBWORD_LANG_MAP
        self.special_lang_id = self.lang_map.get("SPECIAL", 3)
        self.pad_lang_id = self.lang_map.get("PAD", 4)
        self.other_lang_id = self.lang_map.get("OTHER", 2)

    def align_single(
        self,
        words: List[str],
        language_ids: List[str],
    ) -> Dict[str, Any]:
        """Align a single sentence's words and language IDs to subword tokens.

        Args:
            words: List of word tokens from sentence.
            language_ids: List of language ID strings ('HI', 'EN', etc.) matching words.

        Returns:
            Dictionary containing:
            - subwords: List[str]
            - input_ids: List[int]
            - attention_mask: List[int]
            - subword_lang_ids: List[str]
            - subword_lang_ids_numeric: List[int]
            - word_to_subword: List[List[int]]
            - subword_to_word: List[int]
        """
        assert len(words) == len(language_ids), (
            f"Mismatch: len(words)={len(words)} != len(language_ids)={len(language_ids)}"
        )

        subwords: List[str] = []
        subword_lang_ids: List[str] = []
        subword_lang_numeric: List[int] = []
        subword_to_word: List[int] = []
        word_to_subword: List[List[int]] = [[] for _ in range(len(words))]
        input_ids: List[int] = []

        # 1. Add leading special token ([CLS] / <s>)
        cls_token = "<s>"
        cls_id = 0
        if self.tokenizer is not None and hasattr(self.tokenizer, "cls_token_id") and self.tokenizer.cls_token_id is not None:
            cls_token = self.tokenizer.cls_token or "<s>"
            cls_id = self.tokenizer.cls_token_id
        elif self.tokenizer is not None and hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            cls_token = self.tokenizer.bos_token or "<s>"
            cls_id = self.tokenizer.bos_token_id

        subwords.append(cls_token)
        input_ids.append(cls_id)
        subword_lang_ids.append("SPECIAL")
        subword_lang_numeric.append(self.special_lang_id)
        subword_to_word.append(-1)

        # 2. Tokenize each word and inherit language ID
        for w_idx, (word, lang) in enumerate(zip(words, language_ids)):
            numeric_lang = self.lang_map.get(lang.upper(), self.other_lang_id)

            if self.tokenizer is not None:
                # Tokenize individual word without adding special tokens
                word_pieces = self.tokenizer.tokenize(word)
                if not word_pieces:
                    word_pieces = [word]
                piece_ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
            else:
                # Rule-based fallback: split word into pieces if long
                word_pieces = [word]
                piece_ids = [hash(word) % 50000]

            for piece, p_id in zip(word_pieces, piece_ids):
                if len(subwords) >= self.max_seq_length - 1:
                    # Reserve space for closing special token
                    break
                curr_idx = len(subwords)
                subwords.append(piece)
                input_ids.append(p_id)
                subword_lang_ids.append(lang.upper())
                subword_lang_numeric.append(numeric_lang)
                subword_to_word.append(w_idx)
                word_to_subword[w_idx].append(curr_idx)

            if len(subwords) >= self.max_seq_length - 1:
                break

        # 3. Add closing special token ([SEP] / </s>)
        sep_token = "</s>"
        sep_id = 2
        if self.tokenizer is not None and hasattr(self.tokenizer, "sep_token_id") and self.tokenizer.sep_token_id is not None:
            sep_token = self.tokenizer.sep_token or "</s>"
            sep_id = self.tokenizer.sep_token_id
        elif self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id") and self.tokenizer.eos_token_id is not None:
            sep_token = self.tokenizer.eos_token or "</s>"
            sep_id = self.tokenizer.eos_token_id

        subwords.append(sep_token)
        input_ids.append(sep_id)
        subword_lang_ids.append("SPECIAL")
        subword_lang_numeric.append(self.special_lang_id)
        subword_to_word.append(-1)

        attention_mask = [1] * len(input_ids)

        return {
            "subwords": subwords,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "subword_lang_ids": subword_lang_ids,
            "subword_lang_ids_numeric": subword_lang_numeric,
            "word_to_subword": word_to_subword,
            "subword_to_word": subword_to_word,
        }

    def pad_batch(
        self,
        batch_aligned: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Pad a list of aligned single records into a uniform batch tensor/array representation."""
        max_len = max(len(item["input_ids"]) for item in batch_aligned)
        max_len = min(max_len, self.max_seq_length)

        pad_token_id = 1
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id") and self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_lang_numeric = []
        batch_subword_to_word = []

        for item in batch_aligned:
            cur_len = len(item["input_ids"])
            pad_len = max_len - cur_len

            input_ids = item["input_ids"] + [pad_token_id] * pad_len
            attn_mask = item["attention_mask"] + [0] * pad_len
            lang_nums = item["subword_lang_ids_numeric"] + [self.pad_lang_id] * pad_len
            s_to_w = item["subword_to_word"] + [-1] * pad_len

            batch_input_ids.append(input_ids[:max_len])
            batch_attention_mask.append(attn_mask[:max_len])
            batch_lang_numeric.append(lang_nums[:max_len])
            batch_subword_to_word.append(s_to_w[:max_len])

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "subword_lang_ids_numeric": batch_lang_numeric,
            "subword_to_word": batch_subword_to_word,
            "aligned_items": batch_aligned,
        }


def map_gold_spans_to_subwords(
    word_start: int,
    word_end: int,
    word_to_subword: List[List[int]],
    is_training_safe: bool = True,
) -> Tuple[int, int, bool]:
    """Map a word-level inclusive span [word_start, word_end] to subword indices.

    Args:
        word_start: Inclusive start word index (0-indexed).
        word_end: Inclusive end word index (0-indexed).
        word_to_subword: Mapping from word index to list of subword indices.
        is_training_safe: Phase 1 flag (exact/normalized=True, approx/unmatched=False).

    Returns:
        Tuple of (subword_start, subword_end, is_valid_subword_span).
        If mapping fails or is not training-safe, returns (-1, -1, False).
    """
    if not is_training_safe:
        return -1, -1, False

    if word_start < 0 or word_end < 0 or word_start >= len(word_to_subword) or word_end >= len(word_to_subword):
        return -1, -1, False

    if word_start > word_end:
        return -1, -1, False

    start_pieces = word_to_subword[word_start]
    end_pieces = word_to_subword[word_end]

    if not start_pieces or not end_pieces:
        return -1, -1, False

    subword_start = min(start_pieces)
    subword_end = max(end_pieces)

    return subword_start, subword_end, True

