"""Data utilities for training and evaluating DimABSA models.

Handles:
- Loading processed JSONL splits (train, validation, test)
- Sentence-level and annotation-level sample expansion
- Tokenizer-based and rule-based subword alignment
- Dynamic batch padding collation
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset

from src.data.tokenizer_alignment import SubwordAligner, map_gold_spans_to_subwords


class DimABSATensorDataset(Dataset):
    """Dataset producing PyTorch tensor dicts ready for NSSGDimNet and Baseline models."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Optional[Any] = None,
        max_seq_length: int = 128,
        training_safe_only: bool = True,
    ):
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at: {jsonl_path}")

        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.training_safe_only = training_safe_only
        self.aligner = SubwordAligner(tokenizer=tokenizer, max_seq_length=max_seq_length)

        self.samples: List[Dict[str, Any]] = []
        self.sentence_records: List[Dict[str, Any]] = []
        self._load_and_process()

    def _load_and_process(self) -> None:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                self.sentence_records.append(record)

                tokens = record.get("tokens", [])
                lang_ids = record.get("language_ids", ["OTHER"] * len(tokens))
                annotations = record.get("annotations", [])

                if not tokens or not annotations:
                    continue

                # Align sentence tokens and languages to subwords
                aligned = self.aligner.align_single(tokens, lang_ids)
                input_ids = aligned["input_ids"]
                attention_mask = aligned["attention_mask"]
                lang_numeric = aligned["subword_lang_ids_numeric"]
                word_to_subword = aligned["word_to_subword"]

                # Extract all gold aspect and opinion spans for this sentence
                sentence_aspect_spans = []
                sentence_opinion_spans = []
                for ann_item in annotations:
                    a_s = ann_item.get("aspect_start", -1)
                    a_e = ann_item.get("aspect_end", -1)
                    o_s = ann_item.get("opinion_start", -1)
                    o_e = ann_item.get("opinion_end", -1)

                    a_status = ann_item.get("aspect_alignment_status", "exact")
                    o_status = ann_item.get("opinion_alignment_status", "exact")
                    a_safe = bool(ann_item.get("aspect_training_safe", True) and a_status in ["exact", "normalized"])
                    o_safe = bool(ann_item.get("opinion_training_safe", True) and o_status in ["exact", "normalized"])

                    sub_a_s, sub_a_e, a_valid = map_gold_spans_to_subwords(a_s, a_e, word_to_subword, is_training_safe=True)
                    sub_o_s, sub_o_e, o_valid = map_gold_spans_to_subwords(o_s, o_e, word_to_subword, is_training_safe=True)

                    if a_valid and sub_a_s <= sub_a_e:
                        sentence_aspect_spans.append({
                            "start": sub_a_s,
                            "end": sub_a_e,
                            "is_training_safe": a_safe,
                            "text": ann_item.get("aspect_text", ""),
                            "status": a_status,
                        })
                    if o_valid and sub_o_s <= sub_o_e:
                        sentence_opinion_spans.append({
                            "start": sub_o_s,
                            "end": sub_o_e,
                            "is_training_safe": o_safe,
                            "text": ann_item.get("opinion_text", ""),
                            "status": o_status,
                        })

                for ann in annotations:
                    # Check training safe filter
                    if self.training_safe_only:
                        if not ann.get("aspect_training_safe", True) or not ann.get("opinion_training_safe", True):
                            continue

                    a_start = ann.get("aspect_start", -1)
                    a_end = ann.get("aspect_end", -1)
                    o_start = ann.get("opinion_start", -1)
                    o_end = ann.get("opinion_end", -1)

                    sub_a_start, sub_a_end, a_valid = map_gold_spans_to_subwords(
                        a_start, a_end, word_to_subword, is_training_safe=True
                    )
                    sub_o_start, sub_o_end, o_valid = map_gold_spans_to_subwords(
                        o_start, o_end, word_to_subword, is_training_safe=True
                    )

                    # Fallbacks if alignment failed
                    if not a_valid:
                        sub_a_start, sub_a_end = max(0, a_start), max(0, a_end)
                    if not o_valid:
                        sub_o_start, sub_o_end = max(0, o_start), max(0, o_end)

                    sample = {
                        "sentence_id": record.get("sentence_id", 0),
                        "text": record.get("text", ""),
                        "tokens": tokens,
                        "language_ids": lang_ids,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "lang_ids": lang_numeric,
                        "aspect_span": [sub_a_start, sub_a_end],
                        "opinion_span": [sub_o_start, sub_o_end],
                        "all_aspect_spans": sentence_aspect_spans,
                        "all_opinion_spans": sentence_opinion_spans,
                        "aspect_text": ann.get("aspect_text", ""),
                        "opinion_text": ann.get("opinion_text", ""),
                        "valence": float(ann.get("valence", 0.5)),
                        "arousal": float(ann.get("arousal", 0.5)),
                        "raw_annotation": ann,
                    }
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def dimabsa_tensor_collate_fn(batch: List[Dict[str, Any]], max_span_length: int = 15) -> Dict[str, Any]:
    """Dynamically pad a batch of DimABSA samples to max length and construct 2D span target grids."""
    from src.losses.span_loss import construct_gold_span_grid

    max_len = max(len(s["input_ids"]) for s in batch)
    batch_size = len(batch)

    batch_input_ids = []
    batch_attention_mask = []
    batch_lang_ids = []
    batch_aspect_spans = []
    batch_opinion_spans = []
    batch_valence = []
    batch_arousal = []
    batch_sent_ids = []
    batch_tokens = []
    batch_lang_strs = []
    batch_aspect_texts = []
    batch_opinion_texts = []
    batch_raw_anns = []
    batch_gold_aspect_spans = []
    batch_gold_opinion_spans = []

    for s in batch:
        pad_len = max_len - len(s["input_ids"])

        batch_input_ids.append(s["input_ids"] + [1] * pad_len)
        batch_attention_mask.append(s["attention_mask"] + [0] * pad_len)
        batch_lang_ids.append(s["lang_ids"] + [4] * pad_len)  # 4 is PAD_LANG_ID

        batch_aspect_spans.append(s["aspect_span"])
        batch_opinion_spans.append(s["opinion_span"])
        batch_valence.append(s["valence"])
        batch_arousal.append(s["arousal"])

        batch_sent_ids.append(s["sentence_id"])
        batch_tokens.append(s["tokens"])
        batch_lang_strs.append(s["language_ids"])
        batch_aspect_texts.append(s["aspect_text"])
        batch_opinion_texts.append(s["opinion_text"])
        batch_raw_anns.append(s["raw_annotation"])

        # Prioritize sentence-level collection of all safe spans to prevent false negative supervision
        asp_spans = s.get("all_aspect_spans") or [{
            "start": s["aspect_span"][0],
            "end": s["aspect_span"][1],
            "is_training_safe": s["raw_annotation"].get("aspect_training_safe", True),
            "status": s["raw_annotation"].get("aspect_alignment_status", "exact"),
        }]
        op_spans = s.get("all_opinion_spans") or [{
            "start": s["opinion_span"][0],
            "end": s["opinion_span"][1],
            "is_training_safe": s["raw_annotation"].get("opinion_training_safe", True),
            "status": s["raw_annotation"].get("opinion_alignment_status", "exact"),
        }]
        batch_gold_aspect_spans.append(asp_spans)
        batch_gold_opinion_spans.append(op_spans)

    input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.float)

    # Construct valid candidate mask (i <= j, length <= max_span_length, non-pad)
    i_grid = torch.arange(max_len).unsqueeze(1).expand(max_len, max_len)
    j_grid = torch.arange(max_len).unsqueeze(0).expand(max_len, max_len)
    triu_mask = i_grid <= j_grid
    len_mask = (j_grid - i_grid + 1) <= max_span_length
    base_valid = triu_mask & len_mask

    valid_span_mask = base_valid.unsqueeze(0).expand(batch_size, max_len, max_len).clone()
    pad_i = attention_mask_tensor.unsqueeze(2).expand(batch_size, max_len, max_len).bool()
    pad_j = attention_mask_tensor.unsqueeze(1).expand(batch_size, max_len, max_len).bool()
    valid_span_mask = valid_span_mask & pad_i & pad_j

    # Construct gold target grids and supervision masks
    aspect_span_targets, aspect_supervision_mask = construct_gold_span_grid(
        batch_size=batch_size,
        seq_len=max_len,
        gold_spans=batch_gold_aspect_spans,
        valid_mask=valid_span_mask,
    )
    opinion_span_targets, opinion_supervision_mask = construct_gold_span_grid(
        batch_size=batch_size,
        seq_len=max_len,
        gold_spans=batch_gold_opinion_spans,
        valid_mask=valid_span_mask,
    )

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "lang_ids": torch.tensor(batch_lang_ids, dtype=torch.long),
        "aspect_spans": torch.tensor(batch_aspect_spans, dtype=torch.long),
        "opinion_spans": torch.tensor(batch_opinion_spans, dtype=torch.long),
        "valence": torch.tensor(batch_valence, dtype=torch.float),
        "arousal": torch.tensor(batch_arousal, dtype=torch.float),
        "sentence_id": batch_sent_ids,
        "tokens": batch_tokens,
        "language_ids": batch_lang_strs,
        "aspect_text": batch_aspect_texts,
        "opinion_text": batch_opinion_texts,
        "raw_annotations": batch_raw_anns,
        "gold_aspect_spans": batch_gold_aspect_spans,
        "gold_opinion_spans": batch_gold_opinion_spans,
        "aspect_span_targets": aspect_span_targets,
        "opinion_span_targets": opinion_span_targets,
        "aspect_supervision_mask": aspect_supervision_mask,
        "opinion_supervision_mask": opinion_supervision_mask,
        "valid_span_mask": valid_span_mask,
        "span_targets": aspect_span_targets,  # Alias for backward compatibility
    }
