"""Dataset and Collation classes for DimABSA (Hinglish Aspect-Level Valence and Arousal)."""

import os
import json
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import torch
from torch.utils.data import Dataset


class DimABSADataset(Dataset):
    """PyTorch-compatible dataset abstraction for DimABSA.

    Supports:
    - Loading processed JSONL splits (train.jsonl, validation.jsonl, test.jsonl)
    - Loading raw CSV with automatic structuring
    - Indexing by sentence
    - Retrieving tokens, string language IDs, numeric language IDs
    - Retrieving quadruplets: aspect span, opinion span, valence, arousal
    - Retrieving alignment metadata
    """

    def __init__(
        self,
        data_path: Union[str, os.PathLike],
        transform: Optional[Any] = None,
    ):
        """
        Args:
            data_path: Path to processed JSONL file or raw CSV.
            transform: Optional callable transform.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        self.data_path = str(data_path)
        self.transform = transform
        self.records: List[Dict[str, Any]] = []

        if self.data_path.endswith(".jsonl"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.records.append(json.loads(line))
        elif self.data_path.endswith(".csv"):
            from .parser import parse_sentence_record
            df = pd.read_csv(self.data_path)
            for _, row in df.iterrows():
                self.records.append(parse_sentence_record(row.to_dict()))
        else:
            raise ValueError(f"Unsupported file format for dataset: {self.data_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a sentence record with all associated quadruplets.

        Returns dict containing:
        - sentence_id: int
        - text: str
        - tokens: List[str]
        - language_ids: List[str]
        - language_ids_numeric: List[int]
        - annotations: List[Dict[str, Any]] containing aspect/opinion spans and targets
        - valence_scores: List[float]
        - arousal_scores: List[float]
        """
        sample = self.records[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def dimabsa_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for batching variable-length DimABSA samples."""
    return {
        "sentence_id": [d["sentence_id"] for d in batch],
        "text": [d["text"] for d in batch],
        "tokens": [d["tokens"] for d in batch],
        "language_ids": [d["language_ids"] for d in batch],
        "language_ids_numeric": [d["language_ids_numeric"] for d in batch],
        "annotations": [d.get("annotations", []) for d in batch],
        "valence_scores": [d.get("valence_scores", []) for d in batch],
        "arousal_scores": [d.get("arousal_scores", []) for d in batch],
    }
