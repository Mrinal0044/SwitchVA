"""Dataset splitting utility ensuring sentence-level integrity and no leakage."""

import os
import json
import random
import logging
from typing import Any, Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


def split_dataset(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_dir: str = "data/splits",
) -> Dict[str, Any]:
    """Split processed sentence records into train, validation, and test sets.

    Guarantees:
    - Splitting happens at the sentence level (all aspect-opinion annotations stay together).
    - Identical sentence texts are grouped into the same split to avoid data leakage.
    - Fully deterministic with random seed.

    Args:
        records: List of processed model-ready sentence records.
        train_ratio: Fraction of dataset for training (default 0.70).
        val_ratio: Fraction of dataset for validation (default 0.15).
        test_ratio: Fraction of dataset for testing (default 0.15).
        seed: Random seed for reproducibility.
        output_dir: Output directory for JSONL files.

    Returns:
        Split statistics dictionary.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Split ratios must sum to 1.0"

    # Group records by normalized sentence text to prevent duplicate leakage
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = rec["text"].strip().lower()
        groups[key].append(rec)

    # Deterministic shuffle of unique sentence groups
    rng = random.Random(seed)
    unique_keys = list(groups.keys())
    rng.shuffle(unique_keys)

    total_groups = len(unique_keys)
    n_train = int(total_groups * train_ratio)
    n_val = int(total_groups * val_ratio)

    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train : n_train + n_val])
    test_keys = set(unique_keys[n_train + n_val :])

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    for k, recs in groups.items():
        if k in train_keys:
            train_records.extend(recs)
        elif k in val_keys:
            val_records.extend(recs)
        else:
            test_records.extend(recs)

    # Sort each split by sentence_id for clean reproducibility
    train_records.sort(key=lambda x: x["sentence_id"])
    val_records.sort(key=lambda x: x["sentence_id"])
    test_records.sort(key=lambda x: x["sentence_id"])

    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(path: str, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "validation.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")

    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)
    write_jsonl(test_path, test_records)

    # Check for sentence_id intersection
    train_ids = {r["sentence_id"] for r in train_records}
    val_ids = {r["sentence_id"] for r in val_records}
    test_ids = {r["sentence_id"] for r in test_records}

    assert not (train_ids & val_ids), "Train and Validation share sentence IDs!"
    assert not (train_ids & test_ids), "Train and Test share sentence IDs!"
    assert not (val_ids & test_ids), "Validation and Test share sentence IDs!"

    stats = {
        "seed": seed,
        "train_sentences": len(train_records),
        "val_sentences": len(val_records),
        "test_sentences": len(test_records),
        "total_sentences": len(train_records) + len(val_records) + len(test_records),
        "train_pairs": sum(len(r.get("annotations", [])) for r in train_records),
        "val_pairs": sum(len(r.get("annotations", [])) for r in val_records),
        "test_pairs": sum(len(r.get("annotations", [])) for r in test_records),
        "total_pairs": sum(len(r.get("annotations", [])) for r in records),
    }

    logger.info("Split dataset: Train=%d, Val=%d, Test=%d sentences",
                stats["train_sentences"], stats["val_sentences"], stats["test_sentences"])

    return stats
