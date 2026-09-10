"""Comprehensive unit tests for Phase 1 dataset preprocessing, validation, and alignment."""

import os
import tempfile
import pytest
import pandas as pd
import torch

from src.data.loader import load_raw_dataset, EXPECTED_COLUMNS
from src.data.parser import (
    parse_brace_list,
    parse_code_switch,
    encode_language_ids,
    decode_language_ids,
    parse_sentence_record,
)
from src.data.validator import (
    validate_sentence_record,
    check_token_consistency,
    ValidationError,
)
from src.data.aligner import (
    align_span_to_tokens,
    normalize_token,
)
from src.data.splitter import split_dataset
from src.data.dataset import DimABSADataset, dimabsa_collate_fn


# TEST 1: CSV loading
def test_csv_loading():
    df = load_raw_dataset("DimABSA_Final_Dataset_600.csv")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 600


# TEST 2: Expected-column validation
def test_expected_column_validation():
    df = load_raw_dataset("DimABSA_Final_Dataset_600.csv")
    for col in EXPECTED_COLUMNS:
        assert col in df.columns

    # Test error when required column missing
    bad_df = df.drop(columns=["sentence"])
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
        bad_df.to_csv(tmp.name, index=False)
        tmp_name = tmp.name

    try:
        with pytest.raises(ValueError, match="Dataset is missing required columns"):
            load_raw_dataset(tmp_name)
    finally:
        os.remove(tmp_name)


# TEST 3: Annotation parsing
def test_annotation_parsing():
    s1 = "{bowling; stump}"
    s2 = "{ham bowling krte hai; apne stump bhjao}"
    s3 = "{0.592; 0.475}"

    assert parse_brace_list(s1) == ["bowling", "stump"]
    assert parse_brace_list(s2) == ["ham bowling krte hai", "apne stump bhjao"]
    assert parse_brace_list(s3) == ["0.592", "0.475"]
    assert parse_brace_list("{}") == []
    assert parse_brace_list("") == []


# TEST 4: Positional aspect/opinion/VA pairing
def test_positional_pairing():
    row = {
        "sentence_id": 1,
        "sentence": "sanatan bowling krte hai apne stump bhjao",
        "code_switch": "sanatan:HI bowling:EN krte:HI hai:HI apne:HI stump:EN bhjao:HI",
        "all_aspects": "{bowling; stump}",
        "all_opinions": "{bowling krte hai; apne stump bhjao}",
        "valence_scores": "{0.592; 0.475}",
        "arousal_scores": "{0.369; 0.480}",
    }
    rec = parse_sentence_record(row)
    assert len(rec["aspects"]) == len(rec["opinions"]) == len(rec["valence_scores"]) == len(rec["arousal_scores"]) == 2
    assert rec["aspects"][0] == "bowling"
    assert rec["opinions"][0] == "bowling krte hai"
    assert rec["valence_scores"][0] == 0.592
    assert rec["arousal_scores"][0] == 0.369

    assert rec["aspects"][1] == "stump"
    assert rec["opinions"][1] == "apne stump bhjao"
    assert rec["valence_scores"][1] == 0.475
    assert rec["arousal_scores"][1] == 0.480


# TEST 5: VA range validation
def test_va_range_validation():
    valid_rec = {
        "sentence_id": 10,
        "aspects": ["asp"],
        "opinions": ["op"],
        "valence_scores": [0.75],
        "arousal_scores": [0.25],
    }
    errs = validate_sentence_record(valid_rec, mode="REPORT")
    assert len(errs) == 0

    invalid_val_rec = {
        "sentence_id": 11,
        "aspects": ["asp"],
        "opinions": ["op"],
        "valence_scores": [1.5],  # > 1.0
        "arousal_scores": [0.5],
    }
    errs_val = validate_sentence_record(invalid_val_rec, mode="REPORT")
    assert len(errs_val) == 1
    assert errs_val[0]["error_type"] == "VALUE_OUT_OF_RANGE"

    with pytest.raises(ValidationError):
        validate_sentence_record(invalid_val_rec, mode="STRICT")


# TEST 6: Code-switch parsing
def test_code_switch_parsing():
    cs_str = "aap:HI video:EN main:HI beech:EN"
    tokens, lang_ids = parse_code_switch(cs_str)
    assert tokens == ["aap", "video", "main", "beech"]
    assert lang_ids == ["HI", "EN", "HI", "EN"]


# TEST 7: Language-ID mapping
def test_language_id_mapping():
    lang_ids = ["HI", "EN", "HI", "OTHER"]
    numeric = encode_language_ids(lang_ids)
    assert numeric == [0, 1, 0, 2]
    decoded = decode_language_ids(numeric)
    assert decoded == ["HI", "EN", "HI", "OTHER"]


# TEST 8: Token-count consistency
def test_token_count_consistency():
    matched_rec = {
        "sentence_id": 1,
        "text": "aap video main",
        "tokens": ["aap", "video", "main"],
        "raw_code_switch": "aap:HI video:EN main:HI",
    }
    assert check_token_consistency(matched_rec) is None

    mismatched_rec = {
        "sentence_id": 2,
        "text": "aap video main extra",
        "tokens": ["aap", "video", "main"],
        "raw_code_switch": "aap:HI video:EN main:HI",
    }
    res = check_token_consistency(mismatched_rec)
    assert res is not None
    assert res["status"] == "MISMATCH"


# TEST 9: Exact span alignment
def test_exact_span_alignment():
    tokens = ["ham", "bowling", "krte", "hai", "aap", "apne", "stump", "bhjao"]
    res = align_span_to_tokens(tokens, "bowling")
    assert res["alignment_status"] == "exact"
    assert res["start_token"] == 1
    assert res["end_token"] == 1
    assert res["is_training_safe"] is True


# TEST 10: Case-insensitive alignment
def test_case_insensitive_alignment():
    tokens = ["Ham", "Bowling", "Krte", "Hai"]
    res = align_span_to_tokens(tokens, "bowling krte")
    assert res["alignment_status"] == "exact"
    assert res["start_token"] == 1
    assert res["end_token"] == 2
    assert res["is_training_safe"] is True


# TEST 11: Normalized alignment
def test_normalized_alignment():
    tokens = ["ham", "bowling!", "krte,", "hai"]
    res = align_span_to_tokens(tokens, "bowling krte")
    assert res["alignment_status"] == "normalized"
    assert res["start_token"] == 1
    assert res["end_token"] == 2
    assert res["is_training_safe"] is True


# TEST 12: Approximate alignment classification
def test_approximate_alignment():
    tokens = ["girls", "aaj", "fir", "jeeeee", "bhar", "k", "delhi"]
    res = align_span_to_tokens(tokens, "jeeeeee bhar k")
    assert res["alignment_status"] == "approximate"
    assert res["is_training_safe"] is False


# TEST 13: Unmatched annotation handling
def test_unmatched_annotation_handling():
    tokens = ["shameful", "me", "jeete", "aise", "to"]
    res = align_span_to_tokens(tokens, "completely nonexistent phrase")
    assert res["alignment_status"] == "unmatched"
    assert res["start_token"] == -1
    assert res["end_token"] == -1
    assert res["is_training_safe"] is False


# TEST 14: Training-safe flag behavior
def test_training_safe_flag():
    tokens = ["a", "b", "c", "d"]
    exact_res = align_span_to_tokens(tokens, "b c")
    unmatched_res = align_span_to_tokens(tokens, "z")

    assert exact_res["is_training_safe"] is True
    assert unmatched_res["is_training_safe"] is False


# TEST 15: No annotation loss during conversion
def test_no_annotation_loss():
    row = {
        "sentence_id": 99,
        "sentence": "test sentence for verification",
        "code_switch": "test:EN sentence:EN for:EN verification:EN",
        "all_aspects": "{aspect1; aspect2; aspect3}",
        "all_opinions": "{opinion1; opinion2; opinion3}",
        "valence_scores": "{0.1; 0.5; 0.9}",
        "arousal_scores": "{0.2; 0.6; 0.8}",
    }
    rec = parse_sentence_record(row)
    assert len(rec["aspects"]) == 3
    assert len(rec["opinions"]) == 3
    assert len(rec["valence_scores"]) == 3
    assert len(rec["arousal_scores"]) == 3


# TEST 16: Sentence-level dataset splitting
def test_sentence_level_splitting():
    records = [
        {
            "sentence_id": i,
            "text": f"sentence {i}",
            "tokens": [f"tok_{i}"],
            "language_ids": ["EN"],
            "language_ids_numeric": [1],
            "annotations": [{"valence": 0.5, "arousal": 0.5}],
        }
        for i in range(100)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        stats = split_dataset(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, output_dir=tmpdir)
        assert stats["train_sentences"] == 70
        assert stats["val_sentences"] == 15
        assert stats["test_sentences"] == 15


# TEST 17: No sentence leakage across splits
def test_no_sentence_leakage():
    records = [
        {
            "sentence_id": i,
            "text": f"unique sentence {i}",
            "tokens": ["tok"],
            "language_ids": ["EN"],
            "language_ids_numeric": [1],
            "annotations": [],
        }
        for i in range(50)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        split_dataset(records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42, output_dir=tmpdir)
        train_ds = DimABSADataset(os.path.join(tmpdir, "train.jsonl"))
        val_ds = DimABSADataset(os.path.join(tmpdir, "validation.jsonl"))
        test_ds = DimABSADataset(os.path.join(tmpdir, "test.jsonl"))

        train_texts = {r["text"] for r in train_ds}
        val_texts = {r["text"] for r in val_ds}
        test_texts = {r["text"] for r in test_ds}

        assert not (train_texts & val_texts)
        assert not (train_texts & test_texts)
        assert not (val_texts & test_texts)


# TEST 18: Duplicate detection & grouping
def test_duplicate_grouping():
    records = [
        {"sentence_id": 1, "text": "dup sentence", "tokens": ["dup"], "language_ids": ["HI"], "language_ids_numeric": [0], "annotations": []},
        {"sentence_id": 2, "text": "dup sentence", "tokens": ["dup"], "language_ids": ["HI"], "language_ids_numeric": [0], "annotations": []},
        {"sentence_id": 3, "text": "unique sentence", "tokens": ["unq"], "language_ids": ["EN"], "language_ids_numeric": [1], "annotations": []},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        split_dataset(records, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=42, output_dir=tmpdir)
        train_ds = DimABSADataset(os.path.join(tmpdir, "train.jsonl"))
        val_ds = DimABSADataset(os.path.join(tmpdir, "validation.jsonl"))
        test_ds = DimABSADataset(os.path.join(tmpdir, "test.jsonl"))

        # Both dup records must be in the same split file
        all_splits = [list(train_ds), list(val_ds), list(test_ds)]
        found_together = False
        for s in all_splits:
            ids = [r["sentence_id"] for r in s]
            if 1 in ids and 2 in ids:
                found_together = True
        assert found_together, "Duplicate sentences were separated across splits!"


# TEST 19: Model-ready JSON serialization/deserialization
def test_jsonl_serialization():
    rec = {
        "sentence_id": 1,
        "text": "test",
        "tokens": ["test"],
        "language_ids": ["EN"],
        "language_ids_numeric": [1],
        "annotations": [
            {
                "aspect_text": "test",
                "aspect_start": 0,
                "aspect_end": 0,
                "valence": 0.5,
                "arousal": 0.5,
            }
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tmp:
        import json
        tmp.write(json.dumps(rec) + "\n")
        tmp_name = tmp.name

    try:
        ds = DimABSADataset(tmp_name)
        loaded = ds[0]
        assert loaded["sentence_id"] == 1
        assert loaded["annotations"][0]["valence"] == 0.5
    finally:
        os.remove(tmp_name)


# TEST 20: Dataset __getitem__()
def test_dataset_getitem():
    df = load_raw_dataset("DimABSA_Final_Dataset_600.csv")
    ds = DimABSADataset("DimABSA_Final_Dataset_600.csv")
    assert len(ds) == 600
    item = ds[0]
    assert item["sentence_id"] == 1
    assert "tokens" in item
    assert "language_ids" in item
    assert "language_ids_numeric" in item
    assert "aspects" in item
    assert "valence_scores" in item
