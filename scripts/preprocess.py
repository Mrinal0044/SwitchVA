#!/usr/bin/env python3
"""Master Preprocessing Pipeline for NSSG-DimNet (Phase 1).

Executes end-to-end dataset loading, validation, parsing, span alignment,
splitting, statistics calculation, and report generation.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import load_raw_dataset
from src.data.parser import parse_sentence_record
from src.data.validator import validate_sentence_record, check_token_consistency
from src.data.aligner import align_span_to_tokens
from src.data.splitter import split_dataset
from src.utils.logging import setup_logger

logger = setup_logger("Preprocess-Pipeline", log_file="reports/logs/preprocess.log")


def run_pipeline(
    csv_path: str = "DimABSA_Final_Dataset_600.csv",
    output_dir: str = "data/processed",
    splits_dir: str = "data/splits",
    reports_dir: str = "reports",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    mode: str = "REPORT",
) -> Dict[str, Any]:
    """Execute the full preprocessing, validation, alignment, and splitting pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    logger.info("Step 1: Loading raw CSV from %s", csv_path)
    df = load_raw_dataset(csv_path)

    logger.info("Step 2 & 3: Parsing and Validating records...")
    parsed_records: List[Dict[str, Any]] = []
    validation_errors: List[Dict[str, Any]] = []
    token_mismatches: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        rec = parse_sentence_record(row.to_dict())
        errs = validate_sentence_record(rec, mode=mode)
        if errs:
            validation_errors.extend(errs)

        tm = check_token_consistency(rec)
        if tm:
            token_mismatches.append(tm)

        parsed_records.append(rec)

    logger.info("Parsed %d sentences. Found %d validation errors, %d token mismatches.",
                len(parsed_records), len(validation_errors), len(token_mismatches))

    # Step 4 & 5: Span Alignment & Quadruplet Construction
    logger.info("Step 4 & 5: Performing Span Alignment...")
    processed_records: List[Dict[str, Any]] = []
    span_alignment_records: List[Dict[str, Any]] = []

    exact_cnt = 0
    norm_cnt = 0
    approx_cnt = 0
    unmatched_cnt = 0

    for rec in parsed_records:
        tokens = rec["tokens"]
        aspects = rec["aspects"]
        opinions = rec["opinions"]
        val_scores = rec["valence_scores"]
        aro_scores = rec["arousal_scores"]

        annotations = []
        for i in range(len(aspects)):
            asp_text = aspects[i]
            op_text = opinions[i]
            v = val_scores[i]
            a = aro_scores[i]

            asp_align = align_span_to_tokens(tokens, asp_text)
            op_align = align_span_to_tokens(tokens, op_text)

            # Update counters
            for align_obj, ann_type in [(asp_align, "aspect"), (op_align, "opinion")]:
                st = align_obj["alignment_status"]
                if st == "exact":
                    exact_cnt += 1
                elif st == "normalized":
                    norm_cnt += 1
                elif st == "approximate":
                    approx_cnt += 1
                else:
                    unmatched_cnt += 1

                span_alignment_records.append({
                    "sentence_id": rec["sentence_id"],
                    "annotation_type": ann_type,
                    "annotation_text": align_obj["text"],
                    "start_token": align_obj["start_token"],
                    "end_token": align_obj["end_token"],
                    "alignment_status": align_obj["alignment_status"],
                    "is_training_safe": align_obj["is_training_safe"],
                    "candidate_span": align_obj["candidate_span"],
                    "reason": f"Matched via {align_obj['alignment_status']} strategy",
                })

            annotations.append({
                "aspect_text": asp_text,
                "aspect_start": asp_align["start_token"],
                "aspect_end": asp_align["end_token"],
                "aspect_alignment_status": asp_align["alignment_status"],
                "aspect_training_safe": asp_align["is_training_safe"],

                "opinion_text": op_text,
                "opinion_start": op_align["start_token"],
                "opinion_end": op_align["end_token"],
                "opinion_alignment_status": op_align["alignment_status"],
                "opinion_training_safe": op_align["is_training_safe"],

                "valence": v,
                "arousal": a,
            })

        processed_rec = {
            "sentence_id": rec["sentence_id"],
            "text": rec["text"],
            "tokens": tokens,
            "language_ids": rec["language_ids"],
            "language_ids_numeric": rec["language_ids_numeric"],
            "annotations": annotations,
            "valence_scores": val_scores,
            "arousal_scores": aro_scores,
            "raw_code_switch": rec["raw_code_switch"],
        }
        processed_records.append(processed_rec)

    # Save full processed dataset
    processed_all_path = os.path.join(output_dir, "processed_dimabsa_full.jsonl")
    with open(processed_all_path, "w", encoding="utf-8") as f:
        for r in processed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved complete processed dataset to %s", processed_all_path)

    # Step 6: Dataset Splitting
    logger.info("Step 6: Splitting dataset into train/val/test...")
    split_stats = split_dataset(
        processed_records,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_dir=splits_dir,
    )

    # Step 7: Reports Generation
    logger.info("Step 7: Generating validation, token, and span reports...")
    # Validation report
    with open(os.path.join(reports_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_records": len(parsed_records),
            "errors_count": len(validation_errors),
            "errors": validation_errors,
        }, f, indent=2)

    # Token alignment report
    with open(os.path.join(reports_dir, "token_alignment_report.json"), "w", encoding="utf-8") as f:
        json.dump(token_mismatches, f, indent=2)
    pd.DataFrame(token_mismatches).to_csv(os.path.join(reports_dir, "token_alignment_report.csv"), index=False)

    # Span alignment report
    with open(os.path.join(reports_dir, "span_alignment_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "exact_count": exact_cnt,
            "normalized_count": norm_cnt,
            "approximate_count": approx_cnt,
            "unmatched_count": unmatched_cnt,
            "total_spans": len(span_alignment_records),
            "alignments": span_alignment_records,
        }, f, indent=2)
    pd.DataFrame(span_alignment_records).to_csv(os.path.join(reports_dir, "span_alignment_report.csv"), index=False)

    # Split statistics
    with open(os.path.join(reports_dir, "split_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(split_stats, f, indent=2)

    # Step 8: Comprehensive Dataset Statistics & Visualizations
    logger.info("Step 8: Computing comprehensive dataset statistics and generating plots...")
    total_spans = exact_cnt + norm_cnt + approx_cnt + unmatched_cnt
    pairs_per_sentence = [len(r["annotations"]) for r in processed_records]
    all_valences = [ann["valence"] for r in processed_records for ann in r["annotations"]]
    all_arousals = [ann["arousal"] for r in processed_records for ann in r["annotations"]]

    # Language distribution & switch statistics
    total_tokens = sum(len(r["tokens"]) for r in processed_records)
    all_langs = [lang for r in processed_records for lang in r["language_ids"]]
    hi_count = all_langs.count("HI")
    en_count = all_langs.count("EN")
    other_count = total_tokens - (hi_count + en_count)

    sentences_with_switch = 0
    sentences_without_switch = 0
    switch_counts = []

    for r in processed_records:
        l_ids = r["language_ids"]
        sw_c = 0
        for i in range(1, len(l_ids)):
            if l_ids[i] != l_ids[i - 1]:
                sw_c += 1
        switch_counts.append(sw_c)
        if sw_c > 0:
            sentences_with_switch += 1
        else:
            sentences_without_switch += 1

    # Check duplicate sentences & IDs
    unique_ids = len({r["sentence_id"] for r in processed_records})
    duplicate_ids = len(processed_records) - unique_ids
    sentence_texts = [r["text"].lower() for r in processed_records]
    duplicate_sentences = len(sentence_texts) - len(set(sentence_texts))

    dataset_stats = {
        "num_sentences": len(processed_records),
        "num_aspect_opinion_pairs": len(all_valences),
        "avg_pairs_per_sentence": float(np.mean(pairs_per_sentence)),
        "min_pairs_per_sentence": int(np.min(pairs_per_sentence)),
        "max_pairs_per_sentence": int(np.max(pairs_per_sentence)),
        "valence_stats": {
            "mean": float(np.mean(all_valences)),
            "std": float(np.std(all_valences)),
            "min": float(np.min(all_valences)),
            "max": float(np.max(all_valences)),
        },
        "arousal_stats": {
            "mean": float(np.mean(all_arousals)),
            "std": float(np.std(all_arousals)),
            "min": float(np.min(all_arousals)),
            "max": float(np.max(all_arousals)),
        },
        "token_stats": {
            "total_tokens": total_tokens,
            "hi_tokens": hi_count,
            "hi_proportion": float(hi_count / total_tokens),
            "en_tokens": en_count,
            "en_proportion": float(en_count / total_tokens),
            "other_tokens": other_count,
            "other_proportion": float(other_count / total_tokens),
        },
        "code_switch_stats": {
            "sentences_with_switch": sentences_with_switch,
            "sentences_without_switch": sentences_without_switch,
            "mean_switches_per_sentence": float(np.mean(switch_counts)),
            "max_switches_per_sentence": int(np.max(switch_counts)),
        },
        "span_alignment_stats": {
            "exact_count": exact_cnt,
            "exact_pct": float(exact_cnt / total_spans * 100),
            "normalized_count": norm_cnt,
            "normalized_pct": float(norm_cnt / total_spans * 100),
            "approximate_count": approx_cnt,
            "approximate_pct": float(approx_cnt / total_spans * 100),
            "unmatched_count": unmatched_cnt,
            "unmatched_pct": float(unmatched_cnt / total_spans * 100),
            "total_spans": total_spans,
        },
        "integrity_checks": {
            "token_count_mismatches": len(token_mismatches),
            "validation_errors": len(validation_errors),
            "duplicate_sentence_ids": duplicate_ids,
            "duplicate_sentences": duplicate_sentences,
        }
    }

    with open(os.path.join(reports_dir, "dataset_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)

    # Markdown statistics summary
    stats_md = f"""# DimABSA Dataset Statistics & Research Report

- **Total Sentences**: {dataset_stats['num_sentences']}
- **Total Aspect-Opinion Pairs**: {dataset_stats['num_aspect_opinion_pairs']}
- **Average Pairs / Sentence**: {dataset_stats['avg_pairs_per_sentence']:.2f} (Min: {dataset_stats['min_pairs_per_sentence']}, Max: {dataset_stats['max_pairs_per_sentence']})

---

## Continuous Target Distributions (Valence & Arousal)
- **Valence**: Mean = {dataset_stats['valence_stats']['mean']:.4f}, Std = {dataset_stats['valence_stats']['std']:.4f}, Range = [{dataset_stats['valence_stats']['min']:.3f}, {dataset_stats['valence_stats']['max']:.3f}]
- **Arousal**: Mean = {dataset_stats['arousal_stats']['mean']:.4f}, Std = {dataset_stats['arousal_stats']['std']:.4f}, Range = [{dataset_stats['arousal_stats']['min']:.3f}, {dataset_stats['arousal_stats']['max']:.3f}]

---

## Code-Switching & Language Statistics
- **Total Tokens**: {dataset_stats['token_stats']['total_tokens']}
- **Hindi (HI) Tokens**: {dataset_stats['token_stats']['hi_tokens']} ({dataset_stats['token_stats']['hi_proportion']*100:.1f}%)
- **English (EN) Tokens**: {dataset_stats['token_stats']['en_tokens']} ({dataset_stats['token_stats']['en_proportion']*100:.1f}%)
- **Sentences with at least 1 Code-Switch**: {dataset_stats['code_switch_stats']['sentences_with_switch']} ({dataset_stats['code_switch_stats']['sentences_with_switch']/dataset_stats['num_sentences']*100:.1f}%)
- **Sentences without Code-Switch**: {dataset_stats['code_switch_stats']['sentences_without_switch']} ({dataset_stats['code_switch_stats']['sentences_without_switch']/dataset_stats['num_sentences']*100:.1f}%)
- **Average Code-Switches per Sentence**: {dataset_stats['code_switch_stats']['mean_switches_per_sentence']:.2f}

---

## Span Alignment Results
- **Exact Matches (Safe)**: {dataset_stats['span_alignment_stats']['exact_count']} ({dataset_stats['span_alignment_stats']['exact_pct']:.2f}%)
- **Normalized Matches (Safe)**: {dataset_stats['span_alignment_stats']['normalized_count']} ({dataset_stats['span_alignment_stats']['normalized_pct']:.2f}%)
- **Approximate Matches (Unsafe candidate)**: {dataset_stats['span_alignment_stats']['approximate_count']} ({dataset_stats['span_alignment_stats']['approximate_pct']:.2f}%)
- **Unmatched**: {dataset_stats['span_alignment_stats']['unmatched_count']} ({dataset_stats['span_alignment_stats']['unmatched_pct']:.2f}%)
- **Total Spans Checked**: {dataset_stats['span_alignment_stats']['total_spans']}

---

## Dataset Splits (70 / 15 / 15)
- **Train Set**: {split_stats['train_sentences']} sentences ({split_stats['train_pairs']} pairs)
- **Validation Set**: {split_stats['val_sentences']} sentences ({split_stats['val_pairs']} pairs)
- **Test Set**: {split_stats['test_sentences']} sentences ({split_stats['test_pairs']} pairs)
- **Zero Leakage**: Duplicate sentence texts grouped strictly into matching splits.
"""
    with open(os.path.join(reports_dir, "dataset_statistics.md"), "w", encoding="utf-8") as f:
        f.write(stats_md)

    # Generate Figures
    fig_dir = os.path.join(reports_dir, "figures")

    # 1. Pairs per sentence
    plt.figure(figsize=(6, 4))
    plt.hist(pairs_per_sentence, bins=range(1, max(pairs_per_sentence) + 2), color="#2b5c8f", edgecolor="black", align="left", rwidth=0.8)
    plt.title("Aspect-Opinion Pairs per Sentence")
    plt.xlabel("Pairs Count")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "aspect_opinion_pairs_dist.png"), dpi=150)
    plt.close()

    # 2. Valence & Arousal Distribution
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.hist(all_valences, bins=20, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.title("Valence Distribution")
    plt.xlabel("Valence [0, 1]")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(all_arousals, bins=20, color="#ff7f0e", edgecolor="black", alpha=0.8)
    plt.title("Arousal Distribution")
    plt.xlabel("Arousal [0, 1]")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "valence_arousal_dist.png"), dpi=150)
    plt.close()

    # 3. Switch counts per sentence
    plt.figure(figsize=(6, 4))
    plt.hist(switch_counts, bins=20, color="#2ca02c", edgecolor="black", alpha=0.8)
    plt.title("Code-Switch Boundaries per Sentence")
    plt.xlabel("Number of Code-Switches")
    plt.ylabel("Number of Sentences")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "switch_count_dist.png"), dpi=150)
    plt.close()

    # 4. Language distribution
    plt.figure(figsize=(5, 4))
    plt.bar(["Hindi (HI)", "English (EN)"], [hi_count, en_count], color=["#e377c2", "#17becf"], edgecolor="black", width=0.6)
    plt.title("Token-Level Language Distribution")
    plt.ylabel("Token Count")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "language_distribution.png"), dpi=150)
    plt.close()

    # 5. Span alignment status
    plt.figure(figsize=(7, 4))
    statuses = ["Exact", "Normalized", "Approximate", "Unmatched"]
    counts = [exact_cnt, norm_cnt, approx_cnt, unmatched_cnt]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    plt.bar(statuses, counts, color=colors, edgecolor="black", width=0.6)
    plt.title("Span Alignment Status Distribution")
    plt.ylabel("Number of Spans")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "span_alignment_status.png"), dpi=150)
    plt.close()

    # Print smoke test summary
    summary = {
        "total_sentences": len(processed_records),
        "total_pairs": len(all_valences),
        "validation_errors": len(validation_errors),
        "token_mismatches": len(token_mismatches),
        "exact_spans": exact_cnt,
        "normalized_spans": norm_cnt,
        "approximate_spans": approx_cnt,
        "unmatched_spans": unmatched_cnt,
        "train_sentences": split_stats["train_sentences"],
        "val_sentences": split_stats["val_sentences"],
        "test_sentences": split_stats["test_sentences"],
    }

    print("\n" + "=" * 50)
    print("PHASE 1 PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total sentences:             {summary['total_sentences']}")
    print(f"Total aspect-opinion pairs:  {summary['total_pairs']}")
    print(f"Validation errors:           {summary['validation_errors']}")
    print(f"Token mismatches:            {summary['token_mismatches']}")
    print(f"Exact span alignments:       {summary['exact_spans']}")
    print(f"Normalized span alignments:  {summary['normalized_spans']}")
    print(f"Approximate span alignments: {summary['approximate_spans']}")
    print(f"Unmatched span alignments:   {summary['unmatched_spans']}")
    print(f"Train sentences:             {summary['train_sentences']}")
    print(f"Validation sentences:        {summary['val_sentences']}")
    print(f"Test sentences:              {summary['test_sentences']}")
    print("=" * 50 + "\n")

    return summary


if __name__ == "__main__":
    run_pipeline()
