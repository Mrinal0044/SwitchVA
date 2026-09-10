#!/usr/bin/env python3
"""Phase 3 Verification Script: Biaffine Aspect-Opinion Span Extraction & Diagnostics."""

import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.tokenizer_alignment import SubwordAligner, map_gold_spans_to_subwords
from src.models.switch_encoder import SwitchAwareEncoder
from src.models.biaffine_span import BiaffineAspectOpinionSpanExtractor
from src.losses.span_loss import BiaffineSpanLoss
from src.data.dataset import DimABSADataset


def plot_biaffine_score_grid(
    tokens: list,
    aspect_scores: torch.Tensor,
    opinion_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    sentence_id: int,
    output_dir: str = "reports/figures/biaffine",
):
    """Plot and save 2D heatmap score matrices for Aspect and Opinion spans."""
    os.makedirs(output_dir, exist_ok=True)
    seq_len = len(tokens)

    asp_s = aspect_scores.detach().cpu().numpy()
    op_s = opinion_scores.detach().cpu().numpy()
    mask = valid_mask.detach().cpu().numpy()

    # Mask out invalid cells for visual clarity
    asp_masked = np.where(mask, asp_s, np.nan)
    op_masked = np.where(mask, op_s, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Aspect Grid
    im0 = axes[0].imshow(asp_masked, cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"Aspect Span Scores (Sentence {sentence_id})")
    axes[0].set_xlabel("End Token Index")
    axes[0].set_ylabel("Start Token Index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Opinion Grid
    im1 = axes[1].imshow(op_masked, cmap="magma", interpolation="nearest")
    axes[1].set_title(f"Opinion Span Scores (Sentence {sentence_id})")
    axes[1].set_xlabel("End Token Index")
    axes[1].set_ylabel("Start Token Index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Add token ticks if sequence is compact
    if seq_len <= 20:
        for ax in axes:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticks(range(seq_len))
            ax.set_yticklabels(tokens, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"biaffine_scores_sent_{sentence_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def run_phase3_verification():
    print("=" * 65)
    print("PHASE 3 VERIFICATION: BIAFFINE SPAN EXTRACTION (BRANCH 1)")
    print("=" * 65)

    train_path = "data/splits/train.jsonl"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    dataset = DimABSADataset(train_path)
    num_samples = min(5, len(dataset))
    print(f"Loaded {len(dataset)} training records. Running Phase 3 on {num_samples} samples...\n")

    aligner = SubwordAligner(tokenizer=None)
    encoder = SwitchAwareEncoder(hidden_dim=128, max_signed_distance=32, use_mock_backbone=True)
    span_extractor = BiaffineAspectOpinionSpanExtractor(input_dim=128, span_dim=64, max_span_length=15)
    span_loss_fn = BiaffineSpanLoss()

    sample_items = [dataset[i] for i in range(num_samples)]
    aligned_list = [aligner.align_single(item["tokens"], item["language_ids"]) for item in sample_items]

    batch = aligner.pad_batch(aligned_list)
    input_ids = torch.tensor(batch["input_ids"])
    attention_mask = torch.tensor(batch["attention_mask"])
    lang_ids_numeric = torch.tensor(batch["subword_lang_ids_numeric"])

    # 1. Forward through Switch-Aware Encoder
    enc_out = encoder(input_ids, lang_ids_numeric, attention_mask=attention_mask)
    H_gated = enc_out["H_gated"]

    # 2. Forward through Biaffine Span Extractor
    span_out = span_extractor(H_gated, attention_mask=attention_mask, threshold=0.0, top_k=3)

    print(f"H_gated Shape:          {H_gated.shape}")
    print(f"Aspect Scores Shape:    {span_out['aspect_scores'].shape}")
    print(f"Opinion Scores Shape:   {span_out['opinion_scores'].shape}")
    print(f"Aspect Repr Shape:      {span_out['aspect_repr'].shape}")
    print(f"Opinion Repr Shape:     {span_out['opinion_repr'].shape}")

    # Map gold spans from Phase 1 to subword space
    batch_gold_aspects = []
    batch_gold_opinions = []

    for b, item in enumerate(sample_items):
        w_to_sub = aligned_list[b]["word_to_subword"]
        gold_asp_list = []
        gold_op_list = []

        for ann in item.get("annotations", []):
            # Aspect mapping
            asp_s, asp_e, asp_ok = map_gold_spans_to_subwords(
                ann.get("aspect_start", -1),
                ann.get("aspect_end", -1),
                w_to_sub,
                is_training_safe=ann.get("aspect_training_safe", True),
            )
            if asp_ok:
                gold_asp_list.append({"start": asp_s, "end": asp_e, "is_training_safe": True, "text": ann["aspect_text"]})

            # Opinion mapping
            op_s, op_e, op_ok = map_gold_spans_to_subwords(
                ann.get("opinion_start", -1),
                ann.get("opinion_end", -1),
                w_to_sub,
                is_training_safe=ann.get("opinion_training_safe", True),
            )
            if op_ok:
                gold_op_list.append({"start": op_s, "end": op_e, "is_training_safe": True, "text": ann["opinion_text"]})

        batch_gold_aspects.append(gold_asp_list)
        batch_gold_opinions.append(gold_op_list)

    # 3. Compute Phase 3 Span Loss
    loss_dict = span_loss_fn(
        aspect_scores=span_out["aspect_scores"],
        opinion_scores=span_out["opinion_scores"],
        valid_span_mask=span_out["valid_span_mask"],
        gold_aspect_spans=batch_gold_aspects,
        gold_opinion_spans=batch_gold_opinions,
    )

    print(f"\nAspect Span Loss:       {loss_dict['aspect_span_loss'].item():.4f}")
    print(f"Opinion Span Loss:      {loss_dict['opinion_span_loss'].item():.4f}")
    print(f"Total Span Loss:        {loss_dict['total_span_loss'].item():.4f}")

    # Display per-sample results & plot heatmaps
    print("\n" + "-" * 65)
    print("PER-SAMPLE SPAN EXTRACTION & DIAGNOSTICS")
    print("-" * 65)

    for b in range(num_samples):
        item = sample_items[b]
        sid = item["sentence_id"]
        subwords = aligned_list[b]["subwords"]

        print(f"\nSentence ID {sid}: {item['text'][:70]}...")
        print(f"  Gold Aspects:     {[ann['aspect_text'] for ann in item.get('annotations', [])]}")
        print(f"  Gold Opinions:    {[ann['opinion_text'] for ann in item.get('annotations', [])]}")
        print(f"  Subword Gold Asp: {batch_gold_aspects[b]}")
        print(f"  Subword Gold Op:  {batch_gold_opinions[b]}")
        print(f"  Decoded Asp:      {span_out['decoded_aspect_spans'][b]}")
        print(f"  Decoded Op:       {span_out['decoded_opinion_spans'][b]}")

        # Plot heatmap
        fig_path = plot_biaffine_score_grid(
            tokens=subwords[:15],
            aspect_scores=span_out["aspect_scores"][b, :15, :15],
            opinion_scores=span_out["opinion_scores"][b, :15, :15],
            valid_mask=span_out["valid_span_mask"][b, :15, :15],
            sentence_id=sid,
        )
        print(f"  Saved heatmap to: {fig_path}")

    print("\n" + "=" * 65)
    print("Phase 3 Biaffine Span Extractor Smoke Test Succeeded!")
    print("=" * 65)


if __name__ == "__main__":
    run_phase3_verification()
