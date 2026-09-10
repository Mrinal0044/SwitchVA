#!/usr/bin/env python3
"""Phase 2 Verification Script: Synthetic and Real Dataset Encoding Smoke Tests."""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.tokenizer_alignment import SubwordAligner, SUBWORD_LANG_MAP
from src.models.switch_encoder import SwitchAwareEncoder, inspect_switch_gates
from src.data.dataset import DimABSADataset


def run_synthetic_test():
    print("=" * 60)
    print("1. SYNTHETIC FORWARD TEST")
    print("=" * 60)

    encoder = SwitchAwareEncoder(
        hidden_dim=128,
        max_signed_distance=16,
        use_mock_backbone=True,
    )
    aligner = SubwordAligner(tokenizer=None)

    # Example 1: Multi-switch sentence
    words1 = ["phone", "ka", "camera", "amazing", "hai", "but", "battery", "weak", "hai"]
    langs1 = ["EN", "HI", "EN", "EN", "HI", "EN", "EN", "EN", "HI"]

    # Example 2: Monolingual / single-switch
    words2 = ["yeh", "video", "bahut", "achhi", "hai"]
    langs2 = ["HI", "EN", "HI", "HI", "HI"]

    aligned1 = aligner.align_single(words1, langs1)
    aligned2 = aligner.align_single(words2, langs2)

    batch = aligner.pad_batch([aligned1, aligned2])

    input_ids = torch.tensor(batch["input_ids"])
    attention_mask = torch.tensor(batch["attention_mask"])
    lang_ids_numeric = torch.tensor(batch["subword_lang_ids_numeric"])

    print(f"Batch Input Shape:           {input_ids.shape}")
    print(f"Attention Mask Shape:        {attention_mask.shape}")
    print(f"Numeric Language IDs Shape:  {lang_ids_numeric.shape}")

    # Forward pass
    outputs = encoder(input_ids, lang_ids_numeric, attention_mask=attention_mask)

    print(f"Transformer Output (H):      {outputs['H'].shape}")
    print(f"Switch Embedding (E_switch): {outputs['E_switch'].shape}")
    print(f"SP-GSA Gated Output (H_gated): {outputs['H_gated'].shape}")
    print(f"Gate Tensor (G) Shape:       {outputs['gate_values'].shape}")

    # Verify no NaN or Inf
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v).any(), f"NaN in {k}"
            assert not torch.isinf(v).any(), f"Inf in {k}"

    print("\nDiagnostic Table for Synthetic Sentence 1:")
    subwords = aligned1["subwords"]
    sub_langs = aligned1["subword_lang_ids"]
    dists = outputs["signed_distances"][0, :len(subwords)]
    gates = outputs["gate_values"][0, :len(subwords)]

    diag_table = inspect_switch_gates(subwords, sub_langs, dists, gates)
    print(diag_table)
    print("=" * 60 + "\n")


def run_actual_dataset_test():
    print("=" * 60)
    print("2. ACTUAL DATASET FORWARD TEST (train.jsonl sample)")
    print("=" * 60)

    train_path = "data/splits/train.jsonl"
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run Phase 1 preprocessing first.")
        return

    dataset = DimABSADataset(train_path)
    num_samples = min(5, len(dataset))
    print(f"Loaded {len(dataset)} training records. Testing first {num_samples} records...\n")

    aligner = SubwordAligner(tokenizer=None)
    encoder = SwitchAwareEncoder(
        hidden_dim=128,
        max_signed_distance=32,
        use_mock_backbone=True,
    )

    sample_items = [dataset[i] for i in range(num_samples)]
    aligned_list = [
        aligner.align_single(item["tokens"], item["language_ids"])
        for item in sample_items
    ]

    batch = aligner.pad_batch(aligned_list)
    input_ids = torch.tensor(batch["input_ids"])
    attention_mask = torch.tensor(batch["attention_mask"])
    lang_ids_numeric = torch.tensor(batch["subword_lang_ids_numeric"])

    outputs = encoder(input_ids, lang_ids_numeric, attention_mask=attention_mask)

    print(f"Processed Batch Shape:       {input_ids.shape}")
    print(f"H_gated Output Shape:        {outputs['H_gated'].shape}")
    print(f"Mean Gate Magnitude:         {outputs['gate_values'][attention_mask == 1].mean().item():.4f}")

    for idx in range(num_samples):
        item = sample_items[idx]
        subwords = aligned_list[idx]["subwords"]
        sub_langs = aligned_list[idx]["subword_lang_ids"]
        dists = outputs["signed_distances"][idx, :len(subwords)]
        gates = outputs["gate_values"][idx, :len(subwords)]

        print(f"\n--- Sentence ID {item['sentence_id']} ---")
        print(f"Text: {item['text'][:80]}...")
        diag = inspect_switch_gates(subwords[:10], sub_langs[:10], dists[:10], gates[:10])
        print(diag)

    print("\n" + "=" * 60)
    print("Phase 2 Encoding Smoke Test Succeeded!")
    print("=" * 60)


if __name__ == "__main__":
    run_synthetic_test()
    run_actual_dataset_test()
