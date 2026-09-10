"""Demonstration and verification script for Phase 4: H-NSG & RGAT.

This script:
1. Tests synthetic Hinglish sentence graph construction and RGAT propagation.
2. Loads real sentences from `data/splits/train.jsonl`.
3. Passes features through HeterogeneousNeuroSymbolicGraph and RGAT.
4. Extracts aspect and opinion graph span representations.
5. Saves diagnostic graph visualizations to `reports/figures/graph/`.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.data.nrc_vad import NRCVADLexicon
from src.data.dependency_parser import HeuristicDependencyBuilder
from src.models.hnsg import (
    HeterogeneousNeuroSymbolicGraph,
    REL_SYNTACTIC,
    REL_SWITCH,
    REL_ASPECT_OPINION,
)
from src.models.rgat import RGAT, extract_graph_span_representation


def plot_graph_visualization(
    tokens: list[str],
    lang_ids: list[int],
    vad_mask: list[bool],
    vad_priors: list[tuple[float, float]],
    adj_matrix: torch.Tensor,
    output_path: Path,
    title: str = "H-NSG Relational Graph",
):
    """Generates a diagnostic NetworkX plot of the heterogeneous graph."""
    G = nx.MultiDiGraph()
    n_tokens = len(tokens)
    
    # Language color map: 0 = Hindi (Light Blue), 1 = English (Light Salmon), 2 = Mixed (Light Green)
    lang_colors = {0: "#a6cee3", 1: "#fb9a99", 2: "#b2df8a", 3: "#d9d9d9"}
    node_colors = []
    labels = {}
    
    for i, tok in enumerate(tokens):
        G.add_node(i)
        lang_id = lang_ids[i] if i < len(lang_ids) else 0
        node_colors.append(lang_colors.get(lang_id, "#e0e0e0"))
        
        has_vad = vad_mask[i] if i < len(vad_mask) else False
        vad_str = f" ({vad_priors[i][0]:.2f}, {vad_priors[i][1]:.2f})" if has_vad else ""
        lang_lbl = "HI" if lang_id == 0 else ("EN" if lang_id == 1 else "MIX")
        labels[i] = f"{tok}\n[{lang_lbl}]{vad_str}"
        
    edge_colors = []
    # Add edges from adj_matrix: shape (3, N, N)
    for i in range(n_tokens):
        for j in range(n_tokens):
            if i == j:
                continue
            # REL_SYNTACTIC (0) -> Blue
            if adj_matrix[REL_SYNTACTIC, i, j] > 0:
                G.add_edge(i, j, rel="syntactic", key="syn")
                edge_colors.append("#1f78b4")
            # REL_SWITCH (1) -> Red/Orange
            if adj_matrix[REL_SWITCH, i, j] > 0 and i < j:
                G.add_edge(i, j, rel="switch", key="sw")
                edge_colors.append("#e31a1c")
            # REL_ASPECT_OPINION (2) -> Forest Green
            if adj_matrix[REL_ASPECT_OPINION, i, j] > 0 and i < j:
                G.add_edge(i, j, rel="ao", key="ao")
                edge_colors.append("#33a02c")
                
    plt.figure(figsize=(12, 6))
    pos = nx.spring_layout(G, seed=42, k=1.2)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, edgecolors="#333333")
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_family="sans-serif")
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors if edge_colors else "#cccccc",
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.15",
        alpha=0.75,
        width=2.0,
    )
    
    # Custom legend
    legend_elements = [
        plt.Line2D([0], [0], color="#1f78b4", lw=2, label="Syntactic Dependency (Rel 0)"),
        plt.Line2D([0], [0], color="#e31a1c", lw=2, label="Code-Switch Bridge (Rel 1)"),
        plt.Line2D([0], [0], color="#33a02c", lw=2, label="Aspect-Opinion Link (Rel 2)"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title(title, fontsize=12, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_phase4_demo():
    print("============================================================")
    print("PHASE 4: H-NSG & RGAT VERIFICATION AND DIAGNOSTICS")
    print("============================================================")
    
    output_dir = Path("reports/figures/graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lexicon = NRCVADLexicon()
    d_model = 64
    hnsg = HeterogeneousNeuroSymbolicGraph(d_model=d_model, lexicon=lexicon)
    rgat = RGAT(d_model=d_model, num_relations=3, num_layers=2, num_heads=4)
    
    # 1. Synthetic Sentences Demonstration
    synthetic_samples = [
        {
            "id": "synthetic_1",
            "text": "Yeh phone ka camera bohot accha hai",
            "tokens": ["yeh", "phone", "ka", "camera", "bohot", "accha", "hai"],
            "lang_ids": [0, 1, 0, 1, 0, 0, 0],  # HI, EN, HI, EN, HI, HI, HI
            "aspect_opinion_pairs": [((3, 3), (5, 5))],
            "aspect_spans": [(3, 3)],
            "opinion_spans": [(5, 5)],
        },
        {
            "id": "synthetic_2",
            "text": "Battery life bahut poor hai but display is fantastic",
            "tokens": ["battery", "life", "bahut", "poor", "hai", "but", "display", "is", "fantastic"],
            "lang_ids": [1, 1, 0, 1, 0, 1, 1, 1, 1],
            "aspect_opinion_pairs": [((0, 1), (3, 3)), ((6, 6), (8, 8))],
            "aspect_spans": [(0, 1), (6, 6)],
            "opinion_spans": [(3, 3), (8, 8)],
        },
    ]
    
    # 2. Real Sentences from train.jsonl
    train_split_path = Path("data/splits/train.jsonl")
    real_samples = []
    if train_split_path.exists():
        with open(train_split_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 3:
                    break
                data = json.loads(line)
                toks = data.get("tokens", data.get("sentence", "").split())
                lid = data.get("language_ids", [0] * len(toks))
                
                # Extract aspect-opinion pairs from quadruplets if available
                pairs = []
                asp_spans = []
                op_spans = []
                for quad in data.get("quadruplets", []):
                    asp_s, asp_e = quad.get("aspect_span", (0, 0))
                    op_s, op_e = quad.get("opinion_span", (0, 0))
                    pairs.append(((asp_s, asp_e), (op_s, op_e)))
                    asp_spans.append((asp_s, asp_e))
                    op_spans.append((op_s, op_e))
                    
                real_samples.append({
                    "id": f"real_train_{idx+1}",
                    "text": data.get("sentence", " ".join(toks)),
                    "tokens": [t.lower() for t in toks],
                    "lang_ids": lid,
                    "aspect_opinion_pairs": pairs,
                    "aspect_spans": asp_spans,
                    "opinion_spans": op_spans,
                })
                
    all_samples = synthetic_samples + real_samples
    print(f"Loaded {len(all_samples)} total sample graphs to process.")
    
    for idx, sample in enumerate(all_samples):
        tokens = sample["tokens"]
        lang_ids = sample["lang_ids"]
        pairs = sample["aspect_opinion_pairs"]
        N = len(tokens)
        
        # Build H-NSG Graph
        adj = hnsg.build_graph(
            tokens_batch=[tokens],
            lang_ids_batch=[lang_ids],
            aspect_opinion_pairs_batch=[pairs],
            max_seq_len=N,
        )  # (1, 3, N, N)
        
        # Extract VAD priors
        vad_priors, vad_mask = hnsg.extract_lexicon_priors([tokens], max_seq_len=N)
        
        # Forward pass through initial projection + RGAT
        H_gated = torch.randn(1, N, d_model)  # Simulated switch-gated representations
        H_init, _, _ = hnsg(H_gated, [tokens], max_seq_len=N)
        H_graph = rgat(H_init, adj)
        
        # Extract graph span representations
        if sample["aspect_spans"]:
            asp_tensor = torch.tensor([sample["aspect_spans"]])
            G_aspect = extract_graph_span_representation(H_graph, asp_tensor, pool_mode="mean")
            print(f"Sample {sample['id']} G_aspect shape: {G_aspect.shape}")
        
        # Plot and save graph visualization
        plot_path = output_dir / f"{sample['id']}_graph.png"
        plot_graph_visualization(
            tokens=tokens,
            lang_ids=lang_ids,
            vad_mask=vad_mask[0].tolist(),
            vad_priors=vad_priors[0].tolist(),
            adj_matrix=adj[0],
            output_path=plot_path,
            title=f"H-NSG: {sample['text']}",
        )
        print(f"Saved graph visualization to: {plot_path}")
        
    print("\nPhase 4 verification script finished successfully!")


if __name__ == "__main__":
    run_phase4_demo()
