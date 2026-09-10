# Evaluation Dataflow: Mode A vs Mode B

This document explicitly traces the dataflow differences between **Mode A (End-to-End Predicted Spans)** and **Mode B (Gold-Span Oracle)** in NSSG-DimNet.

---

## 1. Architectural Component Comparison

| Component / Stage | Mode A: End-to-End Predicted Spans (Real-World) | Mode B: Gold-Span Oracle (Isolated Regression) |
|---|---|---|
| **Input Sentence & Language IDs** | Unaltered test sentence tokens & language IDs | Unaltered test sentence tokens & language IDs |
| **Encoder Backbone & SP-GSA** | Computes $H_{\text{enc}}$ and $H_{\text{gated}}$ | Computes $H_{\text{enc}}$ and $H_{\text{gated}}$ |
| **Biaffine Span Extraction** | Predicts 2D candidate score grids $S_{\text{asp}}, S_{\text{op}}$ | Predicts 2D candidate score grids $S_{\text{asp}}, S_{\text{op}}$ |
| **Span Coordinate Source** | **Model-predicted** top decoded spans | **Gold ground-truth** boundary annotations |
| **Aspect Representation ($h_{\text{asp}}$)** | Pooled from top predicted aspect span $[s_{\text{pred}}, e_{\text{pred}}]$ | Pooled from gold aspect span $[s_{\text{gold}}, e_{\text{gold}}]$ |
| **Opinion Representation ($h_{\text{op}}$)** | Pooled from top predicted opinion span $[s_{\text{pred}}, e_{\text{pred}}]$ | Pooled from gold opinion span $[s_{\text{gold}}, e_{\text{gold}}]$ |
| **H-NSG Graph Nodes** | Token nodes initialized from $H_{\text{gated}}$ | Token nodes initialized from $H_{\text{gated}}$ |
| **RGAT Span Pooling** | Pooled over top predicted span boundaries | Pooled over gold span boundaries |
| **Cross-Attention Queries** | Conditioned on predicted span vector $h_{\text{asp}}^{\text{pred}}$ | Conditioned on gold span vector $h_{\text{asp}}^{\text{gold}}$ |
| **Fused Representation $Z \in \mathbb{R}^{3d}$** | Derived from predicted span branches | Derived from gold span branches |
| **Regression Inputs** | $Z_{\text{pred}}$ | $Z_{\text{gold}}$ |
| **Valence & Arousal Output** | Reflects joint extraction + regression quality | Reflects isolated dimensional regression quality |

---

## 2. Mode A: End-to-End Execution Flow

```
TEST SENTENCE (Tokens, Lang IDs)
        ↓
Switch Embedding + Multilingual Backbone
        ↓
Switch-Point Gated Self-Attention (SP-GSA) → H_gated ∈ R^(B × N × d)
        ↓
Biaffine Span Extractor:
  - Start/End boundary MLPs
  - Bilinear + Linear scoring
  - Greedy decoding without gold spans
        ↓
Decoded Predicted Spans:
  - Aspect: p_asp = [(s1, e1), (s2, e2), ...]
  - Opinion: p_op = [(s1, e1), (s2, e2), ...]
        ↓ -----------------------------------------> [Span Evaluation: Exact Match vs Gold]
Top Predicted Span Pooling:
  - h_asp = Linear([H_gated[s_top_asp] || H_gated[e_top_asp]])
  - h_op  = Linear([H_gated[s_top_op]  || H_gated[e_top_op]])
        ↓
H-NSG + NRC-VAD Priors + RGAT Relational Graph Attention
        ↓
Aspect-Guided Mutual Cross-Attention:
  - Query: h_asp (predicted)
  - Key/Value: Graph token features
        ↓
Fused Representation Z_pred ∈ R^(3d)
        ↓
Independent Valence & Arousal Linear Regression Heads
        ↓
Predicted Valence & Arousal (Continuous) ----------> [VA Evaluation: Pearson, MAE, RMSE, CCC]
```

---

## 3. Mode B: Gold-Span Oracle Execution Flow

```
TEST SENTENCE + GOLD SPANS (g_asp, g_op)
        ↓
Switch Embedding + Multilingual Backbone
        ↓
Switch-Point Gated Self-Attention (SP-GSA) → H_gated ∈ R^(B × N × d)
        ↓
Gold Span Boundary Pooling (Teacher Forced):
  - h_asp_gold = Linear([H_gated[g_asp_start] || H_gated[g_asp_end]])
  - h_op_gold  = Linear([H_gated[g_op_start]  || H_gated[g_op_end]])
        ↓
H-NSG + NRC-VAD Priors + RGAT Relational Graph Attention
  - Span pooling over gold coordinates
        ↓
Aspect-Guided Mutual Cross-Attention:
  - Query: h_asp_gold
  - Key/Value: Graph token features
        ↓
Fused Representation Z_oracle ∈ R^(3d)
        ↓
Independent Valence & Arousal Linear Regression Heads
        ↓
Predicted Valence & Arousal (Continuous) ----------> [Oracle VA Evaluation: Pearson, MAE, RMSE, CCC]
```

---

## 4. Verification of Separation and Isolation

1. In Mode A, `forward_kwargs["aspect_spans"] = None` and `forward_kwargs["opinion_spans"] = None`. No gold boundaries enter the network.
2. In Mode B, gold coordinates are strictly passed to evaluate the upper-bound capacity of the dimensional regression heads given ground-truth localization.
3. No cross-contamination: Mode A metrics are computed strictly using model predictions without fallback.
