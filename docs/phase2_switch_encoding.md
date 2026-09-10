# Phase 2: Switch-Aware Encoding & SP-GSA Specification

**Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA**

---

## 1. Overview & Architecture Path

Phase 2 implements the complete switch-aware encoding front-end of NSSG-DimNet:

```
[Tokenized Hinglish sentence] + [Language IDs] + [Signed Switch Distance]
                                     │
                                     ▼
        ┌────────────────────────────┴────────────────────────────┐
        ▼                                                         ▼
[Multilingual Transformer Backbone]                     [Switch Embedding]
(HingRoBERTa / mDeBERTa)                                (Language ID + Signed Distance)
H ∈ R^(B × N × d)                                       E_switch ∈ R^(B × N × d)
        │                                                         │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                  [Switch-Point Gated Self-Attention (SP-GSA)]
                  G = sigmoid(W_g [H || E_switch] + b_g)
                  H_gated = LayerNorm(G ⊙ H + H)
                  H_gated ∈ R^(B × N × d)
```

---

## 2. Components Implemented

### 2.1 Language-ID Source & Subword Alignment
- **Source of Truth**: Ground-truth language labels from Phase 1 (`HI`, `EN`).
- **Subword Alignment (`SubwordAligner`)**:
  - Subwords split from words inherit their parent word's language ID.
  - Special tokens (`<s>`, `</s>`, `[CLS]`, `[SEP]`) are assigned `SPECIAL` (ID 3).
  - Padding tokens are assigned `PAD` (ID 4).
  - Maintains bidirectional mappings `word_to_subword` and `subword_to_word`.

### 2.2 Signed Switch-Distance (`SignedSwitchDistance`)
- Computes directional distance $d_i = i - s^*$ to nearest switch point $s^*$.
- $d_i < 0$: leading into switch; $d_i = 0$: at switch boundary; $d_i > 0$: following switch.
- Clamped to $[-D_{max}, D_{max}]$ and mapped to shifted embedding indices.

### 2.3 Switch Embedding (`SwitchEmbedding`)
- Fuses Language ID embedding $E_{lang} \in \mathbb{R}^{d_{lang}}$ and Distance embedding $E_{dist} \in \mathbb{R}^{d_{dist}}$.
- Linear projection + LayerNorm + Dropout $\rightarrow E_{switch} \in \mathbb{R}^{B \times N \times d}$.

### 2.4 Multilingual Transformer Backbone (`MultilingualBackbone`)
- Standardized wrapper supporting:
  - `l3cube-pune/hing-roberta`
  - `microsoft/mdeberta-v3-base`
- Produces token contextual representation $H \in \mathbb{R}^{B \times N \times d}$.
- Supports layer freezing and offline mock transformer execution.

### 2.5 Switch-Point Gated Self-Attention (`SPGSA`)
- Exact formulation:
  $$G = \sigma(W_g [H \,\|\, E_{switch}] + b_g)$$
  $$H_{gated} = \text{LayerNorm}(G \odot H + H)$$
- Preserves residual connection and applies LayerNorm.
- Attention masking zeros out padding positions in both $H_{gated}$ and $G$.

### 2.6 Diagnostic Inspection (`inspect_switch_gates`)
- Formats diagnostic tables showing token, language, signed switch distance, and mean gate activation magnitude.

---

## 3. Tensor Dimensions Summary

| Tensor / Variable | Shape | Description |
| :--- | :--- | :--- |
| `input_ids` | $(B, N)$ | Subword token IDs |
| `attention_mask` | $(B, N)$ | Binary attention mask ($1 = \text{valid}, 0 = \text{pad}$) |
| `lang_ids_numeric` | $(B, N)$ | Language ID indices |
| `signed_distances` | $(B, N)$ | Raw signed distances in $[-D_{max}, D_{max}]$ |
| `distance_indices` | $(B, N)$ | Shifted distance embedding indices in $[0, 2 D_{max} + 2]$ |
| $H$ | $(B, N, d)$ | Raw multilingual transformer output |
| $E_{switch}$ | $(B, N, d)$ | Trainable switch embedding output |
| $G$ | $(B, N, d)$ | Element-wise switch gate tensor in $[0, 1]$ |
| $H_{gated}$ | $(B, N, d)$ | Modulated output of SP-GSA |
