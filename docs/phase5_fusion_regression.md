# Phase 5: Aspect-Guided Mutual Cross-Attention, Fused Representation $Z \in \mathbb{R}^{3d}$, & Continuous VA Regression

## 1. Overview & Role in NSSG-DimNet

Phase 5 implements the final representation fusion and continuous prediction layers of **NSSG-DimNet** (*Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA*).

It brings together:
- **Branch 1**: Biaffine Aspect-Opinion Span Extractor representations ($H_{\text{aspect}}, H_{\text{opinion}} \in \mathbb{R}^d$)
- **Branch 2**: Heterogeneous Neuro-Symbolic Graph + RGAT representations ($G_{\text{aspect}}, G_{\text{opinion}} \in \mathbb{R}^d$ and $H_{\text{graph}} \in \mathbb{R}^{N \times d}$)

via **Aspect-Guided Mutual Cross-Attention**, constructing an exact $3d$-dimensional fused span representation:
$$Z = [Z_{\text{span}} \,\|\, Z_{\text{graph}} \,\|\, Z_{\text{cross}}] \in \mathbb{R}^{3d}$$
which is then mapped to continuous Valence ($\hat{V} \in [0, 1]$) and Arousal ($\hat{A} \in [0, 1]$) scores using independent dual MLP regression heads.

```
Branch 1: Biaffine Span Repr                 Branch 2: RGAT Graph Repr
(H_aspect, H_opinion ∈ R^d)                  (G_aspect, G_opinion ∈ R^d)
            │                                             │
            ├──────────────────────┬──────────────────────┤
            │                      │                      │
            ▼                      ▼                      ▼
  Aspect Guidance Query      Path A (Span -> Graph) Path B (Graph -> Span)
  Q_asp = W_q * H_aspect     A_sg = CrossAttn(S, G) A_gs = CrossAttn(G, S)
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                                   ▼
                   Z_cross = LayerNorm(W_c [A_sg ; A_gs]) ∈ R^d
                   Z_span  = LayerNorm(W_s [H_aspect ; H_opinion]) ∈ R^d
                   Z_graph = LayerNorm(W_g [G_aspect ; G_opinion]) ∈ R^d
                                   │
                                   ▼
                   Z = [Z_span || Z_graph || Z_cross] ∈ R^(3d)
                   Assertion: Z.shape[-1] == 3 * d
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
             Valence Regressor             Arousal Regressor
          Linear -> GELU -> Linear      Linear -> GELU -> Linear
                 -> Sigmoid                    -> Sigmoid
                    │                             │
                    ▼                             ▼
             Valence ∈ [0, 1]              Arousal ∈ [0, 1]
```

---

## 2. Mathematical Formulations

### 2.1 Aspect-Guided Mutual Cross-Attention
Unlike generic sentence-level pooling or unidirectional cross-attention, NSSG-DimNet implements bidirectional mutual cross-attention conditioned explicitly on the target aspect representation:

1. **Aspect Guidance Conditioning**:
   $$q_{\text{guide, span}} = \text{MLP}_{\text{guide}}([H_{\text{aspect}} \,\|\, G_{\text{aspect}}])$$
   $$q_{\text{guide, graph}} = \text{MLP}_{\text{guide}}([G_{\text{aspect}} \,\|\, H_{\text{aspect}}])$$

2. **Path A (Span Branch attending to Graph Memory)**:
   $$Q_A = W_{q,1} q_{\text{guide, span}}, \quad K_A = W_{k,1} M_{\text{graph}}, \quad V_A = W_{v,1} M_{\text{graph}}$$
   $$\text{Attn}_A = \text{Softmax}\left(\frac{Q_A K_A^T}{\sqrt{d_k}}\right) V_A$$

3. **Path B (Graph Branch attending to Span Memory)**:
   $$Q_B = W_{q,2} q_{\text{guide, graph}}, \quad K_B = W_{k,2} M_{\text{span}}, \quad V_B = W_{v,2} M_{\text{span}}$$
   $$\text{Attn}_B = \text{Softmax}\left(\frac{Q_B K_B^T}{\sqrt{d_k}}\right) V_B$$

4. **Mutual Fusion ($Z_{\text{cross}} \in \mathbb{R}^d$)**:
   $$Z_{\text{cross}} = \text{LayerNorm}\left(H_{\text{aspect}} + W_{\text{cross}} [\text{Attn}_A \,\|\, \text{Attn}_B]\right)$$

### 2.2 Exact 3d Fused Span Representation $Z$
The fused representation $Z$ comprises three distinct $d$-dimensional vectors:
- **$Z_{\text{span}} \in \mathbb{R}^d$**: Fused span-boundary representation from Branch 1:
  $$Z_{\text{span}} = \text{LayerNorm}\left(\text{GELU}(W_{\text{span}} [H_{\text{aspect}} \,\|\, H_{\text{opinion}}])\right)$$
- **$Z_{\text{graph}} \in \mathbb{R}^d$**: Fused graph-aware node representation from Branch 2:
  $$Z_{\text{graph}} = \text{LayerNorm}\left(\text{GELU}(W_{\text{graph}} [G_{\text{aspect}} \,\|\, G_{\text{opinion}}])\right)$$
- **$Z_{\text{cross}} \in \mathbb{R}^d$**: Mutual cross-attended context between both branches.

Final concatenation:
$$Z = [Z_{\text{span}} \,\|\, Z_{\text{graph}} \,\|\, Z_{\text{cross}}] \in \mathbb{R}^{3d}$$
with runtime dimension check:
$$\text{dim}(Z) = 3d$$

### 2.3 Independent Valence and Arousal Regressors
Valence and Arousal represent distinct emotional dimensions and use independent parameter weights:
$$\hat{V} = \text{Sigmoid}\left(W_{v2} \text{GELU}(W_{v1} Z + b_{v1}) + b_{v2}\right) \in [0, 1]$$
$$\hat{A} = \text{Sigmoid}\left(W_{a2} \text{GELU}(W_{a1} Z + b_{a1}) + b_{a2}\right) \in [0, 1]$$

---

## 3. Loss Functions & Optimization Objective

### 3.1 Concordance Correlation Coefficient (CCC) Loss
$$\text{CCC}(y, \hat{y}) = \frac{2 \cdot \text{Cov}(y, \hat{y})}{\text{Var}(y) + \text{Var}(\hat{y}) + (\mu_y - \mu_{\hat{y}})^2 + \epsilon}$$
$$\mathcal{L}_{\text{CCC}}(y, \hat{y}) = 1.0 - \text{CCC}(y, \hat{y})$$

Safeguards:
- Handles degenerate $N < 2$ batches without NaN.
- Zero-variance detection to avoid division by zero.
- Bounded in $[0, 2]$.

### 3.2 Joint Total Loss
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ccc}} (\mathcal{L}_{\text{CCC}, V} + \mathcal{L}_{\text{CCC}, A}) + \lambda_{\text{smooth}} (\mathcal{L}_{\text{SmoothL1}, V} + \mathcal{L}_{\text{SmoothL1}, A}) + \lambda_{\text{span}} \mathcal{L}_{\text{span}}$$

---

## 4. Strict Target-Leakage Prevention
> [!IMPORTANT]
> Ground-truth continuous $(V, A)$ labels from `DimABSA_Final_Dataset_600.csv` are used **strictly** as optimization targets in `JointVALoss`. They are never passed as inputs to the transformer backbone, switch embedding, SP-GSA, Biaffine span extractor, H-NSG, RGAT, mutual cross-attention, or regression heads.
