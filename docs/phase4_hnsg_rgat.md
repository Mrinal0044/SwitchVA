# Phase 4: Heterogeneous Neuro-Symbolic Graph (H-NSG) & Relational Graph Attention Network (RGAT)

## 1. Overview & Architectural Role

In **NSSG-DimNet** (*Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA*), **Branch 2** models explicit structured relationships across tokens, code-switching transition bridges, aspect-opinion pairings, and external symbolic affective priors using a Heterogeneous Neuro-Symbolic Graph (**H-NSG**) and Relational Graph Attention Network (**RGAT**).

While Branch 1 (Biaffine Aspect-Opinion Span Extractor) extracts boundary candidate spans from switch-gated continuous token representations $H_{\text{gated}}$, Branch 2 constructs a multi-relational graph and applies relation-specific message passing to compute graph-contextualized token embeddings $H_{\text{graph}} \in \mathbb{R}^{B \times N \times d}$ and pooled graph span representations $G_{\text{aspect}}, G_{\text{opinion}}$.

```
               H_gated ∈ R^(B × N × d)
                        │
                        ▼
         ┌───────────────────────────────┐
         │     Symbolic Prior Injection  │ ◄── NRC-VAD Valence-Arousal Prior ([v, a] ∈ [0, 1]²)
         │      + Relation Construction  │ ◄── Dependency Parsing + Switch Points + Spans
         └──────────────┬────────────────┘
                        │
                        ▼
               H_0 = H_gated + W_p * (vad * m)
               Adjacency A ∈ R^(B × R × N × N)  (R = 3 explicit relations)
                        │
                        ▼
         ┌───────────────────────────────┐
         │         L-Layer RGAT          │
         │  (Relation-Specific W_r,      │
         │   Multi-Head Graph Attention, │
         │   LeakyReLU, LayerNorm, Res)  │
         └──────────────┬────────────────┘
                        │
                        ▼
              H_graph ∈ R^(B × N × d)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
Span Pool (Mean/Max)            Span Pool (Mean/Max)
        ▼                               ▼
G_aspect ∈ R^(B × K_a × d)      G_opinion ∈ R^(B × K_o × d)
```

---

## 2. Explicit Relation Types

H-NSG explicitly avoids collapsing edge semantics into a single adjacency matrix. It maintains an adjacency tensor $A \in \mathbb{R}^{B \times 3 \times N \times N}$ across 3 canonical relation types:

| Relation ID | Constant | Name | Description | Connectivity |
|---|---|---|---|---|
| `0` | `REL_SYNTACTIC` | Syntactic Dependency | Head-to-dependent and dependent-to-head syntax edges derived from syntactic dependency trees / heuristic linear/sliding syntax builder. | Directed/Bidirectional syntax edges + self-loops |
| `1` | `REL_SWITCH` | Code-Switching Bridge | Explicit bidirectional bridge edges linking tokens immediately flanking a language switch transition ($l_i \neq l_{i-1}$, e.g., Hindi $\leftrightarrow$ English). | Bidirectional across boundary tokens |
| `2` | `REL_ASPECT_OPINION` | Aspect-Opinion Link | Full bipartite connection between all tokens within an aspect span $A_k$ and all tokens within its corresponding opinion span $O_k$. | Dense bipartite $(i, j) \in A_k \times O_k$ |

---

## 3. Symbolic NRC-VAD Affective Priors & Leakage Prevention

### 3.1 Prior Lookup and Masking
For each token $i$ with surface text $w_i$, the NRC-VAD affective lexicon provides:
$$v_i = \text{NRC-VAD}_{\text{valence}}(w_i) \in [0, 1], \quad a_i = \text{NRC-VAD}_{\text{arousal}}(w_i) \in [0, 1]$$
$$m_i = \mathbb{I}(w_i \in \text{NRC-VAD}) \in \{0, 1\}$$

If $w_i$ is out-of-vocabulary (OOV), $v_i = 0, a_i = 0, m_i = 0$.

### 3.2 Feature Projection
The 2D prior vector $[v_i, a_i]$ is masked and projected to model dimension $d$:
$$p_i = W_{\text{prior}} (m_i \cdot [v_i, a_i]^T) \in \mathbb{R}^d$$
$$H_0 = H_{\text{gated}} + p_i$$

### 3.3 Strict Leakage Prevention
> [!IMPORTANT]
> Ground-truth Continuous Dimensional Valence ($V \in [1, 9]$) and Arousal ($A \in [1, 9]$) from `DimABSA_Final_Dataset_600.csv` are downstream training targets. They are **never** passed into H-NSG or RGAT. The only affective values in Branch 2 are unsupervised, dictionary-level NRC-VAD lexical priors.

---

## 4. Relational Graph Attention Network (RGAT) Formulation

Let $h_i^{(l)} \in \mathbb{R}^d$ be the node representation of token $i$ at layer $l$.

### 4.1 Relation-Specific Projection
For each relation $r \in \{0, 1, 2\}$, node features are transformed by a dedicated linear projection matrix $W_r^{(l)} \in \mathbb{R}^{d \times d}$:
$$z_{i, r}^{(l)} = W_r^{(l)} h_i^{(l)}$$

### 4.2 Multi-Head Relational Attention
For attention head $m \in \{1, \dots, M\}$ (with head dimension $d_h = d / M$):
$$\tilde{e}_{ij, r}^{(l, m)} = \text{LeakyReLU}\left( \mathbf{a}_{r, \text{src}}^{(m) T} z_{i, r}^{(l, m)} + \mathbf{a}_{r, \text{dst}}^{(m) T} z_{j, r}^{(l, m)} \right)$$

For each node $i$, attention coefficients $\alpha_{ij, r}^{(l, m)}$ are normalized over all active incoming neighbors and active relation types using masked softmax:
$$\alpha_{ij, r}^{(l, m)} = \frac{\exp(\tilde{e}_{ij, r}^{(l, m)})}{\sum_{r' \in \mathcal{R}} \sum_{k \in \mathcal{N}_{r'}(i)} \exp(\tilde{e}_{ik, r'}^{(l, m)})}$$

### 4.3 Message Aggregation and Residual Update
$$\tilde{h}_i^{(l+1)} = \Big\|_{m=1}^M \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij, r}^{(l, m)} z_{j, r}^{(l, m)}$$
$$h_i^{(l+1)} = \text{LayerNorm}\left( h_i^{(l)} + \text{Dropout}(\text{GELU}(W_o \tilde{h}_i^{(l+1)})) \right)$$

---

## 5. Span Representation Extraction

For an extracted or gold span $[s, e]$, the graph span representation is pooled from the final RGAT node states $H_{\text{graph}} = h^{(L)}$:
$$G_{\text{span}} = \text{Pool}\left( \{ h_k^{(L)} \}_{k=s}^e \right)$$

Supported pooling modes:
- `mean`: Average of token representations in $[s, e]$.
- `max`: Element-wise maximum across $[s, e]$.
- `endpoints`: Concatenation $[h_s^{(L)}; h_e^{(L)}]$ projected back to $d$.
- `span_rep`: Span boundary concatenation with mean context $[h_s; h_e; \bar{h}]$ projected to $d$.

---

## 6. Tensor Shape Contract

| Tensor | Shape | Meaning |
|---|---|---|
| `H_gated` | `(B, N, d)` | Token representations from SP-GSA |
| `vad_priors` | `(B, N, 2)` | Normalized valence-arousal priors in $[0, 1]$ |
| `vad_mask` | `(B, N)` | Boolean mask for lexicon availability |
| `adj_matrix` | `(B, 3, N, N)` | Multi-relational adjacency tensor |
| `attention_mask` | `(B, N)` | Padding mask (1 = token, 0 = pad) |
| `H_graph` | `(B, N, d)` | Contextualized graph token states |
| `aspect_spans` | `(B, K_a, 2)` | Aspect span token indices $[s, e]$ |
| `G_aspect` | `(B, K_a, d)` | Aspect graph span embeddings |
| `opinion_spans` | `(B, K_o, 2)` | Opinion span token indices $[s, e]$ |
| `G_opinion` | `(B, K_o, d)` | Opinion graph span embeddings |
