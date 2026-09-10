# Phase 3: Biaffine Aspect-Opinion Span Extractor (Branch 1)

**Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA**

---

## 1. Motivation

Traditional Aspect-Based Sentiment Analysis models often rely on sequence-tagging approaches (such as BIO schemes) which suffer from:
- Inability to naturally model overlapping or multi-token discontinuous aspects and opinions.
- Error propagation across sequential transitions in code-switched Hindi-English sentences.

The **Biaffine Aspect-Opinion Span Extractor** models span boundaries directly on a 2D candidate matrix $(N \times N)$, enabling the simultaneous, robust scoring and extraction of multiple candidate aspect spans and opinion spans per sentence.

---

## 2. Mathematical Formulation

Given switch-gated contextual representations $H_{gated} \in \mathbb{R}^{B \times N \times d}$ from Phase 2:

### 2.1 Boundary Representation MLPs
Start and end boundary projections are computed separately for Aspects and Opinions:
$$h_{i}^{(asp, start)} = \text{MLP}_{asp\_start}(H_{gated, i}) \in \mathbb{R}^{d_{span}}$$
$$h_{j}^{(asp, end)} = \text{MLP}_{asp\_end}(H_{gated, j}) \in \mathbb{R}^{d_{span}}$$
$$h_{i}^{(op, start)} = \text{MLP}_{op\_start}(H_{gated, i}) \in \mathbb{R}^{d_{span}}$$
$$h_{j}^{(op, end)} = \text{MLP}_{op\_end}(H_{gated, j}) \in \mathbb{R}^{d_{span}}$$

### 2.2 2D Biaffine Scoring Grid
For each token pair $(i, j)$ representing candidate span $[i, j]$:
$$\text{Score}_{asp}(i, j) = {h_{i}^{(asp, start)}}^T U_{asp} h_{j}^{(asp, end)} + W_{asp} [h_{i}^{(asp, start)} \,\|\, h_{j}^{(asp, end)}] + b_{asp}$$
$$\text{Score}_{op}(i, j) = {h_{i}^{(op, start)}}^T U_{op} h_{j}^{(op, end)} + W_{op} [h_{i}^{(op, start)} \,\|\, h_{j}^{(op, end)}] + b_{op}$$

where $U_{asp}, U_{op} \in \mathbb{R}^{d_{span} \times d_{span}}$ are learnable bilinear weight matrices, $W_{asp}, W_{op} \in \mathbb{R}^{2 d_{span}}$ are linear weights, and $b_{asp}, b_{op} \in \mathbb{R}$ are scalar biases.

---

## 3. Valid Span Masking & Subword Mapping

1. **Span Validity Constraint**:
   A candidate $(i, j)$ is valid if and only if:
   - $i \le j$ (start precedes or equals end)
   - $j - i + 1 \le \text{max\_span\_length}$ (default 15)
   - $x_i, x_j$ are valid non-padding tokens.

2. **Word-to-Subword Gold Span Mapping**:
   Phase 1 gold spans $[w_{start}, w_{end}]$ are converted to subword ranges $[s_{sub}, e_{sub}]$ using `word_to_subword`:
   $$s_{sub} = \min(\text{word\_to\_subword}[w_{start}]), \quad e_{sub} = \max(\text{word\_to\_subword}[w_{end}])$$
   
3. **Training Safety Policy**:
   - **Exact / Normalized Matches (`training_safe = True`)**: Assigned binary label $1.0$ at $(s_{sub}, e_{sub})$ in the gold grid.
   - **Approximate / Unmatched Matches (`training_safe = False`)**: Excluded from negative loss calculation to avoid penalizing uncertain spans.

---

## 4. Span-Level Loss

$$\mathcal{L}_{span} = \lambda_{asp} \text{BCEWithLogits}(\text{Score}_{asp}[M], \text{Target}_{asp}[M]) + \lambda_{op} \text{BCEWithLogits}(\text{Score}_{op}[M], \text{Target}_{op}[M])$$
where $M$ is the boolean mask of valid supervised candidate cells.

---

## 5. Span Representation Extraction

For each decoded or target span $[i, j]$:
$$\text{SpanRepr} = \text{LayerNorm}(\text{GELU}(W_{repr} [H_{gated, i} \,\|\, H_{gated, j}])) \in \mathbb{R}^{d}$$
providing the aspect representation $H_{aspect} \in \mathbb{R}^d$ and opinion representation $H_{opinion} \in \mathbb{R}^d$ for downstream cross-attention and graph integration.
