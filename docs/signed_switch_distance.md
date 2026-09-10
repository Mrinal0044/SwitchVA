# Signed Switch-Distance Specification

**Mathematical Definition and Directionality Convention in NSSG-DimNet**

---

## 1. Motivation

In code-switched Hinglish text, sentiment polarity and affective shifts often correlate strongly with language switching points (e.g. switching from Hindi context to English opinion or vice-versa). Standard absolute positional embeddings lack awareness of code-switching transitions.

**Signed Switch-Distance** encodes both the **proximity** and the **direction** of each token relative to the nearest code-switch point in the sequence.

---

## 2. Mathematical Definition

Let $X = (x_1, x_2, \dots, x_N)$ be a sequence of $N$ subword/word tokens with associated language identifiers $L = (l_1, l_2, \dots, l_N)$, where $l_i \in \{\text{HI}, \text{EN}, \text{OTHER}, \text{SPECIAL}, \text{PAD}\}$.

### 2.1 Switch-Point Set $\mathcal{S}$
Let $\mathcal{V} \subseteq \{1, \dots, N\}$ denote the set of valid content token indices (excluding special tokens $\text{SPECIAL}$ and padding tokens $\text{PAD}$).
A **switch point** occurs at token index $s \in \mathcal{V}$ where the language changes relative to the preceding valid token:
$$\mathcal{S} = \{s_k \in \mathcal{V} \mid l_{s_k} \neq l_{\text{prev}(s_k)}\}$$
where $\text{prev}(s_k)$ is the immediately preceding valid content token in the sequence.

### 2.2 Nearest Switch Point $s^*(i)$
For any active token at position $i$:
$$s^*(i) = \arg\min_{s \in \mathcal{S}} |i - s|$$

### 2.3 Directional Signed Distance $d_i$
The signed distance from token $i$ to its nearest switch point is defined as:
$$d_i = i - s^*(i)$$

---

## 3. Directionality & Sign Convention

| Token Relative Position | Sign of $d_i$ | Meaning | Example |
| :--- | :--- | :--- | :--- |
| **Before Switch Point** | $d_i < 0$ | Token occurs in the language regime leading up to the switch | $d_i = -2$ (2 tokens before switch) |
| **At Switch Point** | $d_i = 0$ | First token of the new language regime | $d_i = 0$ (switch boundary token) |
| **After Switch Point** | $d_i > 0$ | Token occurs in the language regime following the switch | $d_i = +2$ (2 tokens after switch) |

### Concrete Example:
```
Token:      ham     bowling     krte    hai     aap     apne    stump   bhjao
Language:   HI      EN          HI      HI      HI      HI      EN      HI
Index (i):  0       1           2       3       4       5       6       7
Switch (S):         [1]         [2]                             [6]     [7]

SignedDist: -1      0           0       +1      -2      -1      0       0
```
*(Here indices 1, 2, 6, 7 are switch points where language transitions occur)*.

---

## 4. Boundary, Clipping, and Masking Rules

1. **Distance Clipping**:
   $$d_i^{\text{clamped}} = \max(-D_{max}, \min(D_{max}, d_i))$$
   where $D_{max}$ is a configurable threshold (default $D_{max} = 32$).

2. **Embedding Index Shift**:
   To index a non-negative embedding table of size $2 D_{max} + 3$:
   $$\text{idx}(i) = d_i^{\text{clamped}} + D_{max} + 1 \quad \in [1, 2 D_{max} + 1]$$

3. **No-Switch Sequence**:
   If a sequence contains zero switch points (monolingual text):
   $$\text{idx}(i) = 0 \quad (\text{NO\_SWITCH\_INDEX})$$

4. **Padding Positions**:
   Padding tokens are assigned a dedicated index:
   $$\text{idx}(i) = 2 D_{max} + 2 \quad (\text{PAD\_INDEX})$$
   with `padding_idx` configured so that padding vectors contribute zero gradients.
