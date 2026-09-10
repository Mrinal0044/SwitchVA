# Phase 7A: NSSG-DimNet Training and Inference Diagnostic Audit Report

**Project**: NSSG-DimNet (*Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA*)  
**Phase**: Phase 7A Diagnostic Audit  
**Date**: September 11, 2026  
**Auditor**: Antigravity Research Verification System  

---

## Executive Summary

Following the research evaluation audit (which corrected an evaluation fallback issue and established genuine unguided metrics: Aspect F1 = 0.0085, Opinion F1 = 0.0064, End-to-End Valence $r = 0.0931$, Arousal $r = 0.0532$), this **Diagnostic Audit** investigated the underlying training and inference pipelines to determine why span extraction performance is low.

### Primary Diagnostic Discovery
The poor span extraction performance is caused by **two confirmed implementation bugs in the training pipeline**:
1. **Auxiliary Span Loss Was Never Computed During Training (`train_loss_span = 0.0000`)**:  
   In [`src/training/data_utils.py`](file:///Users/kmrinal/SwitchVA/src/training/data_utils.py), the batch collator `dimabsa_tensor_collate_fn` never constructed or returned `span_targets`. In [`src/training/trainer.py`](file:///Users/kmrinal/SwitchVA/src/training/trainer.py), `self.loss_fn` received `span_targets=None`. In [`src/losses/va_loss.py`](file:///Users/kmrinal/SwitchVA/src/losses/va_loss.py), the condition:
   ```python
   if span_logits is not None and span_targets is not None and self.span_loss_weight > 0.0:
   ```
   evaluated to `False` on **100% of batches across all 25 training epochs**.
2. **Zero Gradient Flow to Biaffine Scorer and Projections (`norm = 0.000000`)**:  
   During Phase 6 training, gold spans were passed to the model. When explicit spans are provided, `BiaffineSpanExtractor.extract_span_representation` slices directly from $H_{\text{gated}}$ using `span_repr_proj`, **completely bypassing** the biaffine score grids. Because `loss_span = 0.0`, the modules `aspect_start_mlp`, `aspect_end_mlp`, `aspect_scorer`, `opinion_start_mlp`, `opinion_end_mlp`, and `opinion_scorer` received **EXACTLY ZERO GRADIENT** during training. The biaffine extractor evaluated at test time was using **untrained, random Xavier-initialized weights**.

---

## 1. Audit 1 — Gold Span Grid Verification

Inspected on 10 training samples from `data/splits/train.jsonl`:

| Sample / Sentence ID | Aspect Text | Word Span | Subword Span | Grid Positive Cell | Opinion Text | Word Span | Subword Span | Grid Positive Cell | Start $\le$ End | Grid Match |
|---|---|---|---|---|---|---|---|---|---|---|
| **#1 (ID: 3)** | `"shehnaaz ki entry"` | [21, 23] | [22, 24] | [[22, 24]] | `"us din se bb13 ke..."` | [25, 33] | [26, 34] | [[26, 34]] | True | **EXACT** |
| **#2 (ID: 3)** | `"shehnaaz"` | [21, 21] | [22, 22] | [[22, 22]] | `"i miss you shehnaaz"` | [46, 49] | [47, 50] | [[47, 50]] | True | **EXACT** |
| **#3 (ID: 3)** | `"bb13"` | [4, 4] | [5, 5] | [[5, 5]] | `"miss you bb13"` | [51, 53] | [52, 54] | [[52, 54]] | True | **EXACT** |
| **#4 (ID: 4)** | `"nagan sarir ko pradarshan"` | [7, 10] | [8, 11] | [[8, 11]] | `"sobha nahi deta"` | [12, 14] | [13, 15] | [[13, 15]] | True | **EXACT** |
| **#5 (ID: 4)** | `"ye sab"` | [33, 34] | [34, 35] | [[34, 35]] | `"srif 3 din hai ye sab rok do"` | [29, 36] | [30, 37] | [[30, 37]] | True | **EXACT** |
| **#6 (ID: 5)** | `"jatti"` | [6, 6] | [7, 7] | [[7, 7]] | `"sad it doesn t look nyc"` | [7, 12] | [8, 13] | [[8, 13]] | True | **EXACT** |
| **#7 (ID: 5)** | `"hasdi"` | [15, 15] | [16, 16] | [[16, 16]] | `"wadia lgdi apne gang nal"` | [16, 20] | [17, 21] | [[17, 21]] | True | **EXACT** |
| **#8 (ID: 8)** | `"katora khan"` | [2, 3] | [3, 4] | [[3, 4]] | `"ne kyu li vaccine bhikari"` | [4, 8] | [5, 9] | [[5, 9]] | True | **EXACT** |
| **#9 (ID: 8)** | `"vaccine"` | [7, 7] | [8, 8] | [[8, 8]] | `"ki bhi bhik mang rahe"` | [20, 24] | [21, 25] | [[21, 25]] | True | **EXACT** |
| **#10 (ID: 8)** | `"betiyon"` | [34, 34] | [35, 35] | [[35, 35]] | `"ko bech diya china me"` | [35, 39] | [36, 40] | [[36, 40]] | True | **EXACT** |

- **Verification Result**: `gold_start <= gold_end` is valid for 100% of samples. The positive cell in the $(N \times N)$ grid maps identically to the subword span coordinates.
- **Classification**: **NO ISSUE FOUND**

---

## 2. Audit 2 — Span Class Imbalance

Computed across all 420 sentences in the frozen training set (`data/splits/train.jsonl`):

| Metric | Aspect Spans | Opinion Spans |
|---|---|---|
| **Total Valid Span Candidate Cells** | 160,756 | 160,756 |
| **Positive Grid Cells ($y=1$)** | 733 | 680 |
| **Negative Grid Cells ($y=0$)** | 160,023 | 160,076 |
| **Positive Ratio** | **0.4560%** | **0.4230%** |
| **Negative Ratio** | **99.5440%** | **99.5770%** |
| **Class Imbalance Ratio (Neg : Pos)** | **218.3 : 1** | **235.4 : 1** |

- **Finding**: Over 99.5% of valid cells are negative. Standard unweighted BCE or standard logit thresholds ($\ge 0.0$) naturally push all logits negative unless counterbalanced by strong positive class weighting ($pos\_weight \sim 50\text{--}100$) and calibrated thresholding.
- **Classification**: **LIKELY HYPERPARAMETER ISSUE**

---

## 3. Audit 3 — Span Loss Behavior & Score Distributions

### Historical Training Curve
Inspection of `checkpoints/nssg_dimnet/training_history.json`:
- `train_loss_span` across epochs 1–25:
  `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]`

### Raw Test Set Score Distributions (Existing Checkpoint)

| Statistic | Aspect Spans | Opinion Spans |
|---|---|---|
| Positive Sample Count | 159 | 158 |
| Negative Sample Count | 33,771 | 33,772 |
| **Mean Logit (Positive Cells)** | **0.3607 $\pm$ 1.8985** | **0.0918 $\pm$ 1.8881** |
| **Mean Logit (Negative Cells)** | **0.3846 $\pm$ 1.9430** | **0.4591 $\pm$ 1.9334** |
| **Mean Sigmoid Prob (Positive Cells)** | **0.5568** | **0.5114** |
| **Mean Sigmoid Prob (Negative Cells)** | **0.5578** | **0.5701** |
| **Logit Separation ($\Delta = \text{Pos} - \text{Neg}$)** | **-0.0239** | **-0.3673** |

- **Finding**: The model places virtually identical probabilities ($\sim 0.55$) on positive and negative cells. The negative cells actually have slightly higher logits than the positive cells. This confirms that the biaffine parameters were never shaped by supervision loss.
- **Classification**: **CONFIRMED BUG**

---

## 4. Audit 4 — Decoder Threshold Analysis

Using the existing trained checkpoint, raw biaffine span scores were decoded under varied thresholds without retraining:

| Threshold | Aspect Precision | Aspect Recall | Aspect F1 | Opinion Precision | Opinion Recall | Opinion F1 |
|---|---|---|---|---|---|---|
| **0.01** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **0.05** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **0.10** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **0.20** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **0.30** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **0.50** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **1.00** | 0.0123 | 0.0024 | **0.0040** | 0.0181 | 0.0036 | **0.0060** |
| **2.00** | 0.0123 | 0.0024 | **0.0040** | 0.0120 | 0.0024 | **0.0040** |

- **Finding**: Across all 20 tested examples, max logits ranged from 4.4 to 8.0, and between 68 and 495 spans had logits $> 0.0$. Changing the logit threshold does not improve F1 because the top-5 candidate rankings are based on untrained, random weights.
- **Classification**: **CONFIRMED BUG & HYPERPARAMETER ISSUE**

---

## 5. Audit 5 — Tiny-Data Overfit Test

A diagnostic training test was conducted on 10 training quadruplets from the frozen train split:

### Experiment 5A: Current Training Pipeline (JointVALoss, No Span Targets)
- Epoch 1: Total Loss = 6.2430, Span Loss = **0.0000**
- Epoch 10: Total Loss = 0.8571, Span Loss = **0.0000**
- Epoch 20: Total Loss = 0.3328, Span Loss = **0.0000**
- **Result**: Valence/Arousal regression overfits rapidly ($L \to 0.33$), but **Aspect F1 = 0.0000** and **Opinion F1 = 0.0000** (0 correct out of 10). The span extractor learned nothing.

### Experiment 5B: Supervised Biaffine Span Loss (`BiaffineSpanLoss`)
- Epoch 1: Span Loss = 10.4371
- Epoch 10: Span Loss = 1.7394
- Epoch 20: Span Loss = 0.9039
- Epoch 40: Span Loss = **0.7767**
- **Result**: When supervised with `BiaffineSpanLoss`, the span loss decreases monotonically by over 92% ($10.43 \to 0.77$). This proves that the underlying Biaffine Span Extractor architecture is mathematically sound and capable of learning when given supervision gradients.
- **Classification**: **CONFIRMED BUG in Pipeline Wiring**

---

## 6. Audit 6 — Loss Contribution Analysis

Logged across all training batches using the current setup:

| Loss Component | Unweighted Mean | Loss Weight | Weighted Value | % of Total Loss |
|---|---|---|---|---|
| **$L_{\text{CCC\_V}}$** | 0.966514 | 1.0 | 0.966514 | 46.41% |
| **$L_{\text{CCC\_A}}$** | 0.947319 | 1.0 | 0.947319 | 45.49% |
| **$L_{\text{Smooth\_V}}$** | 0.222440 | 0.5 | 0.111220 | 5.34% |
| **$L_{\text{Smooth\_A}}$** | 0.114837 | 0.5 | 0.057418 | 2.76% |
| **$L_{\text{span}}$** | **0.000000** | 0.3 | **0.000000** | **0.00%** |
| **$L_{\text{total}}$** | **2.082472** | — | **2.082472** | **100.0%** |

- **Finding**: $L_{\text{span}}$ was not just numerically negligible—it was **identically 0.000000%** of the optimization objective.
- **Classification**: **CONFIRMED BUG**

---

## 7. Audit 7 — Mode A vs Mode B Tracing

Full execution trace documented in [`reports/mode_a_mode_b_flow.md`](file:///Users/kmrinal/SwitchVA/reports/mode_a_mode_b_flow.md).
- **Mode A (End-to-End)**: Passes `aspect_spans=None, opinion_spans=None`. Downstream pooling and cross-attention queries are conditioned strictly on top predicted candidate spans.
- **Mode B (Oracle)**: Passes gold aspect/opinion spans to isolate dimensional regression performance.
- **Verification Result**: Zero leakage or cross-contamination between modes.
- **Classification**: **NO ISSUE FOUND**

---

## 8. Audit 8 — Target Leakage Verification

- Verified via automated regression test `test_audit_3_no_gold_va_leakage` in [`tests/test_evaluation_audit.py`](file:///Users/kmrinal/SwitchVA/tests/test_evaluation_audit.py).
- Deliberate mutation of gold valence and arousal targets ($V_1 = 0.20, A_1 = 0.80 \to V_2 = 0.90, A_2 = 0.10$) resulted in $\Delta \hat{V} = 0.000000, \Delta \hat{A} = 0.000000$ for both Baseline and NSSG-DimNet.
- Target VA labels never enter H-NSG, RGAT, cross-attention, $Z$, or regression heads.
- **Classification**: **NO ISSUE FOUND**

---

## 9. Audit 9 — Baseline Consistency Verification

- Baseline uses identical input sentences, frozen splits (420 / 89 / 91), tokenizer, subword aligner, evaluation metrics, and evaluation harness.
- Intended architectural differences: Baseline excludes signed switch distance, switch embeddings, SP-GSA, H-NSG, RGAT, NRC-VAD, and cross-attention, pooling directly via an MLP.
- **Classification**: **NO ISSUE FOUND**

---

## 10. Audit 10 — Module Gradient Norm Inspection

Gradient norms measured after 1 real training step on a training batch:

| Module | Gradient Norm | Status |
|---|---|---|
| `backbone` | 3.758688 | ACTIVE |
| `switch_embedding` | 0.158410 | ACTIVE |
| `sp_gsa` | 0.377044 | ACTIVE |
| **`biaffine_aspect_start_mlp`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| **`biaffine_aspect_end_mlp`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| **`biaffine_aspect_scorer`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| **`biaffine_opinion_start_mlp`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| **`biaffine_opinion_end_mlp`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| **`biaffine_opinion_scorer`** | **0.000000** | **ZERO GRADIENT (DEAD)** |
| `biaffine_span_repr_proj` | 3.128446 | ACTIVE |
| `hnsg_builder` | 0.000000 | ACTIVE (Symbolic, no parameters) |
| `rgat` | 0.145240 | ACTIVE |
| `cross_attention` | 0.219608 | ACTIVE |
| `fusion` | 2.285999 | ACTIVE |
| `valence_head` | 3.256818 | ACTIVE |
| `arousal_head` | 2.819232 | ACTIVE |

- **Finding**: Exactly the 6 parameter modules responsible for computing aspect and opinion span scores received **zero gradient**.
- **Classification**: **CONFIRMED BUG**

---

## 11. Audit 11 — Prediction Examples (20 Test Samples)

Selected from `reports/predictions_nssg_dimnet.jsonl` (Mode A: End-to-End Predicted Spans):

```
Sample 1 (Sentence ID: 7):
  Sentence: "sanvee arre haan yaar agar ek baar hindu saare marne pe utre na to saale jo ye uchalte h na..."
  Gold Aspect: "hindu" | Pred Aspect: ['marne pe utre na to saale jo ye uchalte h na chup chap', 'marne pe utre na to saale']
  Gold Opinion: "saare marne pe utre" | Pred Opinion: ['jo ye uchalte', 'arre haan yaar agar ek baar hindu saare marne pe utre na']
  Valence: Gold = 0.2500, Pred = 0.0882 | Arousal: Gold = 0.8500, Pred = 0.7343
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 2 (Sentence ID: 26):
  Sentence: "for me asif can be way better player at 4 agr aqal istimal kre inme se ifti sohaib will still do..."
  Gold Aspect: "asif" | Pred Aspect: ['apne kaha', '6 pr yeah azam asif hain us position']
  Gold Opinion: "can be way better player at 4 agr aqal istimal kre" | Pred Opinion: ['4 agr aqal istimal kre inme', 'better player at 4']
  Valence: Gold = 0.5380, Pred = 0.0759 | Arousal: Gold = 0.4920, Pred = 0.8768
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 3 (Sentence ID: 26):
  Sentence: "for me asif can be way better player at 4 agr aqal istimal kre inme se ifti sohaib will still do..."
  Gold Aspect: "ifti sohaib" | Pred Aspect: ['apne kaha', '6 pr yeah azam asif hain us position']
  Gold Opinion: "will still do better than asif at 6" | Pred Opinion: ['4 agr aqal istimal kre inme', 'better player at 4']
  Valence: Gold = 0.6550, Pred = 0.0759 | Arousal: Gold = 0.4490, Pred = 0.8768
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 4 (Sentence ID: 26):
  Sentence: "for me asif can be way better player at 4 agr aqal istimal kre inme se ifti sohaib will still do..."
  Gold Aspect: "azam asif" | Pred Aspect: ['apne kaha', '6 pr yeah azam asif hain us position']
  Gold Opinion: "hain us position ke bs" | Pred Opinion: ['4 agr aqal istimal kre inme', 'better player at 4']
  Valence: Gold = 0.6170, Pred = 0.0759 | Arousal: Gold = 0.4630, Pred = 0.8768
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 5 (Sentence ID: 26):
  Sentence: "for me asif can be way better player at 4 agr aqal istimal kre inme se ifti sohaib will still do..."
  Gold Aspect: "ye sab" | Pred Aspect: ['apne kaha', '6 pr yeah azam asif hain us position']
  Gold Opinion: "apne kaha team me hone chaye" | Pred Opinion: ['4 agr aqal istimal kre inme', 'better player at 4']
  Valence: Gold = 0.6990, Pred = 0.0759 | Arousal: Gold = 0.4630, Pred = 0.8768
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 6 (Sentence ID: 28):
  Sentence: "sid sidhearts ke pass itna time nhi h hm apne ideal me busy h but hote to itna sa markar insta..."
  Gold Aspect: "sid sidhearts" | Pred Aspect: ['sidhearts ke pass itna time nhi h hm', 'h hm apne ideal']
  Gold Opinion: "itna time nhi h" | Pred Opinion: ['to itna sa markar insta pr rone ke layk', 'but hote to itna sa markar insta pr rone ke layk nhi']
  Valence: Gold = 0.5020, Pred = 0.3520 | Arousal: Gold = 0.3940, Pred = 0.6419
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 7 (Sentence ID: 28):
  Sentence: "sid sidhearts ke pass itna time nhi h hm apne ideal me busy h but hote to itna sa markar insta..."
  Gold Aspect: "ideal" | Pred Aspect: ['sidhearts ke pass itna time nhi h hm', 'h hm apne ideal']
  Gold Opinion: "busy h" | Pred Opinion: ['to itna sa markar insta pr rone ke layk', 'but hote to itna sa markar insta pr rone ke layk nhi']
  Valence: Gold = 0.5260, Pred = 0.3520 | Arousal: Gold = 0.4750, Pred = 0.6419
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 8 (Sentence ID: 28):
  Sentence: "sid sidhearts ke pass itna time nhi h hm apne ideal me busy h but hote to itna sa markar insta..."
  Gold Aspect: "insta" | Pred Aspect: ['sidhearts ke pass itna time nhi h hm', 'h hm apne ideal']
  Gold Opinion: "rone ke layk nhi chodhte" | Pred Opinion: ['to itna sa markar insta pr rone ke layk', 'but hote to itna sa markar insta pr rone ke layk nhi']
  Valence: Gold = 0.1000, Pred = 0.3520 | Arousal: Gold = 0.8500, Pred = 0.6419
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 9 (Sentence ID: 31):
  Sentence: "ye bi sahi hai wohi baat hui apne log achay lagtay hain irdgird not all..."
  Gold Aspect: "apne log" | Pred Aspect: ['hui apne log achay', 'hai']
  Gold Opinion: "achay lagtay hain" | Pred Opinion: ['irdgird not', 'log achay lagtay hain irdgird']
  Valence: Gold = 0.6500, Pred = 0.2427 | Arousal: Gold = 0.4000, Pred = 0.7111
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 10 (Sentence ID: 33):
  Sentence: "jaise he orange shirt wale ne mulle jab kaate jayenge kaha pointed banda apne dost ko lekar..."
  Gold Aspect: "orange shirt wale" | Pred Aspect: ['hai', 'banda apne dost ko lekar waha se nikal pada isko delhi pe bharosa nahi hai']
  Gold Opinion: "mulle jab kaate jayenge kaha" | Pred Opinion: ['[26, 28]', '[22, 28]']
  Valence: Gold = 0.2420, Pred = 0.1993 | Arousal: Gold = 0.6590, Pred = 0.7286
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 11 (Sentence ID: 33):
  Sentence: "jaise he orange shirt wale ne mulle jab kaate jayenge kaha pointed banda apne dost ko lekar..."
  Gold Aspect: "delhi" | Pred Aspect: ['hai', 'banda apne dost ko lekar waha se nikal pada isko delhi pe bharosa nahi hai']
  Gold Opinion: "bharosa nahi hai" | Pred Opinion: ['[26, 28]', '[22, 28]']
  Valence: Gold = 0.2000, Pred = 0.1993 | Arousal: Gold = 0.5500, Pred = 0.7286
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 12 (Sentence ID: 45):
  Sentence: "meer shaiq pura pakistan tumhare bap ka nahi hai pehley pashtun ko unke hisseki jameen de..."
  Gold Aspect: "pura pakistan" | Pred Aspect: ['aapne', 'meer shaiq pura pakistan tumhare bap ka']
  Gold Opinion: "tumhare bap ka nahi hai" | Pred Opinion: ['shaiq pura', 'nahi hai pehley pashtun ko unke hisseki jameen de']
  Valence: Gold = 0.1000, Pred = 0.2194 | Arousal: Gold = 0.8500, Pred = 0.6384
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 13 (Sentence ID: 45):
  Sentence: "meer shaiq pura pakistan tumhare bap ka nahi hai pehley pashtun ko unke hisseki jameen de..."
  Gold Aspect: "pashtun" | Pred Aspect: ['aapne', 'meer shaiq pura pakistan tumhare bap ka']
  Gold Opinion: "unke hisseki jameen de" | Pred Opinion: ['shaiq pura', 'nahi hai pehley pashtun ko unke hisseki jameen de']
  Valence: Gold = 0.3000, Pred = 0.2194 | Arousal: Gold = 0.7000, Pred = 0.6384
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 14 (Sentence ID: 45):
  Sentence: "meer shaiq pura pakistan tumhare bap ka nahi hai pehley pashtun ko unke hisseki jameen de..."
  Gold Aspect: "pakistan" | Pred Aspect: ['aapne', 'meer shaiq pura pakistan tumhare bap ka']
  Gold Opinion: "apne dalde" | Pred Opinion: ['shaiq pura', 'nahi hai pehley pashtun ko unke hisseki jameen de']
  Valence: Gold = 0.0500, Pred = 0.2194 | Arousal: Gold = 0.9000, Pred = 0.6384
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 15 (Sentence ID: 47):
  Sentence: "hun yawrr lakh lannat wese girls boys sy maza b to leti hen apne b liye hungy..."
  Gold Aspect: "girls" | Pred Aspect: ['yawrr lakh lannat wese girls boys sy maza b to leti hen', 'apne b liye hungy']
  Gold Opinion: "lakh lannat wese girls boys sy maza b to leti hen" | Pred Opinion: ['[12, 18]', 'hen']
  Valence: Gold = 0.3520, Pred = 0.0923 | Arousal: Gold = 0.6730, Pred = 0.7888
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 16 (Sentence ID: 58):
  Sentence: "bro ismein apni nakami wali konsi bat hai pakistani backward areas apne dekhe nhi hain..."
  Gold Aspect: "nakami" | Pred Aspect: ['meri itni intelligent classfellows ko sirf larki hone ki waja', 'opportunities avail nhi karne']
  Gold Opinion: "nakami wali konsi bat hai" | Pred Opinion: ['din it', 'din it s your luck ke apko ache log']
  Valence: Gold = 0.3620, Pred = 0.1906 | Arousal: Gold = 0.4800, Pred = 0.6227
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 17 (Sentence ID: 58):
  Sentence: "bro ismein apni nakami wali konsi bat hai pakistani backward areas apne dekhe nhi hain..."
  Gold Aspect: "intelligent classfellows" | Pred Aspect: ['meri itni intelligent classfellows ko sirf larki hone ki waja', 'opportunities avail nhi karne']
  Gold Opinion: "sirf larki hone ki waja se opportunities avail nhi karne din" | Pred Opinion: ['din it', 'din it s your luck ke apko ache log']
  Valence: Gold = 0.4640, Pred = 0.1906 | Arousal: Gold = 0.5810, Pred = 0.6227
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 18 (Sentence ID: 58):
  Sentence: "bro ismein apni nakami wali konsi bat hai pakistani backward areas apne dekhe nhi hain..."
  Gold Aspect: "ache log" | Pred Aspect: ['meri itni intelligent classfellows ko sirf larki hone ki waja', 'opportunities avail nhi karne']
  Gold Opinion: "it s your luck ke apko ache log mil gaye" | Pred Opinion: ['din it', 'din it s your luck ke apko ache log']
  Valence: Gold = 0.6350, Pred = 0.1906 | Arousal: Gold = 0.4890, Pred = 0.6227
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 19 (Sentence ID: 68):
  Sentence: "suno gor se soniyo walon buri nazar na hmpe dalo chahe jitna zor laglo sabse aage honge ipznians..."
  Gold Aspect: "ipznians" | Pred Aspect: ['laglo sabse aage honge ipznians hamne kha hai tumbhi manolo aao hum', 'kha hai']
  Gold Opinion: "sabse aage honge" | Pred Opinion: ['zor laglo sabse aage honge ipznians hamne kha hai', 'gor se soniyo walon buri nazar na hmpe']
  Valence: Gold = 0.8200, Pred = 0.6784 | Arousal: Gold = 0.7500, Pred = 0.5053
  Origin: PREDICTED SPAN (Mode A End-to-End)

Sample 20 (Sentence ID: 68):
  Sentence: "suno gor se soniyo walon buri nazar na hmpe dalo chahe jitna zor laglo sabse aage honge ipznians..."
  Gold Aspect: "show" | Pred Aspect: ['laglo sabse aage honge ipznians hamne kha hai tumbhi manolo aao hum', 'kha hai']
  Gold Opinion: "waapis laaye" | Pred Opinion: ['zor laglo sabse aage honge ipznians hamne kha hai', 'gor se soniyo walon buri nazar na hmpe']
  Valence: Gold = 0.7500, Pred = 0.6784 | Arousal: Gold = 0.7000, Pred = 0.5053
  Origin: PREDICTED SPAN (Mode A End-to-End)
```

---

## 12. Root Cause Summary & Classification

| Item | Empirical Evidence | Formal Classification |
|---|---|---|
| **Omission of Span Supervision during Training** | `span_targets` was `None` in batches; `train_loss_span = 0.0000` at all epochs; $L_{\text{span}}$ made up 0.000000% of loss. | **CONFIRMED BUG** |
| **Dead Gradient Flow to Biaffine Scorer Modules** | Parameter gradient norms for all 6 biaffine projection/scorer MLPs were identically `0.000000` after training steps. | **CONFIRMED BUG** |
| **Loss Function Signature Mismatch** | `JointVALoss` in Phase 6 expected 4D CE logits `(N, L, L, num_labels)` while `BiaffineSpanLoss` (Phase 3 2D BCE) was omitted from `trainer.py`. | **CONFIRMED BUG** |
| **Extreme Class Imbalance in Grid** | Over 99.5% of valid candidate span cells are negative (220:1 ratio); uncalibrated BCE suppresses logits. | **LIKELY HYPERPARAMETER ISSUE** |
| **Gold Span Grid Alignment** | Verified on 10 real training examples; 100% exact subword coordinate match; $s \le e$ valid. | **NO ISSUE FOUND** |
| **Ground-Truth Target Leakage** | Perturbation of gold VA labels produced bitwise identical predictions ($\Delta = 0.000000$). | **NO ISSUE FOUND** |
| **Baseline Fairness & Consistency** | Baseline uses identical data splits, tokenizer, decoding conventions, and metric calculation. | **NO ISSUE FOUND** |
| **Mode A vs Mode B Separation** | Explicitly separated with zero cross-contamination. | **NO ISSUE FOUND** |

---

## 13. Recommended Corrections (For Next Phase)

*In accordance with the critical constraint of Phase 7A, no modifications have been implemented.* The recommended technical fixes for the subsequent phase are:

1. **Batch Collation of Span Targets**:
   In `dimabsa_tensor_collate_fn` (`src/training/data_utils.py`), construct and batch `gold_aspect_spans` and `gold_opinion_spans` in the dictionary format expected by `BiaffineSpanLoss`.
2. **Integration of `BiaffineSpanLoss` into Trainer**:
   In `src/training/trainer.py`, connect the existing `BiaffineSpanLoss` module (with positive class weighting $pos\_weight \sim 50\text{--}100$) into `train_epoch()` so that the biaffine start/end MLPs and scorers receive active backpropagation gradients.
3. **Adaptive / Top-$k$ Span Decoding**:
   In `BiaffineSpanExtractor.decode_spans` (`src/models/biaffine_span.py`), sort valid candidate spans by logit descending and select the top-$k$ candidates without applying a hard `threshold = 0.0` logit filter.
