# NSSG-DimNet Official Experimental Results

Evaluation results separated into **Mode A (End-to-End Predicted Spans)** and **Mode B (Gold-Span Oracle)**.

## 1. Mode A: End-to-End Predicted Spans (Real-World Inference)

In this mode, aspect and opinion spans are decoded directly by the biaffine module and used for downstream VA regression.

| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ | Aspect F1 | Opinion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Switch-Unaware Baseline** | -0.0518 | 0.2285 | -0.0461 | 0.0560 | 0.1293 | 0.0473 | 0.0021 | 0.0064 | 0.0021 |
| **NSSG-DimNet (Proposed)** | **0.0116** | **0.1919** | **0.0050** | **0.0714** | **0.1242** | **0.0487** | **0.0415** | **0.0467** | **0.0085** |
| *Delta ($\Delta$)* | *+0.0634* | *-0.0366* | *+0.0512* | *+0.0155* | *-0.0050* | *+0.0014* | *+0.0394* | *+0.0403* | *+0.0064* |

### Span Extraction Detail (Mode A)

| Entity | Model | Precision | Recall | F1 | Correct Spans | Total Predicted | Total Gold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aspect** | Baseline | 0.0038 | 0.0191 | 0.0064 | 3 | 785 | 157 |
| **Aspect** | NSSG-DimNet | 0.0280 | 0.1401 | 0.0467 | 22 | 785 | 157 |
| **Opinion** | Baseline | 0.0013 | 0.0064 | 0.0021 | 1 | 785 | 157 |
| **Opinion** | NSSG-DimNet | 0.0051 | 0.0255 | 0.0085 | 4 | 785 | 157 |

## 2. Mode B: Gold-Span Oracle (Dimensional Regression in Isolation)

In this mode, gold aspect and opinion boundaries are supplied directly to evaluate the upper-bound dimensional regression performance.

| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Switch-Unaware Baseline** | 0.0115 | 0.2146 | 0.0099 | 0.1052 | 0.1274 | 0.0933 | 0.0584 |
| **NSSG-DimNet (Proposed)** | **0.0752** | **0.1898** | **0.0333** | **-0.0006** | **0.1236** | **-0.0005** | **0.0373** |
| *Delta ($\Delta$)* | *+0.0637* | *-0.0248* | *+0.0233* | *-0.1059* | *-0.0038* | *-0.0937* | *-0.0211* |

## 3. Code-Switch Density Stratification Analysis (Mode B Oracle)

| Density Tier | Threshold | Count | Baseline $r_V$ | NSSG $r_V$ | $\Delta r_V$ | Baseline $r_A$ | NSSG $r_A$ | $\Delta r_A$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Low | `< 0.15` | 21 | 0.0719 | 0.1086 | +0.0368 | -0.0767 | 0.1498 | +0.2265 |
| Medium | `0.15 - 0.30` | 72 | 0.0994 | 0.1208 | +0.0213 | 0.2518 | -0.1836 | -0.4353 |
| High | `>= 0.30` | 64 | -0.0859 | 0.0298 | +0.1157 | -0.0111 | 0.1746 | +0.1857 |
