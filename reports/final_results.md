# NSSG-DimNet Official Experimental Results

Evaluation results separated into **Mode A (End-to-End Predicted Spans)** and **Mode B (Gold-Span Oracle)**.

## 1. Mode A: End-to-End Predicted Spans (Real-World Inference)

In this mode, aspect and opinion spans are decoded directly by the biaffine module and used for downstream VA regression.

| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ | Aspect F1 | Opinion F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Switch-Unaware Baseline** | 0.0729 | 0.2141 | 0.0588 | 0.1192 | 0.1241 | 0.0939 | 0.0961 | 0.0042 | 0.0042 |
| **NSSG-DimNet (Proposed)** | **0.0931** | **0.2489** | **0.0879** | **0.0532** | **0.1479** | **0.0480** | **0.0732** | **0.0085** | **0.0064** |
| *Delta ($\Delta$)* | *+0.0202* | *+0.0348* | *+0.0291* | *-0.0659* | *+0.0238* | *-0.0458* | *-0.0229* | *+0.0042* | *+0.0021* |

### Span Extraction Detail (Mode A)

| Entity | Model | Precision | Recall | F1 | Correct Spans | Total Predicted | Total Gold |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Aspect** | Baseline | 0.0025 | 0.0127 | 0.0042 | 2 | 785 | 157 |
| **Aspect** | NSSG-DimNet | 0.0051 | 0.0255 | 0.0085 | 4 | 785 | 157 |
| **Opinion** | Baseline | 0.0025 | 0.0127 | 0.0042 | 2 | 785 | 157 |
| **Opinion** | NSSG-DimNet | 0.0038 | 0.0191 | 0.0064 | 3 | 785 | 157 |

## 2. Mode B: Gold-Span Oracle (Dimensional Regression in Isolation)

In this mode, gold aspect and opinion boundaries are supplied directly to evaluate the upper-bound dimensional regression performance.

| Model | Valence $r$ | Valence MAE | Valence CCC | Arousal $r$ | Arousal MAE | Arousal CCC | Mean $r$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Switch-Unaware Baseline** | -0.0264 | 0.2105 | -0.0229 | -0.0523 | 0.1381 | -0.0450 | -0.0393 |
| **NSSG-DimNet (Proposed)** | **0.0583** | **0.2340** | **0.0568** | **0.0351** | **0.1427** | **0.0332** | **0.0467** |
| *Delta ($\Delta$)* | *+0.0847* | *+0.0236* | *+0.0797* | *+0.0874* | *+0.0046* | *+0.0782* | *+0.0860* |

## 3. Code-Switch Density Stratification Analysis (Mode B Oracle)

| Density Tier | Threshold | Count | Baseline $r_V$ | NSSG $r_V$ | $\Delta r_V$ | Baseline $r_A$ | NSSG $r_A$ | $\Delta r_A$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Low | `< 0.15` | 21 | 0.1056 | -0.0563 | -0.1620 | 0.1646 | -0.1873 | -0.3519 |
| Medium | `0.15 - 0.30` | 72 | 0.1022 | 0.1422 | +0.0400 | -0.0080 | -0.0381 | -0.0301 |
| High | `>= 0.30` | 64 | -0.1973 | -0.0020 | +0.1953 | -0.1843 | 0.1596 | +0.3439 |
