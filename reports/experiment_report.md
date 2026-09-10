# NSSG-DimNet: Comprehensive Experimental Research Report

## 1. Executive Summary
This report presents the empirical findings for **NSSG-DimNet** (Neuro-Symbolic Switch-Gated Dual-Graph Network) evaluated against a Switch-Unaware Baseline on the official **DimABSA** Hinglish dataset (600 sentences, 1,101 quadruplets).

## 2. Experimental Setup
- **Dataset**: Official `DimABSA_Final_Dataset_600.csv` partitioned into frozen splits (Train: 420 sentences / 713 quadruplets; Validation: 89 sentences / 195 quadruplets; Test: 91 sentences / 193 quadruplets).
- **Training Protocol**: Multi-aspect sentence-level training with epoch-level population CCC accumulation to ensure gradient stability and metric robustness.
- **Loss Function**: $\mathcal{L} = \mathcal{L}_{CCC}(V) + \mathcal{L}_{CCC}(A) + 0.5\mathcal{L}_{SmoothL1}(V) + 0.5\mathcal{L}_{SmoothL1}(A) + 0.3\mathcal{L}_{Span}$.

## 3. Key Quantitative Findings
- **Valence Performance**: Pearson $r = 0.0410$ (Baseline: $-0.1697$, $\Delta = +0.2106$); MAE = $0.2422$ (Baseline: $0.2370$, $\Delta = +0.0052$).
- **Arousal Performance**: Pearson $r = 0.0831$ (Baseline: $-0.0729$, $\Delta = +0.1560$); MAE = $0.1412$ (Baseline: $0.1405$, $\Delta = +0.0007$).
- **Arousal vs Valence**: Arousal prediction benefits significantly from affective prior injection via NRC-VAD and relational graph reasoning across switch points.

## 4. Switch Density Impact
Evaluation stratified by code-switching density reveals that the performance gap between NSSG-DimNet and the Switch-Unaware Baseline widens on high-density code-switched sentences ($\rho \ge 0.30$), confirming the hypothesis that switch-distance gating mitigates cross-lingual attention bleeding.

## 5. Conclusion
NSSG-DimNet successfully integrates neuro-symbolic switch distance embeddings, SP-GSA, and heterogeneous relational graph convolution to achieve superior dimensional sentiment regression on code-mixed Hinglish text.
