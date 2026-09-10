# NSSG-DimNet: Neuro-Symbolic Switch-Gated Dual-Graph Network for Hinglish DimABSA

Official research implementation for **Dimensional Aspect-Based Sentiment Analysis (DimABSA)** in code-switched Hinglish.

---

## 🔬 Overview

NSSG-DimNet models code-switched Hindi-English social text by tightly integrating code-switching dynamics (signed switch-distance & language identification), dynamic gating (SP-GSA), dual-branch structural modeling (Biaffine span extraction + Heterogeneous Neuro-Symbolic Graph with NRC-VAD affective priors), and mutual cross-attention to predict continuous aspect-level **Valence** ($V \in [0, 1]$) and **Arousal** ($A \in [0, 1]$).

For the detailed architectural specification, mathematical formulas, and component flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 📁 Repository Structure

```
SwitchVA/
├── DimABSA_Final_Dataset_600.csv  # Official working dataset (preserved ground truth)
├── ARCHITECTURE.md                 # Detailed architecture specification & forward flow
├── README.md                       # Repository guide and instructions
├── requirements.txt                # Python dependencies
├── configs/
│   ├── nssg_dimnet.yaml           # Full approved NSSG-DimNet configuration
│   ├── baseline.yaml              # Comparative transformer baseline configuration
│   └── ablations.yaml             # Ablation study configurations
├── data/
│   ├── raw/                       # Raw data references
│   ├── processed/                 # Preprocessed dataset cache (later phase)
│   └── splits/                    # Train/Val/Test split definitions (later phase)
├── src/
│   ├── data/                      # Dataset classes and collator interfaces
│   ├── models/                    # Modular PyTorch architectural blocks
│   │   ├── backbone.py            # Multilingual Transformer Backbone (HingRoBERTa/mDeBERTa)
│   │   ├── switch_embedding.py    # Signed switch-distance & language ID embeddings
│   │   ├── sp_gsa.py              # Switch-Point Gated Self-Attention
│   │   ├── biaffine_span.py       # Branch 1: Biaffine Aspect-Opinion Span Extractor
│   │   ├── hnsg.py                # Branch 2: Heterogeneous Neuro-Symbolic Graph Builder
│   │   ├── rgat.py                # Branch 2: Relational Graph Attention Network
│   │   ├── cross_attention.py     # Aspect-Guided Mutual Cross-Attention (Fused Z in R^(3d))
│   │   ├── regression.py          # Independent Valence & Arousal Regression Heads
│   │   └── nssg_dimnet.py         # End-to-end NSSG-DimNet model
│   ├── losses/                    # CCC loss & Joint CCC + Smooth L1 loss
│   ├── training/                  # Training coordinator & harness
│   ├── evaluation/                # CCC, Pearson r, RMSE, and MAE evaluation metrics
│   └── utils/                     # Reproducibility, seed, device, logging, and checkpointing
├── scripts/                       # CLI execution scripts
├── tests/                         # Pytest unit tests for all modules
├── notebooks/                     # Exploratory research notebooks
├── checkpoints/                   # Model weight checkpoints
├── reports/                       # Experiment logs and metric reports
└── streamlit_app/                 # Interactive inference demo application (later phase)
```

---

## 🚀 Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running Unit Tests
```bash
pytest -v tests/
```

### Configuration
Configurations are managed via YAML files in `configs/`:
- `configs/nssg_dimnet.yaml`: Main model and training hyperparameters
- `configs/baseline.yaml`: Vanilla transformer baseline settings
- `configs/ablations.yaml`: 6 structured ablation setups
