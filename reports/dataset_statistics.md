# DimABSA Dataset Statistics & Research Report

- **Total Sentences**: 600
- **Total Aspect-Opinion Pairs**: 1101
- **Average Pairs / Sentence**: 1.83 (Min: 1, Max: 6)

---

## Continuous Target Distributions (Valence & Arousal)
- **Valence**: Mean = 0.4111, Std = 0.2421, Range = [0.020, 0.950]
- **Arousal**: Mean = 0.6051, Std = 0.1366, Range = [0.200, 0.950]

---

## Code-Switching & Language Statistics
- **Total Tokens**: 18176
- **Hindi (HI) Tokens**: 13213 (72.7%)
- **English (EN) Tokens**: 4963 (27.3%)
- **Sentences with at least 1 Code-Switch**: 600 (100.0%)
- **Sentences without Code-Switch**: 0 (0.0%)
- **Average Code-Switches per Sentence**: 8.73

---

## Span Alignment Results
- **Exact Matches (Safe)**: 2125 (96.50%)
- **Normalized Matches (Safe)**: 5 (0.23%)
- **Approximate Matches (Unsafe candidate)**: 43 (1.95%)
- **Unmatched**: 29 (1.32%)
- **Total Spans Checked**: 2202

---

## Dataset Splits (70 / 15 / 15)
- **Train Set**: 420 sentences (762 pairs)
- **Validation Set**: 89 sentences (173 pairs)
- **Test Set**: 91 sentences (166 pairs)
- **Zero Leakage**: Duplicate sentence texts grouped strictly into matching splits.
