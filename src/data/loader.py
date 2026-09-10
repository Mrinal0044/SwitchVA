"""Dataset loader for official DimABSA CSV dataset."""

import os
import logging
from typing import Dict, List, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "sentence_id",
    "sentence",
    "code_switch",
    "all_aspects",
    "all_opinions",
    "valence_scores",
    "arousal_scores",
]


def load_raw_dataset(csv_path: str = "DimABSA_Final_Dataset_600.csv") -> pd.DataFrame:
    """Load the official DimABSA CSV dataset and perform schema check.

    Args:
        csv_path: Path to DimABSA CSV file.

    Returns:
        pandas DataFrame containing loaded records.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Official dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate presence of expected columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    logger.info("Loaded %d rows from %s with columns: %s", len(df), csv_path, list(df.columns))
    return df
