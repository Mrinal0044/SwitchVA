"""Evaluation metrics for dimensional Valence-Arousal regression and span extraction."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import pearsonr


def compute_ccc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Calculate Concordance Correlation Coefficient (CCC).

    Args:
        y_true: 1D array of ground-truth values.
        y_pred: 1D array of predicted values.
        eps: Small epsilon for numerical stability.

    Returns:
        CCC scalar value in [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2:
        return 0.0

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    if var_true < eps and var_pred < eps:
        return 1.0 if abs(mean_true - mean_pred) < 1e-5 else 0.0

    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    numerator = 2.0 * cov
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2 + eps

    return float(np.clip(numerator / denominator, -1.0, 1.0))


def compute_pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Pearson Correlation Coefficient (r)."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    r, _ = pearsonr(y_true, y_pred)
    if np.isnan(r) or np.isinf(r):
        return 0.0
    return float(r)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_predictions(
    y_true_v: Union[np.ndarray, list],
    y_pred_v: Union[np.ndarray, list],
    y_true_a: Union[np.ndarray, list],
    y_pred_a: Union[np.ndarray, list],
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for both Valence and Arousal."""
    y_true_v = np.asarray(y_true_v, dtype=np.float64)
    y_pred_v = np.asarray(y_pred_v, dtype=np.float64)
    y_true_a = np.asarray(y_true_a, dtype=np.float64)
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64)

    ccc_v = compute_ccc(y_true_v, y_pred_v)
    ccc_a = compute_ccc(y_true_a, y_pred_a)
    r_v = compute_pearson_r(y_true_v, y_pred_v)
    r_a = compute_pearson_r(y_true_a, y_pred_a)
    rmse_v = compute_rmse(y_true_v, y_pred_v)
    rmse_a = compute_rmse(y_true_a, y_pred_a)
    mae_v = compute_mae(y_true_v, y_pred_v)
    mae_a = compute_mae(y_true_a, y_pred_a)

    return {
        "ccc_v": ccc_v,
        "ccc_a": ccc_a,
        "ccc_mean": (ccc_v + ccc_a) / 2.0,
        "pearson_v": r_v,
        "pearson_a": r_a,
        "pearson_mean": (r_v + r_a) / 2.0,
        "rmse_v": rmse_v,
        "rmse_a": rmse_a,
        "mae_v": mae_v,
        "mae_a": mae_a,
    }


def compute_span_metrics(
    gold_spans: List[List[Tuple[int, int]]],
    pred_spans: List[List[Tuple[int, int]]],
) -> Dict[str, float]:
    """Compute Exact Match Precision, Recall, and F1 for extracted spans.

    Args:
        gold_spans: List of gold span lists per sentence: [[(s1, e1), ...], ...]
        pred_spans: List of predicted span lists per sentence: [[(s1, e1), ...], ...]

    Returns:
        Dictionary with precision, recall, f1.
    """
    total_gold = 0
    total_pred = 0
    correct = 0

    def _to_span_tuple(item):
        if isinstance(item, dict):
            return (int(item.get("start", -1)), int(item.get("end", -1)))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            return (int(item[0]), int(item[1]))
        return (-1, -1)

    for g_list, p_list in zip(gold_spans, pred_spans):
        g_set = {_to_span_tuple(x) for x in g_list}
        g_set = {p for p in g_set if p[0] >= 0 and p[1] >= 0}
        p_set = {_to_span_tuple(x) for x in p_list}
        p_set = {p for p in p_set if p[0] >= 0 and p[1] >= 0}

        total_gold += len(g_set)
        total_pred += len(p_set)
        correct += len(g_set.intersection(p_set))

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "total_gold": total_gold,
        "total_pred": total_pred,
    }


def evaluate_by_switch_density(
    records: List[Dict[str, Any]],
    low_thresh: float = 0.15,
    high_thresh: float = 0.30,
) -> Dict[str, Dict[str, float]]:
    """Compute dimensional performance partitioned by code-switching density.

    Groups:
    - low: density < low_thresh
    - medium: low_thresh <= density < high_thresh
    - high: density >= high_thresh
    """
    bins: Dict[str, Dict[str, List[float]]] = {
        "low": {"v_true": [], "v_pred": [], "a_true": [], "a_pred": []},
        "medium": {"v_true": [], "v_pred": [], "a_true": [], "a_pred": []},
        "high": {"v_true": [], "v_pred": [], "a_true": [], "a_pred": []},
    }

    for rec in records:
        tokens = rec.get("tokens", [])
        lang_ids = rec.get("language_ids", [])
        n_tok = max(1, len(tokens))

        # Count code-switch transitions
        sw_count = 0
        for i in range(1, len(lang_ids)):
            l_prev = str(lang_ids[i - 1]).upper()
            l_curr = str(lang_ids[i]).upper()
            if l_prev != l_curr and l_prev not in ["PAD", "SPECIAL"] and l_curr not in ["PAD", "SPECIAL"]:
                sw_count += 1

        density = sw_count / n_tok
        b_name = "low" if density < low_thresh else ("medium" if density < high_thresh else "high")

        # Predictions vs gold targets
        preds = rec.get("predictions", [])
        golds = rec.get("annotations", rec.get("quadruplets", []))
        n_p = min(len(preds), len(golds))

        for k in range(n_p):
            bins[b_name]["v_pred"].append(float(preds[k].get("valence", 0.5)))
            bins[b_name]["a_pred"].append(float(preds[k].get("arousal", 0.5)))
            bins[b_name]["v_true"].append(float(golds[k].get("valence", 0.5)))
            bins[b_name]["a_true"].append(float(golds[k].get("arousal", 0.5)))

    results: Dict[str, Dict[str, float]] = {}
    for group_name, data in bins.items():
        if len(data["v_true"]) >= 2:
            metrics = evaluate_predictions(
                data["v_true"], data["v_pred"],
                data["a_true"], data["a_pred"]
            )
            metrics["count"] = len(data["v_true"])
            results[group_name] = metrics
        else:
            results[group_name] = {
                "count": len(data["v_true"]),
                "pearson_v": 0.0, "pearson_a": 0.0,
                "mae_v": 0.0, "mae_a": 0.0,
                "ccc_v": 0.0, "ccc_a": 0.0,
            }

    return results


def compute_model_deltas(
    baseline_metrics: Dict[str, float],
    nssg_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Compute Δ = NSSG-DimNet - Baseline."""
    deltas = {}
    for k in baseline_metrics:
        if k in nssg_metrics and isinstance(baseline_metrics[k], (int, float)):
            deltas[f"delta_{k}"] = nssg_metrics[k] - baseline_metrics[k]
    return deltas
