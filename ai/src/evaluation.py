"""
Evaluation metrics for single-task and masked multi-task predictions: RMSE, R2,
Spearman, Kendall, and interval coverage/width for regression; ROC-AUC, PR-AUC,
and ECE for classification.
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def _ranking(y_true, y_score) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _rank_correlation(fn, y_true, y_score) -> float:
    if y_true.size < 2 or np.ptp(y_true) == 0 or np.ptp(y_score) == 0:
        return float("nan")
    return float(fn(y_true, y_score).statistic)


def evaluation_metrics(y_true, y_score, task_type, mask=None):
    """Compute evaluation metrics for regression and classification tasks."""

    if task_type not in {"regression", "classification", "multitask"}:
        raise ValueError(
            "task_type must be 'regression', 'classification', or 'multitask'"
        )

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    expected_ndim = 2 if task_type == "multitask" else 1

    if y_true.ndim != expected_ndim:
        raise ValueError(f"{task_type} requires {expected_ndim}D arrays")

    if mask is None:
        mask = np.ones(y_true.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true")

    if task_type == "multitask":
        per_task = [
            _ranking(y_true[mask[:, j], j], y_score[mask[:, j], j])
            for j in range(y_true.shape[1])
        ]
        roc = [t["roc_auc"] for t in per_task]
        valid_scores = [score for score in roc if not np.isnan(score)]

        return {
            "per_task_roc_auc": roc,
            "per_task_pr_auc": [t["pr_auc"] for t in per_task],
            "mean_roc_auc": (
                float(np.mean(valid_scores)) if valid_scores else float("nan")
            ),
        }

    y_true = y_true[mask]
    y_score = y_score[mask]

    if task_type == "regression":
        y_true = y_true.astype(float)

        return {
            "rmse": (
                float(np.sqrt(mean_squared_error(y_true, y_score)))
                if y_true.size > 0
                else float("nan")
            ),
            "r2": (
                float(r2_score(y_true, y_score)) if y_true.size >= 2 else float("nan")
            ),
            "spearman": _rank_correlation(spearmanr, y_true, y_score),
            "kendall": _rank_correlation(kendalltau, y_true, y_score),
        }

    return _ranking(y_true, y_score)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return float("nan")
    bins = np.digitize(probs, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    return float(
        sum(
            (bins == b).mean() * abs(probs[bins == b].mean() - labels[bins == b].mean())
            for b in np.unique(bins)
        )
    )


def interval_metrics(
    y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if len(y_true) == 0:
        return {"coverage": float("nan"), "width": float("nan")}
    return {
        "coverage": float(((y_true >= lo) & (y_true <= hi)).mean()),
        "width": float((hi - lo).mean()),
    }
