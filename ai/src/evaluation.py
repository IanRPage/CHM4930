import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true, y_pred):
    """Compute evaluation metrics for a regression task."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": float(rmse),
        "r2": float(r2),
    }


def classification_metrics(y_true, y_score):
    """Compute ROC-AUC for a binary classification task."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    roc_auc = roc_auc_score(y_true, y_score)

    return {
        "roc_auc": float(roc_auc),
    }


def multitask_roc_auc(y_true, y_score, mask=None):
    """Compute ROC-AUC for multiple binary classification tasks."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if y_true.ndim != 2:
        raise ValueError("y_true and y_score must be 2D arrays")

    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true")

    task_scores = []

    for task_index in range(y_true.shape[1]):
        valid = mask[:, task_index]

        task_true = y_true[valid, task_index]
        task_pred = y_score[valid, task_index]

        if len(task_true) == 0 or len(np.unique(task_true)) < 2:
            task_scores.append(np.nan)
            continue

        score = roc_auc_score(task_true, task_pred)
        task_scores.append(float(score))

    valid_scores = [score for score in task_scores if not np.isnan(score)]

    mean_score = float(np.mean(valid_scores)) if valid_scores else np.nan

    return {
        "per_task_roc_auc": task_scores,
        "mean_roc_auc": mean_score,
    }
