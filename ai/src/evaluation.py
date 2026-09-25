import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


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
        raise ValueError(
            f"{task_type} requires {expected_ndim}D arrays"
        )

    if mask is None:
        mask = np.ones(y_true.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != y_true.shape:
            raise ValueError("mask must have the same shape as y_true")

    if task_type == "multitask":
        task_scores = []

        for task_index in range(y_true.shape[1]):
            valid = mask[:, task_index]

            task_true = y_true[valid, task_index]
            task_pred = y_score[valid, task_index]

            if np.unique(task_true).size < 2:
                task_scores.append(float("nan"))
            else:
                task_scores.append(
                    float(roc_auc_score(task_true, task_pred))
                )

        valid_scores = [
            score for score in task_scores
            if not np.isnan(score)
        ]

        return {
            "per_task_roc_auc": task_scores,
            "mean_roc_auc": (
                float(np.mean(valid_scores))
                if valid_scores
                else float("nan")
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
                float(r2_score(y_true, y_score))
                if y_true.size >= 2
                else float("nan")
            ),
        }

    return {
        "roc_auc": (
            float(roc_auc_score(y_true, y_score))
            if np.unique(y_true).size >= 2
            else float("nan")
        ),
    }