import numpy as np
import pytest

import evaluation as ev


def test_regression_metrics_perfect_predictions():
    y_true = [5.0, 6.0, 7.0]
    y_pred = [5.0, 6.0, 7.0]

    result = ev.regression_metrics(y_true, y_pred)

    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regression_metrics_known_error():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]

    result = ev.regression_metrics(y_true, y_pred)

    expected_rmse = np.sqrt(1 / 3)

    assert result["rmse"] == pytest.approx(expected_rmse)
    assert result["r2"] < 1.0


def test_regression_metrics_rejects_different_shapes():
    with pytest.raises(ValueError):
        ev.regression_metrics([1.0, 2.0], [1.0])


def test_classification_metrics_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]

    result = ev.classification_metrics(y_true, y_score)

    assert result["roc_auc"] == pytest.approx(1.0)


def test_classification_metrics_known_result():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    result = ev.classification_metrics(y_true, y_score)

    assert 0.0 <= result["roc_auc"] <= 1.0


def test_classification_metrics_rejects_different_shapes():
    with pytest.raises(ValueError):
        ev.classification_metrics([0, 1], [0.2])


def test_multitask_roc_auc_with_mask():
    y_true = [
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],
    ]

    y_score = [
        [0.1, 0.9],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
    ]

    mask = [
        [True, True],
        [True, True],
        [True, True],
        [True, False],
    ]

    result = ev.multitask_roc_auc(y_true, y_score, mask)

    assert result["per_task_roc_auc"][0] == pytest.approx(1.0)
    assert result["per_task_roc_auc"][1] == pytest.approx(1.0)
    assert result["mean_roc_auc"] == pytest.approx(1.0)


def test_multitask_roc_auc_skips_single_class_task():
    y_true = [
        [0, 1],
        [1, 1],
        [0, 1],
    ]

    y_score = [
        [0.1, 0.8],
        [0.9, 0.9],
        [0.2, 0.7],
    ]

    result = ev.multitask_roc_auc(y_true, y_score)

    assert result["per_task_roc_auc"][0] == pytest.approx(1.0)
    assert np.isnan(result["per_task_roc_auc"][1])
    assert result["mean_roc_auc"] == pytest.approx(1.0)


def test_multitask_roc_auc_rejects_wrong_mask_shape():
    y_true = [
        [0, 1],
        [1, 0],
    ]

    y_score = [
        [0.1, 0.9],
        [0.8, 0.2],
    ]

    mask = [
        [True],
        [True],
    ]

    with pytest.raises(ValueError):
        ev.multitask_roc_auc(y_true, y_score, mask)
