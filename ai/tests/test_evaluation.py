import numpy as np
import pytest

import evaluation as ev


def test_regression_perfect_predictions():
    y_true = [5.0, 6.0, 7.0]
    y_pred = [5.0, 6.0, 7.0]

    result = ev.evaluation_metrics(
        y_true,
        y_pred,
        task_type="regression",
    )

    assert result["rmse"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_regression_known_error():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]

    result = ev.evaluation_metrics(
        y_true,
        y_pred,
        task_type="regression",
    )

    expected_rmse = np.sqrt(1 / 3)

    assert result["rmse"] == pytest.approx(expected_rmse)
    assert result["r2"] < 1.0


def test_regression_rejects_different_shapes():
    with pytest.raises(ValueError):
        ev.evaluation_metrics(
            [1.0, 2.0],
            [1.0],
            task_type="regression",
        )


def test_classification_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]

    result = ev.evaluation_metrics(
        y_true,
        y_score,
        task_type="classification",
    )

    assert result["roc_auc"] == pytest.approx(1.0)


def test_classification_known_result():
    y_true = [0, 1, 0, 1]
    y_score = [0.1, 0.8, 0.4, 0.7]

    result = ev.evaluation_metrics(
        y_true,
        y_score,
        task_type="classification",
    )

    assert 0.0 <= result["roc_auc"] <= 1.0


def test_classification_rejects_different_shapes():
    with pytest.raises(ValueError):
        ev.evaluation_metrics(
            [0, 1],
            [0.2],
            task_type="classification",
        )


def test_multitask_with_mask():
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

    result = ev.evaluation_metrics(
        y_true,
        y_score,
        task_type="multitask",
        mask=mask,
    )

    assert result["per_task_roc_auc"][0] == pytest.approx(1.0)
    assert result["per_task_roc_auc"][1] == pytest.approx(1.0)
    assert result["mean_roc_auc"] == pytest.approx(1.0)


def test_multitask_skips_single_class_task():
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

    result = ev.evaluation_metrics(
        y_true,
        y_score,
        task_type="multitask",
    )

    assert result["per_task_roc_auc"][0] == pytest.approx(1.0)
    assert np.isnan(result["per_task_roc_auc"][1])
    assert result["mean_roc_auc"] == pytest.approx(1.0)


def test_multitask_rejects_wrong_mask_shape():
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
        ev.evaluation_metrics(
            y_true,
            y_score,
            task_type="multitask",
            mask=mask,
        )


def test_rejects_invalid_task_type():
    with pytest.raises(ValueError):
        ev.evaluation_metrics(
            [0, 1],
            [0.2, 0.8],
            task_type="invalid",
        )
