import numpy as np
import pytest
import torch

from evaluation import evaluation_metrics
from mufu.calibration import (
    CalibrationParams,
    evaluate_calibration,
    fit_calibration,
)
from mufu.contract import ACTIVE_COLUMNS, N_BIO, N_TOX, TOX_COLUMNS
from mufu.loss.task_loss import dataset_pos_weights
from mufu.train import predict, predict_logits


def test_fit_calibration_rejects_unknown_mode(model, labeled_dataset):
    logits = predict_logits(model, labeled_dataset)
    with pytest.raises(ValueError, match="unknown calibration"):
        fit_calibration("bad_mode", logits, labeled_dataset)


def test_log_weight_calibration_uses_transformed_pos_weights(model, labeled_dataset):
    logits = predict_logits(model, labeled_dataset)
    pw_cap5 = dataset_pos_weights(labeled_dataset, mode="cap5")
    cal = fit_calibration("log_weight", logits, labeled_dataset, pos_weights=pw_cap5)

    assert torch.allclose(cal.active_scale, torch.ones(N_BIO))
    assert torch.allclose(cal.tox_scale, torch.ones(N_TOX))
    assert torch.allclose(cal.active_shift, -torch.log(pw_cap5["bio_cls"]))
    assert torch.allclose(cal.tox_shift, -torch.log(pw_cap5["tox"]))

    pw_none = dataset_pos_weights(labeled_dataset, mode="none")
    cal_none = fit_calibration(
        "log_weight", logits, labeled_dataset, pos_weights=pw_none
    )
    assert torch.allclose(cal_none.active_shift, torch.zeros(N_BIO))
    assert torch.allclose(cal_none.tox_shift, torch.zeros(N_TOX))


@pytest.mark.parametrize("mode", ["none", "log_weight", "temperature", "platt"])
def test_calibration_preserves_roc_auc_and_roundtrips_state_dict(
    mode, model, labeled_dataset
):
    logits = predict_logits(model, labeled_dataset)
    pw = dataset_pos_weights(labeled_dataset, mode="raw")
    cal = fit_calibration(mode, logits, labeled_dataset, pos_weights=pw)
    restored = CalibrationParams.from_state_dict(cal.state_dict())

    raw_preds = predict(model, labeled_dataset)
    cal_preds = predict(model, labeled_dataset, calibration=restored)

    for j, col in enumerate(ACTIVE_COLUMNS):
        mask = labeled_dataset.masks["mask_bio"][:, j].numpy()
        y = labeled_dataset.labels["y_active"][mask, j].numpy()
        raw_auc = evaluation_metrics(
            y, raw_preds[col][mask], task_type="classification"
        )["roc_auc"]
        cal_auc = evaluation_metrics(
            y, cal_preds[col][mask], task_type="classification"
        )["roc_auc"]
        assert np.isclose(raw_auc, cal_auc)


def test_evaluate_calibration_computes_metrics_and_summaries(model, labeled_dataset):
    preds = predict(model, labeled_dataset)
    table = evaluate_calibration(preds, labeled_dataset)

    expected_endpoints = {
        *ACTIVE_COLUMNS,
        *TOX_COLUMNS,
        "macro_bio_cls",
        "macro_tox21",
        "macro_clintox",
        "macro_tox",
    }
    assert set(table["endpoint"]) == expected_endpoints
    for col in ("prevalence", "mean_prob", "bias", "abs_bias", "brier", "bce", "ece"):
        assert np.isfinite(table[col]).all()
