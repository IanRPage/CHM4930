import json

import numpy as np
import pandas as pd
import pytest

from baseline import benchmark as bm
from baseline import stacking
from baseline.results import file_sha256


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["C", "CC", "CCC", "N", "CN", "O", "CO", "CCO", "CNO", "CNC"],
            "split": ["val"] * 5 + ["test"] * 5,
            "pIC50_BACE1": [5, 7, 8, 6, np.nan] * 2,
            "active_BACE1": [0, 1, 1, 1, np.nan] * 2,
            "pIC50_EGFR": [4, 6, 8, 7, 5] * 2,
            "active_EGFR": [0, 1, 1, 1, 0] * 2,
            "NR-AR": [0, 1, 0, 1, np.nan] * 2,
            "NR-AR-LBD": [0, 1, 0, 1, np.nan] * 2,
            "CT_TOX": [0, 1, 0, 1, np.nan] * 2,
            "FDA_APPROVED": [0, 0, 0, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    for endpoint in bm.TOX_TASKS:
        if endpoint not in df:
            df[endpoint] = np.nan
    data_path = tmp_path / "snapshot.csv"
    df.to_csv(data_path, index=False)
    monkeypatch.setattr(bm, "load_benchmark_split", lambda *_: df)

    run_dirs = {
        model: bm.RUNS_DIR / f"{model}_default" for model in stacking.RUN_ENDPOINTS
    }
    promoted = {}
    for model, run_dir in run_dirs.items():
        promoted[model] = {"default": run_dir.name}
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "model": model,
                    "split_id": "default",
                    "data_sha256": file_sha256(data_path),
                    "split_sha256": bm.split_sha256(df),
                    "status": "complete",
                    "seeds": list(bm.SEEDS),
                    "completed_seeds": list(bm.SEEDS),
                }
            )
        )
    bm.PROMOTED_PATH.write_text(json.dumps(promoted))

    labels = np.array([5, 7, 8, 6, np.nan])
    correct = [0.1, 0.9, 0.2, 0.8, np.nan]
    reversed_rank = [0.99, 0.01, 0.99, 0.01, 0.2]
    for seed in bm.SEEDS:
        for split in ("val", "test"):
            weight = seed / 4
            mu_reg = labels + (2 - 2 * weight if split == "val" else -1)
            rf_reg = labels + (-2 * weight if split == "val" else 1)
            if split == "val":
                mu_reg[3] = np.nan
            elif seed == 0:
                rf_reg[3] = np.nan
            mufu = {
                "pIC50_BACE1": mu_reg,
                "pIC50_EGFR": [4, 6, 8, 7, 5],
                "NR-AR": [0.5, 0.501, 0.5, 0.501, np.nan],
                "NR-AR_uncal": reversed_rank,
                "NR-AR-LBD": correct,
                "NR-AR-LBD_uncal": correct,
                "CT_TOX": correct,
                "CT_TOX_uncal": correct,
                "FDA_APPROVED": correct,
                "FDA_APPROVED_uncal": correct,
            }
            rf_tox = {
                "NR-AR": reversed_rank,
                "NR-AR-LBD": correct
                if split == "val"
                else [0.1, np.nan, 0.2, np.nan, np.nan],
                "CT_TOX": correct,
                "FDA_APPROVED": [0.1, 0.9, 0.2, 0.8, 0.6],
            }
            for model, scores in (
                ("mufu", mufu),
                ("rf_bioactivity", {"pIC50_BACE1": rf_reg}),
                ("rf_toxicity", rf_tox),
            ):
                # Reverse one model's rows with their predictions to exercise SMILES joins.
                frame = df if model != "rf_bioactivity" else df.iloc[::-1]
                if model == "rf_bioactivity":
                    scores = {
                        endpoint: values[::-1] for endpoint, values in scores.items()
                    }
                bm.save_split_predictions(
                    frame,
                    split,
                    scores,
                    run_dirs[model] / "predictions" / f"{split}_seed_{seed}.csv",
                )
    return data_path, run_dirs


def test_validation_weights_and_paired_test_summary(artifacts):
    data_path, _ = artifacts
    summary, weights = stacking.evaluate_stacking(data_path=data_path)
    fits = weights.set_index(["endpoint", "seed"])
    bace = fits.loc["pIC50_BACE1"]
    np.testing.assert_allclose(bace["mufu_weight"], [0, 0.25, 0.5, 0.75, 1])
    assert bace["val_labeled"].tolist() == [4] * 5
    assert bace["val_paired"].tolist() == [3] * 5
    assert bace["test_paired"].tolist() == [3, 4, 4, 4, 4]
    assert set(fits.loc["NR-AR", "mufu_weight"]) == {1.0}
    assert set(fits.loc["CT_TOX", "mufu_weight"]) == {0.5}
    assert set(fits.loc["FDA_APPROVED", "status"]) == {"single-class validation labels"}
    assert fits.loc["FDA_APPROVED", "mufu_weight"].isna().all()
    assert set(fits.loc["pIC50_EGFR", "status"]) == {"unsupported by RF"}

    scores = summary.set_index(["endpoint", "metric", "model"]).sort_index()
    rf, mu, blend = (
        scores.loc[("pIC50_BACE1", "rmse", name)] for name in ("RF", "MuFu", "Blend")
    )
    assert rf["mean"] == mu["mean"] == 1
    expected = np.array([1, 0.5, 0, 0.5, 1])
    assert blend["mean"] == pytest.approx(expected.mean())
    assert blend["std"] == pytest.approx(expected.std(ddof=1))
    assert blend["n_seeds"] == 5
    # NR-AR-LBD has only negatives after pairing on test, so only NR-AR
    # contributes to the Tox21 macro for all three models.
    assert scores.loc[("macro_tox21", "roc_auc", "Blend"), "mean"] == 1
    assert scores.loc[("macro_tox21", "roc_auc", "RF"), "mean"] == 0
    assert scores.loc[("macro_clintox", "roc_auc", "Blend"), "mean"] == 1
    assert (scores.loc[("FDA_APPROVED", "roc_auc"), "n_seeds"] == 0).all()


@pytest.mark.parametrize(
    ("replacement", "error"),
    [("CCCC", "prediction SMILES do not match"), ("CC", "duplicate prediction SMILES")],
)
def test_rejects_misaligned_smiles(artifacts, replacement, error):
    data_path, run_dirs = artifacts
    path = run_dirs["mufu"] / "predictions" / "val_seed_0.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "smiles"] = replacement
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        stacking.evaluate_stacking(data_path=data_path)


@pytest.mark.parametrize(
    "field", ["split_id", "split_sha256", "data_sha256", "completed_seeds"]
)
def test_rejects_mismatched_run_metadata(artifacts, field):
    data_path, run_dirs = artifacts
    path = run_dirs["rf_bioactivity"] / "run.json"
    meta = json.loads(path.read_text())
    meta[field] = bm.SEEDS[:-1] if field == "completed_seeds" else "wrong"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=field):
        stacking.evaluate_stacking(data_path=data_path)


@pytest.mark.parametrize(
    ("column", "value", "error"),
    [
        ("NR-AR", 1.1, "probabilities must be in"),
        ("NR-AR_uncal", None, "missing calibration evidence"),
    ],
)
def test_rejects_non_probability_or_uncalibrated_inputs(
    artifacts, column, value, error
):
    data_path, run_dirs = artifacts
    path = run_dirs["mufu"] / "predictions" / "val_seed_0.csv"
    frame = pd.read_csv(path)
    if value is None:
        frame = frame.drop(columns=column)
    else:
        frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=error):
        stacking.evaluate_stacking(data_path=data_path)


def test_lists_missing_prediction_artifacts(artifacts):
    data_path, run_dirs = artifacts
    (run_dirs["mufu"] / "predictions" / "test_seed_1.csv").unlink()
    (run_dirs["rf_toxicity"] / "predictions" / "val_seed_4.csv").unlink()
    with pytest.raises(FileNotFoundError) as exc:
        stacking.evaluate_stacking(data_path=data_path)
    assert "test_seed_1.csv" in str(exc.value)
    assert "val_seed_4.csv" in str(exc.value)


def test_rejects_endpoints_that_disagree_between_val_and_test(artifacts):
    data_path, run_dirs = artifacts
    path = run_dirs["rf_bioactivity"] / "predictions" / "test_seed_0.csv"
    pd.read_csv(path).drop(columns="pIC50_BACE1").to_csv(path, index=False)
    with pytest.raises(
        ValueError, match="validation/test endpoint mismatch.*pIC50_BACE1"
    ):
        stacking.evaluate_stacking(data_path=data_path)


def test_fixed_ensemble_and_validation_fit_for_each_rf_seed(artifacts):
    data_path, run_dirs = artifacts
    # Make seed 0 different and change its file order. The five-member
    # mean now predicts BACE1 exactly on test. Validation stays unchanged.
    path = run_dirs["mufu"] / "predictions" / "test_seed_0.csv"
    frame = pd.read_csv(path)
    frame["pIC50_BACE1"] += 5
    frame["NR-AR"] = frame["NR-AR_uncal"]
    frame.iloc[::-1].to_csv(path, index=False)
    summary, fits = stacking.evaluate_stacking(data_path=data_path, ensemble=True)
    scores = summary.set_index(["model", "endpoint", "metric"])
    fixed = scores.loc[("MuFu ensemble", "pIC50_BACE1", "rmse")]
    assert fixed["mean"] == 0
    assert fixed["n_evaluations"] == 1
    assert np.isnan(fixed["std"])
    assert scores.loc[("MuFu ensemble", "macro_tox21", "roc_auc"), "mean"] == 0.5

    expected_weights = np.array([0, 1 / 3, 1 / 2, 3 / 5, 2 / 3])
    np.testing.assert_allclose(
        fits.loc[fits.endpoint == "pIC50_BACE1", "mufu_weight"], expected_weights
    )
    blend = scores.loc[("Ensemble + RF", "pIC50_BACE1", "rmse")]
    errors = 1 - expected_weights
    assert blend["n_evaluations"] == 5
    assert blend["mean"] == pytest.approx(errors.mean())
    assert blend["std"] == pytest.approx(errors.std(ddof=1))


def test_ensemble_rejects_inconsistent_member_endpoints(artifacts):
    data_path, run_dirs = artifacts
    path = run_dirs["mufu"] / "predictions" / "val_seed_4.csv"
    pd.read_csv(path).drop(columns="NR-AR").to_csv(path, index=False)
    with pytest.raises(ValueError, match="MuFu seed 4: ensemble endpoint mismatch"):
        stacking.evaluate_stacking(data_path=data_path, ensemble=True)
