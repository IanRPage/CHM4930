import json

import numpy as np
import pandas as pd
import pytest

from baseline import baseline_toxicity as bt


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bt, "record_results", lambda rows, model: calls.append((model, rows))
    )
    return calls


@pytest.fixture
def fake_forests(monkeypatch):
    models = []

    class Forest:
        # Deliberately reverse the class order to catch a hard-coded [:, 1].
        classes_ = np.array([1, 0])

        def __init__(self, **kwargs):
            self.config = kwargs
            self.max_features = kwargs["max_features"]
            self.predicted = []
            models.append(self)

        def fit(self, X, y):
            self.X = X.copy()
            self.y = y.copy()
            return self

        def predict_proba(self, X):
            self.predicted.extend(X[:, 0].tolist())
            # Validation (6, 7) favors log2; test (8, 9) favors sqrt.
            scores = {
                "sqrt": {6: 0.9, 7: 0.1, 8: 0.1, 9: 0.9},
                "log2": {6: 0.1, 7: 0.9, 8: 0.9, 9: 0.1},
            }.get(self.max_features, {})
            positive = np.array([scores.get(i, 0.5) for i in X[:, 0]])
            return np.column_stack([positive, 1 - positive])

        def predict(self, X):
            raise AssertionError("ROC-AUC must use probabilities, not hard labels")

    monkeypatch.setattr(bt, "RandomForestClassifier", Forest)
    return models


@pytest.fixture
def combined(monkeypatch):
    tasks = ["T1", "T2"]
    monkeypatch.setattr(bt, "TOX_TASKS", tasks)
    monkeypatch.setattr(
        bt, "DATASETS", {"tox21": {"tasks": ["T1"]}, "clintox": {"tasks": ["T2"]}}
    )
    return pd.DataFrame(
        {
            "smiles": ["C" * i for i in range(1, 10)],
            "T1": [np.nan, 0, 1, np.nan, np.nan, 0, 1, 0, 1],
            "T2": [np.nan, np.nan, 1, 0, np.nan, np.nan, np.nan, 1, 1],
            "in_tox21": True,
            "in_clintox": True,
        },
        index=range(20, 29),
    )


def test_main_keeps_shared_split_masks_and_fingerprints_aligned(
    monkeypatch, capsys, combined, fake_forests, recorded
):
    original = combined.copy(deep=True)
    monkeypatch.setattr(bt, "load_combined", lambda: combined)

    def split_full_table(df):
        pd.testing.assert_frame_equal(df, original)
        return df.assign(split=["train"] * 4 + ["val"] * 3 + ["test"] * 2)

    monkeypatch.setattr(bt, "scaffold_split", split_full_table)
    featurized = []

    def fingerprint(mol, n_bits):
        assert n_bits == 2048
        featurized.append(mol.GetNumAtoms())
        return np.array([mol.GetNumAtoms()])

    monkeypatch.setattr(bt, "mol_to_ecfp4", fingerprint)
    bt.main()

    # Unlabeled rows are removed only after splitting; each remaining row is
    # featurized once, even if it has labels for multiple tasks.
    assert featurized == [2, 3, 4, 6, 7, 8, 9]
    assert len(fake_forests) == 5  # four candidates for T1, sqrt fallback for T2
    for model in fake_forests[:4]:
        np.testing.assert_array_equal(model.X[:, 0], [2, 3])
        np.testing.assert_array_equal(model.y, [0, 1])
        assert model.config == {
            "n_estimators": 500,
            "max_features": model.max_features,
            "n_jobs": -1,
            "random_state": 42,
        }
        if model.max_features != "log2":
            assert set(model.predicted) == {6, 7}
    np.testing.assert_array_equal(fake_forests[-1].X[:, 0], [3, 4])
    np.testing.assert_array_equal(fake_forests[-1].y, [1, 0])
    pd.testing.assert_frame_equal(combined, original)

    out = capsys.readouterr().out
    assert "T1: max_features=log2, val ROC-AUC=1.0000, test ROC-AUC=0.0000" in out
    assert "T2: max_features=sqrt, val ROC-AUC=nan, test ROC-AUC=nan" in out
    assert "validation selection unavailable" in out
    assert "val overall: macro ROC-AUC=1.0000 (1/2 evaluable endpoints)" in out
    assert "test tox21: macro ROC-AUC=0.0000 (1/1 evaluable endpoints)" in out
    assert "test clintox: macro ROC-AUC=nan (0/1 evaluable endpoints)" in out

    [(model, rows)] = recorded
    assert model == "rf_toxicity"
    rows = pd.DataFrame(rows).set_index(["endpoint", "split"])
    assert len(rows) == 10
    assert (rows["metric"] == "roc_auc").all()
    assert (rows["encoder"] == "ecfp4-2048").all()

    t1 = rows.loc[("T1", "test")]
    assert (t1["dataset"], t1["value"], t1["n_labeled"], t1["n_positive"]) == (
        "tox21",
        0.0,
        2,
        1,
    )
    assert json.loads(t1["params"]) == {"max_features": "log2", "n_estimators": 500}
    assert rows.loc[("T1", "val"), "value"] == 1.0

    t2_val, t2_test = rows.loc[("T2", "val")], rows.loc[("T2", "test")]
    assert t2_val["dataset"] == "clintox"
    assert np.isnan(t2_val["value"]) and np.isnan(t2_test["value"])
    assert (t2_val["n_labeled"], t2_val["n_positive"]) == (0, 0)
    assert (t2_test["n_labeled"], t2_test["n_positive"]) == (2, 2)
    assert json.loads(t2_test["params"])["max_features"] == "sqrt"

    assert rows.loc[("macro_overall", "val"), "value"] == 1.0
    assert rows.loc[("macro_tox21", "test"), "value"] == 0.0
    macro = rows.loc[("macro_clintox", "test")]
    assert macro["dataset"] == "clintox"
    assert np.isnan(macro["value"]) and np.isnan(macro["n_labeled"])


@pytest.mark.parametrize(
    ("labels", "train_idx", "reason"),
    [
        ([np.nan, np.nan], [], "no observed training labels"),
        ([0, 0], [0, 1], "only one class"),
        ([1, 1], [0, 1], "only one class"),
    ],
)
def test_skip_untrainable_endpoint(labels, train_idx, reason, fake_forests):
    model, status = bt.fit_endpoint(
        np.ones((2, 1)),
        np.array(labels),
        np.array(train_idx, dtype=int),
        np.array([], dtype=int),
    )
    assert model is None
    assert reason in status
    assert fake_forests == []


@pytest.mark.parametrize("val_idx", [[], [2], [2, 3]])
def test_unavailable_validation_uses_only_sqrt(val_idx, fake_forests):
    model, status = bt.fit_endpoint(
        np.array([[2], [3], [6], [7]]),
        np.array([0, 1, 0, 0]),
        np.array([0, 1]),
        np.array(val_idx, dtype=int),
    )
    assert len(fake_forests) == 1
    assert model.max_features == "sqrt"
    assert "validation selection unavailable" in status


@pytest.mark.parametrize("test_labels", [[0, 1], [1, 0]])
def test_selection_is_independent_of_test_labels(test_labels, fake_forests):
    model, status = bt.fit_endpoint(
        np.array([[2], [3], [6], [7], [8], [9]]),
        np.array([0, 1, 0, 1, *test_labels]),
        np.array([0, 1]),
        np.array([2, 3]),
    )
    assert model.max_features == "log2"
    assert status == "selected by validation ROC-AUC"
    assert all(set(m.predicted) == {6, 7} for m in fake_forests)


@pytest.mark.parametrize("training_label", [np.nan, 0, 1])
def test_main_all_untrainable_reports_nan(
    monkeypatch, capsys, combined, fake_forests, training_label
):
    combined[bt.TOX_TASKS] = training_label
    monkeypatch.setattr(bt, "load_combined", lambda: combined)
    monkeypatch.setattr(
        bt,
        "scaffold_split",
        lambda df: df.assign(split=["train"] * 5 + ["val"] * 2 + ["test"] * 2),
    )
    bt.main()
    assert fake_forests == []
    out = capsys.readouterr().out
    assert "val overall: macro ROC-AUC=nan (0/2 evaluable endpoints)" in out
    assert "test overall: macro ROC-AUC=nan (0/2 evaluable endpoints)" in out
    assert "skipped:" in out


def test_real_forest_smoke(monkeypatch):
    monkeypatch.setattr(bt, "N_ESTIMATORS", 2)
    monkeypatch.setattr(bt, "MAX_FEATURES_OPTIONS", ["sqrt"])
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    model, _ = bt.fit_endpoint(X, np.array([0, 1, 0, 1]), np.arange(4), np.arange(4))
    scores = bt.positive_probabilities(model, X)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
