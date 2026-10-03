import json

import numpy as np
import pandas as pd
import pytest

from baseline import baseline_bioactivity as bb

# keyed by atom count, which fake fingerprint puts in column 0
PIC50 = {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 6: 5.0, 7: 7.0, 8: 5.0, 9: 8.0}


@pytest.fixture
def fake_forests(monkeypatch):
    models = []

    class Forest:
        def __init__(self, **kwargs):
            self.config = kwargs
            models.append(self)

        def fit(self, X, y):
            self.X = X.copy()
            self.y = y.copy()
            return self

        def predict(self, X):
            # log2 predicts every label exactly, the rest predict a constant
            if self.config["max_features"] == "log2":
                return np.array([PIC50[i] for i in X[:, 0]])
            return np.full(len(X), 6.5)

    monkeypatch.setattr(bb, "RandomForestRegressor", Forest)
    return models


@pytest.fixture
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bb, "record_results", lambda rows, model: calls.append((model, rows))
    )
    return calls


@pytest.fixture
def combined(monkeypatch):
    pic50 = [PIC50.get(i, np.nan) for i in range(1, 10)]
    df = pd.DataFrame({"smiles": ["C" * i for i in range(1, 10)], "pIC50_BACE1": pic50})
    df["active_BACE1"] = (
        (df["pIC50_BACE1"] >= 6).astype(float).where(df["pIC50_BACE1"].notna())
    )
    monkeypatch.setattr(bb, "load_combined", lambda: df)
    monkeypatch.setattr(
        bb,
        "scaffold_split",
        lambda df: df.assign(split=["train"] * 4 + ["val"] * 3 + ["test"] * 2),
    )
    monkeypatch.setattr(
        bb, "mol_to_ecfp4", lambda mol, n_bits: np.array([mol.GetNumAtoms()])
    )
    return df


def test_main_selects_on_val_and_records_val_and_test_rows(
    capsys, combined, fake_forests, recorded
):
    bb.main()

    assert len(fake_forests) == len(bb.MAX_FEATURES_OPTIONS)
    for model in fake_forests:
        np.testing.assert_array_equal(model.X[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(model.y, [5.0, 6.0, 7.0, 8.0])

    out = capsys.readouterr().out
    assert "Validation samples: 2" in out
    assert "Best max_features: log2" in out
    assert "Test: RMSE=0.0000, R2=1.0000, ROC-AUC=1.0000" in out

    [(model, rows)] = recorded
    assert model == "rf_bioactivity"
    rows = pd.DataFrame(rows).set_index(["split", "metric"])
    assert sorted(rows.index) == sorted(
        (split, metric)
        for split in ("val", "test")
        for metric in ("rmse", "r2", "roc_auc")
    )
    assert (rows["endpoint"] == "pIC50_BACE1").all()
    assert (rows["dataset"] == "bace1").all()
    assert (rows["encoder"] == "ecfp4-2048").all()
    assert (rows["n_labeled"] == 2).all()
    assert (rows["n_positive"] == 1).all()
    assert rows.loc[("test", "rmse"), "value"] == 0.0
    assert rows.loc[("val", "roc_auc"), "value"] == 1.0
    assert json.loads(rows.loc[("test", "r2"), "params"]) == {
        "max_features": "log2",
        "n_estimators": 500,
    }
