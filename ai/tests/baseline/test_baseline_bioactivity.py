import json

import numpy as np
import pandas as pd
import pytest

from baseline import baseline_bioactivity as bb

# keyed by atom count, which fake fingerprint puts in column 0
# BACE1 labels molecules 1-9, EGFR labels 10-18
PIC50 = {
    "BACE1": {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 6: 5.0, 7: 7.0, 8: 5.0, 9: 8.0},
    "EGFR": {10: 6.0, 11: 7.0, 12: 5.0, 13: 8.0, 15: 5.5, 16: 7.5, 17: 6.5, 18: 4.0},
}
ALL_PIC50 = PIC50["BACE1"] | PIC50["EGFR"]


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
                return np.array([ALL_PIC50[i] for i in X[:, 0]])
            return np.full(len(X), 6.5)

    monkeypatch.setattr(bb, "RandomForestRegressor", Forest)
    return models


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    calls = []
    monkeypatch.setattr(
        bb, "record_results", lambda rows, model: calls.append((model, rows))
    )
    return calls


@pytest.fixture
def combined(monkeypatch):
    atoms = range(1, 19)
    df = pd.DataFrame({"smiles": ["C" * i for i in atoms]})
    for target, labels in PIC50.items():
        pic50 = pd.Series([labels.get(i, np.nan) for i in atoms])
        df[f"pIC50_{target}"] = pic50
        df[f"active_{target}"] = (pic50 >= 6).astype(float).where(pic50.notna())
    monkeypatch.setattr(bb, "load_combined", lambda: df)
    monkeypatch.setattr(
        bb,
        "scaffold_split",
        lambda df: df.assign(split=(["train"] * 4 + ["val"] * 3 + ["test"] * 2) * 2),
    )
    monkeypatch.setattr(
        bb, "mol_to_ecfp4", lambda mol, n_bits: np.array([mol.GetNumAtoms()])
    )
    return df


def test_main_fits_each_target_on_its_own_labels(capsys, combined, fake_forests):
    bb.main()

    n = len(bb.MAX_FEATURES_OPTIONS)
    assert len(fake_forests) == 2 * n
    for model in fake_forests[:n]:
        np.testing.assert_array_equal(model.X[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(model.y, [5.0, 6.0, 7.0, 8.0])
    for model in fake_forests[n:]:
        np.testing.assert_array_equal(model.X[:, 0], [10, 11, 12, 13])
        np.testing.assert_array_equal(model.y, [6.0, 7.0, 5.0, 8.0])

    out = capsys.readouterr().out
    assert out.count("Validation samples: 2") == 2
    assert out.count("Best max_features: log2") == 2
    assert out.count("Test: RMSE=0.0000, R2=1.0000, ROC-AUC=1.0000") == 2
    assert out.index("BACE1") < out.index("EGFR")


@pytest.mark.parametrize("target", ["BACE1", "EGFR"])
def test_main_records_val_and_test_rows_per_target(
    target, combined, fake_forests, recorded
):
    bb.main()

    [(model, rows)] = recorded
    assert model == "rf_bioactivity"
    rows = pd.DataFrame(rows)
    assert len(rows) == 12
    rows = rows[rows["endpoint"] == f"pIC50_{target}"].set_index(["split", "metric"])
    assert sorted(rows.index) == sorted(
        (split, metric)
        for split in ("val", "test")
        for metric in ("rmse", "r2", "roc_auc")
    )
    assert (rows["dataset"] == target.lower()).all()
    assert (rows["encoder"] == "ecfp4-2048").all()
    assert (rows["n_labeled"] == 2).all()
    assert (rows["n_positive"] == 1).all()
    assert rows.loc[("test", "rmse"), "value"] == 0.0
    assert rows.loc[("val", "roc_auc"), "value"] == 1.0
    assert json.loads(rows.loc[("test", "r2"), "params"]) == {
        "max_features": "log2",
        "n_estimators": 500,
    }
