import json

import numpy as np
import pandas as pd
import pytest

from baseline import baseline_bioactivity as bb
from baseline import benchmark, results

# keyed by atom count, which fake fingerprint puts in column 0
# BACE1 labels molecules 1-9, EGFR labels 10-18
PIC50 = {
    "BACE1": {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 6: 5.0, 7: 7.0, 8: 5.0, 9: 8.0},
    "EGFR": {10: 6.0, 11: 7.0, 12: 5.0, 13: 8.0, 15: 5.5, 16: 7.5, 17: 6.5, 18: 4.0},
}
ALL_PIC50 = PIC50["BACE1"] | PIC50["EGFR"]
TEST_SEEDS = (0, 1)
BIO_METRICS = ("rmse", "r2", "spearman", "kendall", "roc_auc", "pr_auc", "prevalence")


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
def recorded(monkeypatch, tmp_path):
    calls = []
    out_path = tmp_path / "results.csv"
    data_path = tmp_path / "combined.csv"
    data_path.write_text("smiles\nCCO\n")
    monkeypatch.setattr(results, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(bb, "SEEDS", TEST_SEEDS)
    monkeypatch.setattr(benchmark, "SEEDS", TEST_SEEDS)
    monkeypatch.setattr(bb, "SPLIT_IDS", ("default",))

    real_record = benchmark.record_results

    def capture_record(rows, model, path=out_path, data_path=data_path, **kwargs):
        calls.append((model, rows))
        return real_record(
            rows, model=model, path=out_path, data_path=data_path, **kwargs
        )

    monkeypatch.setattr(benchmark, "record_results", capture_record)
    return calls


@pytest.fixture
def combined(monkeypatch):
    atoms = range(1, 19)
    df = pd.DataFrame({"smiles": ["C" * i for i in atoms]})
    for target, labels in PIC50.items():
        pic50 = pd.Series([labels.get(i, np.nan) for i in atoms])
        df[f"pIC50_{target}"] = pic50
        df[f"active_{target}"] = (pic50 >= 6).astype(float).where(pic50.notna())
    df["split"] = (["train"] * 4 + ["val"] * 3 + ["test"] * 2) * 2
    monkeypatch.setattr(bb, "load_benchmark_split", lambda **_: df)
    monkeypatch.setattr(
        bb, "mol_to_ecfp4", lambda mol, n_bits: np.array([mol.GetNumAtoms()])
    )
    return df


def test_main_selects_once_and_fits_each_target_per_seed(
    capsys, combined, fake_forests
):
    bb.main()

    n_options = len(bb.MAX_FEATURES_OPTIONS)
    n_seeds = len(TEST_SEEDS)
    # 2 targets * n_options for selection + 2 targets * n_seeds for benchmark
    assert len(fake_forests) == 2 * n_options + 2 * n_seeds

    # Selection models: first n_options for BACE1, next n_options for EGFR, all seeded with SEEDS[0]
    for model in fake_forests[:n_options]:
        np.testing.assert_array_equal(model.X[:, 0], [1, 2, 3, 4])
        np.testing.assert_array_equal(model.y, [5.0, 6.0, 7.0, 8.0])
        assert model.config["random_state"] == TEST_SEEDS[0]
    for model in fake_forests[n_options : 2 * n_options]:
        np.testing.assert_array_equal(model.X[:, 0], [10, 11, 12, 13])
        np.testing.assert_array_equal(model.y, [6.0, 7.0, 5.0, 8.0])
        assert model.config["random_state"] == TEST_SEEDS[0]

    # Benchmark models: for each seed, BACE1 then EGFR with selected max_features='log2'
    bench_models = fake_forests[2 * n_options :]
    for s_idx, seed in enumerate(TEST_SEEDS):
        bace_model = bench_models[2 * s_idx]
        egfr_model = bench_models[2 * s_idx + 1]
        assert bace_model.config["max_features"] == "log2"
        assert bace_model.config["random_state"] == seed
        assert egfr_model.config["max_features"] == "log2"
        assert egfr_model.config["random_state"] == seed

    out = capsys.readouterr().out
    assert out.count("Validation samples: 2") == 2
    assert out.count("Best max_features: log2") == 2
    assert out.count("Test: RMSE=0.0000, R2=1.0000, ROC-AUC=1.0000") == 2
    assert out.index("BACE1") < out.index("EGFR")


@pytest.mark.parametrize("target", ["BACE1", "EGFR"])
def test_main_records_val_and_test_rows_per_target_and_seed(
    target, combined, fake_forests, recorded
):
    bb.main()

    # one record_results call per seed, each holding only that seed's rows
    assert [model for model, _ in recorded] == ["rf_bioactivity"] * len(TEST_SEEDS)
    assert [{row["seed"] for row in rows} for _, rows in recorded] == [
        {seed} for seed in TEST_SEEDS
    ]
    rows = pd.DataFrame([row for _, seed_rows in recorded for row in seed_rows])
    # 2 targets * 2 splits * 7 metrics per seed
    assert len(rows) == 28 * len(TEST_SEEDS)
    rows = rows[rows["endpoint"] == f"pIC50_{target}"].set_index(
        ["split", "seed", "metric"]
    )
    assert sorted(rows.index) == sorted(
        (split, seed, metric)
        for split in ("val", "test")
        for seed in TEST_SEEDS
        for metric in BIO_METRICS
    )
    assert (rows["dataset"] == target.lower()).all()
    assert (rows["encoder"] == "ecfp4-2048").all()
    assert (rows["n_labeled"] == 2).all()
    assert (rows["n_positive"] == 1).all()
    assert rows.loc[("test", 0, "rmse"), "value"] == 0.0
    assert rows.loc[("val", 1, "roc_auc"), "value"] == 1.0
    assert json.loads(rows.loc[("test", 0, "r2"), "params"]) == {
        "max_features": {"BACE1": "log2", "EGFR": "log2"},
        "n_estimators": 500,
    }
