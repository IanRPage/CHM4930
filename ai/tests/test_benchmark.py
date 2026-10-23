import hashlib
import json
import random

import numpy as np
import pandas as pd
import pytest
import torch

import benchmark as bm
from baseline import results
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.split import scaffold_split


def test_shared_split_is_reproducible_and_independent_of_training_seed(snapshot_csv):
    bm.set_training_seed(0)
    first = bm.load_benchmark(snapshot_csv)
    bm.set_training_seed(42)
    second = bm.load_benchmark(snapshot_csv)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].tolist() == ["train"] * 16 + ["val"] * 2 + ["test"] * 2


def test_full_table_is_split_before_endpoint_filtering(snapshot_csv, monkeypatch):
    calls = []

    def split_full_table(df, **kwargs):
        calls.append((len(df), df["pIC50_BACE1"].isna().sum(), kwargs))
        return scaffold_split(df, **kwargs)

    monkeypatch.setattr(bm, "scaffold_split", split_full_table)
    full = bm.load_benchmark(snapshot_csv)
    endpoint = full.loc[full["pIC50_BACE1"].notna()]
    assert calls == [(20, 8, {"frac": (0.8, 0.1, 0.1), "seed": None})]
    assert endpoint["split"].value_counts().to_dict() == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    separate = scaffold_split(endpoint.drop(columns=["scaffold", "split"]))
    assert full.loc[16, "split"] == "val"
    assert separate.loc[16, "split"] == "train"


def test_load_is_read_only_and_missing_snapshot_raises(
    snapshot_csv, tmp_path, monkeypatch
):
    before = snapshot_csv.read_bytes()
    monkeypatch.setattr(
        pd.DataFrame,
        "to_csv",
        lambda *args, **kwargs: pytest.fail("benchmark loading must not write"),
    )
    bm.load_benchmark(snapshot_csv)
    assert snapshot_csv.read_bytes() == before

    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        bm.load_benchmark(missing)
    assert not missing.exists()


def test_metadata_is_stable_json_and_ignores_row_order_and_index(snapshot_csv):
    full = bm.load_benchmark(snapshot_csv)
    metadata = bm.benchmark_metadata(full, snapshot_csv)
    reordered = full.iloc[::-1].set_axis(range(100, 120))
    assert metadata == bm.benchmark_metadata(reordered, snapshot_csv)
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["split_fractions"] == [0.8, 0.1, 0.1]
    assert metadata["split_seed"] is None
    assert metadata["training_seed"] == 42
    assert metadata["active_threshold"] == 6.0
    assert metadata["n_molecules"] == 20
    assert metadata["bio_targets"] == ["BACE1", "EGFR"]
    assert len(metadata["tox_tasks"]) == 14
    bm.verify_benchmark_metadata(metadata, reordered, snapshot_csv)
    assert bm.config_json({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    with pytest.raises(ValueError, match="discard RF results"):
        bm.verify_export(metadata, full, snapshot_csv)
    observed = full["pIC50_BACE1"].notna()
    np.testing.assert_array_equal(
        full.loc[observed, "active_BACE1"],
        full.loc[observed, "pIC50_BACE1"] >= 6.0,
    )
    assert full.loc[~observed, "active_BACE1"].isna().all()


def test_split_identity_changes_when_membership_changes(snapshot_csv):
    full = bm.load_benchmark(snapshot_csv)
    original = bm.benchmark_metadata(full, snapshot_csv)
    changed = full.copy()
    changed.loc[0, "split"] = "test"
    updated = bm.benchmark_metadata(changed, snapshot_csv)
    assert original["data_sha256"] == updated["data_sha256"]
    assert original["split_sha256"] != updated["split_sha256"]
    with pytest.raises(ValueError, match="split_sha256"):
        bm.verify_benchmark_metadata(original, changed, snapshot_csv)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_version", 2),
        ("data_sha256", "old"),
        ("split_fractions", [0.6, 0.2, 0.2]),
        ("split_seed", 42),
        ("training_seed", 0),
        ("active_threshold", 7.0),
        ("metrics", {}),
        ("split_seed", None),  # deleted key test
    ],
)
def test_verification_rejects_incompatible_metadata(snapshot_csv, field, value):
    full = bm.load_benchmark(snapshot_csv)
    metadata = bm.benchmark_metadata(full, snapshot_csv)
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    with pytest.raises(ValueError, match=field):
        bm.verify_benchmark_metadata(metadata, full, snapshot_csv)


@pytest.mark.parametrize(
    "change",
    [
        "filter",
        "label",
        "active",
        "source",
        "missing_column",
        "duplicate",
        "missing_smiles",
        "split",
    ],
)
def test_metadata_rejects_modified_or_invalid_table(snapshot_csv, change):
    full = bm.load_benchmark(snapshot_csv)
    if change == "filter":
        full = full[full["pIC50_BACE1"].notna()]
    elif change in ("label", "active", "source"):
        col = {"label": "pIC50_BACE1", "active": "active_BACE1", "source": "in_bace1"}[
            change
        ]
        full.loc[0, col] = False if change == "source" else 99.0
    elif change == "missing_column":
        full = full.drop(columns="split")
    elif change == "duplicate":
        full.loc[0, "smiles"] = full.loc[1, "smiles"]
    elif change == "missing_smiles":
        full.loc[0, "smiles"] = None
    else:
        full.loc[0, "split"] = "validation"
    with pytest.raises(ValueError):
        bm.benchmark_metadata(full, snapshot_csv)


def test_evaluate_predictions_masks_and_task_types():
    labels = pd.Series([5.0, np.nan, 6.0, 7.0], index=[30, 10, 90, 20])
    predictions = pd.Series([5.0, np.nan, 6.0, 8.0], index=labels.index)
    metrics = bm.evaluate_predictions(labels, predictions, "regression")
    assert metrics == {"rmse": pytest.approx(np.sqrt(1 / 3)), "r2": pytest.approx(0.5)}
    selected = labels.notna()
    assert metrics == bm.evaluate_predictions(
        labels[selected], predictions[selected], "regression"
    )
    assert bm.evaluate_predictions(
        [0, np.nan, 1], [0.1, np.nan, 0.9], "classification", mask=[True] * 3
    ) == {"roc_auc": 1.0}

    mt_labels = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    mt_preds = torch.tensor(
        [[0.1, 0.9], [0.9, 0.1], [float("nan"), float("inf")]], requires_grad=True
    )
    mt_mask = torch.tensor([[True, True], [True, True], [False, False]])
    assert bm.evaluate_predictions(mt_labels, mt_preds, "multitask", mt_mask) == {
        "per_task_roc_auc": [1.0, 1.0],
        "mean_roc_auc": 1.0,
    }


@pytest.mark.parametrize(
    "mask",
    [
        [np.nan, True],
        [0, True],
        ["True", True],
        np.array([0, 1]),
        pd.Series([True, pd.NA], dtype="boolean"),
        torch.tensor([0, 1]),
    ],
)
def test_nonboolean_masks_are_rejected(mask):
    with pytest.raises(ValueError, match="mask must contain only boolean values"):
        bm.evaluate_predictions([np.nan, 1], [np.nan, 0.9], "classification", mask)


@pytest.mark.parametrize(
    "mask",
    [
        [True, True, False, True],
        np.array([True, True, False, True]),
        np.array([True, True, False, True], dtype=object),
        pd.Series([True, True, False, True], dtype="boolean"),
        torch.tensor([True, True, False, True]),
    ],
)
def test_boolean_mask_formats_preserve_missing_label_and_prediction_behavior(mask):
    assert bm.evaluate_predictions(
        [0, 1, 0, np.nan], [0.1, 0.9, np.nan, np.inf], "classification", mask
    ) == {"roc_auc": 1.0}


@pytest.mark.parametrize("target", ["prediction", "mask"])
def test_misaligned_pandas_indexes_and_columns_are_rejected(target):
    labels = pd.Series([0, 1], index=[10, 20])
    preds = pd.Series([0.1, 0.9], index=labels.index)
    mask = pd.Series([True, True], index=labels.index)
    with pytest.raises(ValueError, match=f"{target} row index"):
        bm.evaluate_predictions(
            labels,
            preds.iloc[::-1] if target == "prediction" else preds,
            "classification",
            mask.iloc[::-1] if target == "mask" else mask,
        )

    df_labels = pd.DataFrame([[0, 1], [1, 0]], columns=["T1", "T2"])
    df_preds = df_labels.astype(float)
    df_mask = pd.DataFrame(True, index=df_labels.index, columns=df_labels.columns)
    with pytest.raises(ValueError, match=f"{target} task columns"):
        bm.evaluate_predictions(
            df_labels,
            df_preds[["T2", "T1"]] if target == "prediction" else df_preds,
            "multitask",
            df_mask[["T2", "T1"]] if target == "mask" else df_mask,
        )


@pytest.mark.parametrize(
    ("labels", "predictions", "mask", "message"),
    [
        ([1, 2], [1], None, "same shape"),
        ([1, 2], [1, 2], [True], "mask must have the same shape"),
        ([1, np.inf], [1, 2], None, "observed labels must be finite"),
        ([1, 2], [1, np.inf], None, "observed predictions must be finite"),
        ([1, 2], [1, np.nan], None, "observed predictions must be finite"),
    ],
)
def test_evaluation_rejects_invalid_observed_inputs(labels, predictions, mask, message):
    with pytest.raises(ValueError, match=message):
        bm.evaluate_predictions(labels, predictions, "regression", mask)


def test_training_seed_reproducibility_cuda_and_validation(monkeypatch):
    def sample(seed):
        generator = bm.set_training_seed(seed)
        return (
            random.random(),
            np.random.random(4),
            torch.rand(4),
            torch.randperm(20, generator=generator),
        )

    first, second = sample(42), sample(42)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])
    assert not torch.equal(first[3], sample(0)[3])
    assert bm.set_training_seed().initial_seed() == 42

    for bad_seed in (-1, 2**32, True, 0.5):
        with pytest.raises(ValueError, match="training seed"):
            bm.set_training_seed(bad_seed)

    calls = []
    monkeypatch.setattr(torch, "manual_seed", lambda s: calls.append(("torch", s)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda s: calls.append(("cuda", s))
    )
    bm.set_training_seed(42)
    assert calls == [("torch", 42), ("cuda", 42)]


def test_lf_and_crlf_checksums_match_without_rewriting_files(tmp_path):
    lf = b"smiles,pIC50_BACE1\nCCO,6.0\nCCN,5.0\n"
    expected = hashlib.sha256(lf).hexdigest()
    for name, contents in (("lf.csv", lf), ("crlf.csv", lf.replace(b"\n", b"\r\n"))):
        path = tmp_path / name
        path.write_bytes(contents)
        assert results.file_sha256(path) == expected
        assert path.read_bytes() == contents


def test_frozen_benchmark_matches_recorded_rf_checksum_and_counts(frozen_benchmark):
    recorded = pd.read_csv(results.RESULTS_PATH)
    rf = recorded[recorded["model"].isin(["rf_bioactivity", "rf_toxicity"])]
    assert len(rf) == 46
    original = "b340ddf73f38e10f8b6a2948156417661a269ffc9a6d9b097b538c4690e1b041"
    assert set(rf["data_sha256"]) == {original}
    assert results.file_sha256(COMBINED_CSV_PATH) == original
    assert bm.benchmark_metadata(frozen_benchmark)["data_sha256"] == original

    for row in rf[rf["n_labeled"].notna()].itertuples(index=False):
        labels = frozen_benchmark.loc[
            frozen_benchmark["split"] == row.split, row.endpoint
        ].dropna()
        assert len(labels) == row.n_labeled
        positive = (
            labels >= bm.ACTIVE_THRESHOLD
            if row.model == "rf_bioactivity"
            else labels == 1
        )
        assert positive.sum() == row.n_positive
