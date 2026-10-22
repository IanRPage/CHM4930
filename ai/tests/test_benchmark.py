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


@pytest.fixture
def snapshot(tmp_path):
    df = pd.DataFrame(
        {
            "smiles": ["C1" + "C" * (n - 1) + "1" for n in range(3, 23)],
            "pIC50_BACE1": [
                5.0 + (i % 4) * 0.5 if i < 8 or i >= 16 else np.nan for i in range(20)
            ],
            "pIC50_EGFR": [6.0 + i / 10 for i in range(20)],
            "in_bace1": True,
            "in_egfr": True,
        }
    )
    path = tmp_path / "combined.csv"
    df.to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture(scope="module")
def frozen_benchmark():
    before = {
        path: path.read_bytes() for path in (COMBINED_CSV_PATH, results.RESULTS_PATH)
    }
    yield bm.load_benchmark()
    for path, contents in before.items():
        assert path.read_bytes() == contents


@pytest.fixture
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)


def test_shared_split_is_reproducible_and_independent_of_training_seed(
    snapshot, preserve_rng_state
):
    bm.set_training_seed(0)
    first = bm.load_benchmark(snapshot)
    bm.set_training_seed(42)
    second = bm.load_benchmark(snapshot)
    pd.testing.assert_frame_equal(first, second)
    assert first["split"].tolist() == ["train"] * 16 + ["val"] * 2 + ["test"] * 2


def test_full_table_is_split_before_endpoint_filtering(snapshot, monkeypatch):
    calls = []

    def split_full_table(df, **kwargs):
        calls.append((len(df), df["pIC50_BACE1"].isna().sum(), kwargs))
        return scaffold_split(df, **kwargs)

    monkeypatch.setattr(bm, "scaffold_split", split_full_table)
    full = bm.load_benchmark(snapshot)
    endpoint = full.loc[full["pIC50_BACE1"].notna()]
    assert calls == [(20, 8, {"frac": (0.8, 0.1, 0.1), "seed": None})]
    assert endpoint["split"].value_counts().to_dict() == {
        "train": 8,
        "val": 2,
        "test": 2,
    }
    # Splitting just the endpoint changes this molecule's partition.
    separate = scaffold_split(endpoint.drop(columns=["scaffold", "split"]))
    assert full.loc[16, "split"] == "val"
    assert separate.loc[16, "split"] == "train"


def test_load_only_reads_snapshot_and_never_records_results(snapshot, monkeypatch):
    before = snapshot.read_bytes()

    def unexpected_write(*args, **kwargs):
        raise AssertionError("benchmark loading must not write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", unexpected_write)
    monkeypatch.setattr(bm, "record_results", unexpected_write)
    bm.load_benchmark(snapshot)
    assert snapshot.read_bytes() == before


def test_missing_snapshot_is_not_built_or_downloaded(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        bm.load_benchmark(missing)
    assert not missing.exists()


def test_metadata_is_stable_json_and_ignores_row_order_and_index(snapshot):
    full = bm.load_benchmark(snapshot)
    metadata = bm.benchmark_metadata(full, snapshot)
    reordered = full.iloc[::-1].set_axis(range(100, 120))
    assert metadata == bm.benchmark_metadata(reordered, snapshot)
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["split_fractions"] == [0.8, 0.1, 0.1]
    assert metadata["split_seed"] is None
    assert metadata["training_seed"] == 42
    assert metadata["active_threshold"] == 6.0
    assert metadata["n_molecules"] == 20
    assert metadata["bio_targets"] == ["BACE1", "EGFR"]
    assert len(metadata["tox_tasks"]) == 14
    bm.verify_benchmark_metadata(metadata, reordered, snapshot)
    observed = full["pIC50_BACE1"].notna()
    np.testing.assert_array_equal(
        full.loc[observed, "active_BACE1"],
        full.loc[observed, "pIC50_BACE1"] >= 6.0,
    )
    assert full.loc[~observed, "active_BACE1"].isna().all()


def test_split_identity_changes_when_membership_changes(snapshot):
    full = bm.load_benchmark(snapshot)
    original = bm.benchmark_metadata(full, snapshot)
    changed = full.copy()
    changed.loc[0, "split"] = "test"
    updated = bm.benchmark_metadata(changed, snapshot)
    assert original["data_sha256"] == updated["data_sha256"]
    assert original["split_sha256"] != updated["split_sha256"]
    with pytest.raises(ValueError, match="split_sha256"):
        bm.verify_benchmark_metadata(original, changed, snapshot)


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
    ],
)
def test_verification_rejects_incompatible_metadata(snapshot, field, value):
    full = bm.load_benchmark(snapshot)
    metadata = bm.benchmark_metadata(full, snapshot)
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        bm.verify_benchmark_metadata(metadata, full, snapshot)


def test_verification_requires_even_nullable_metadata_fields(snapshot):
    full = bm.load_benchmark(snapshot)
    metadata = bm.benchmark_metadata(full, snapshot)
    del metadata["split_seed"]
    with pytest.raises(ValueError, match="split_seed"):
        bm.verify_benchmark_metadata(metadata, full, snapshot)


@pytest.mark.parametrize("change", ["filter", "label", "active", "source"])
def test_metadata_rejects_filtered_or_modified_snapshot(snapshot, change):
    full = bm.load_benchmark(snapshot)
    if change == "filter":
        full = full[full["pIC50_BACE1"].notna()]
    else:
        column = {
            "label": "pIC50_BACE1",
            "active": "active_BACE1",
            "source": "in_bace1",
        }[change]
        full.loc[0, column] = False if change == "source" else 99.0
    with pytest.raises(ValueError, match="full, unchanged snapshot"):
        bm.benchmark_metadata(full, snapshot)


@pytest.mark.parametrize(
    "change", ["missing_column", "duplicate", "missing_smiles", "split"]
)
def test_metadata_rejects_invalid_identity(snapshot, change):
    full = bm.load_benchmark(snapshot)
    if change == "missing_column":
        full = full.drop(columns="split")
    elif change == "duplicate":
        full.loc[0, "smiles"] = full.loc[1, "smiles"]
    elif change == "missing_smiles":
        full.loc[0, "smiles"] = None
    else:
        full.loc[0, "split"] = "validation"
    with pytest.raises(ValueError):
        bm.benchmark_metadata(full, snapshot)


def test_regression_masks_missing_labels_and_keeps_filtered_alignment():
    labels = pd.Series([5.0, np.nan, 6.0, 7.0], index=[30, 10, 90, 20])
    predictions = pd.Series([5.0, np.nan, 6.0, 8.0], index=labels.index)
    metrics = bm.evaluate_predictions(labels, predictions, "regression")
    assert metrics == {"rmse": pytest.approx(np.sqrt(1 / 3)), "r2": pytest.approx(0.5)}
    selected = labels.notna()
    assert metrics == bm.evaluate_predictions(
        labels[selected], predictions[selected], "regression"
    )


def test_classification_masks_missing_labels_even_with_explicit_true_mask():
    metrics = bm.evaluate_predictions(
        [0, np.nan, 1], [0.1, np.nan, 0.9], "classification", mask=[True] * 3
    )
    assert metrics == {"roc_auc": 1.0}


@pytest.mark.parametrize(
    "invalid", [np.nan, np.inf, -np.inf, 0.5, -1, 0, 1, "True", "False", None, pd.NA]
)
def test_nonboolean_mask_values_are_rejected_even_for_missing_labels(invalid):
    with pytest.raises(ValueError, match="mask must contain only boolean values"):
        bm.evaluate_predictions(
            [np.nan, 0, 1],
            [np.nan, 0.1, 0.9],
            "classification",
            mask=[invalid, True, True],
        )


@pytest.mark.parametrize(
    "mask",
    [
        np.array([0, 1]),
        np.array([False, np.nan], dtype=object),
        pd.Series([True, pd.NA], dtype="boolean"),
        torch.tensor([0, 1]),
        torch.tensor([True, float("inf")]),
        torch.tensor([0, 1], dtype=torch.bfloat16),
    ],
)
def test_nonboolean_mask_containers_are_rejected(mask):
    with pytest.raises(ValueError, match="mask must contain only boolean values"):
        bm.evaluate_predictions([0, 1], [0.1, 0.9], "classification", mask)


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


def test_filled_multitask_labels_use_contract_masks_and_tensor_predictions():
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    predictions = torch.tensor(
        [[0.1, 0.9], [0.9, 0.1], [float("nan"), float("inf")]],
        requires_grad=True,
    )
    mask = torch.tensor([[True, True], [True, True], [False, False]])
    metrics = bm.evaluate_predictions(labels, predictions, "multitask", mask)
    assert metrics == {"per_task_roc_auc": [1.0, 1.0], "mean_roc_auc": 1.0}


@pytest.mark.parametrize("value", ["prediction", "mask"])
def test_misaligned_pandas_row_indexes_are_rejected(value):
    labels = pd.Series([0, 1], index=[10, 20])
    predictions = pd.Series([0.1, 0.9], index=labels.index)
    mask = pd.Series([True, True], index=labels.index)
    if value == "prediction":
        predictions = predictions.iloc[::-1]
    else:
        mask = mask.iloc[::-1]
    with pytest.raises(ValueError, match=f"{value} row index"):
        bm.evaluate_predictions(labels, predictions, "classification", mask)


@pytest.mark.parametrize("value", ["prediction", "mask"])
def test_misaligned_task_columns_are_rejected(value):
    labels = pd.DataFrame([[0, 1], [1, 0]], columns=["T1", "T2"])
    predictions = labels.astype(float)
    mask = pd.DataFrame(True, index=labels.index, columns=labels.columns)
    if value == "prediction":
        predictions = predictions[["T2", "T1"]]
    else:
        mask = mask[["T2", "T1"]]
    with pytest.raises(ValueError, match=f"{value} task columns"):
        bm.evaluate_predictions(labels, predictions, "multitask", mask)


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


def test_undefined_metrics_keep_existing_nan_behavior():
    assert np.isnan(bm.evaluate_predictions([], [], "regression")["rmse"])
    assert np.isnan(bm.evaluate_predictions([1], [1], "regression")["r2"])
    assert np.isnan(
        bm.evaluate_predictions([1, 1], [0.1, 0.9], "classification")["roc_auc"]
    )


def test_seed_setup_repeats_all_cpu_random_streams(preserve_rng_state):
    deterministic = torch.are_deterministic_algorithms_enabled()

    def sample(seed):
        generator = bm.set_training_seed(seed)
        return (
            random.random(),
            np.random.random(4),
            torch.rand(4),
            torch.randperm(20, generator=generator),
        )

    first = sample(42)
    second = sample(42)
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])
    assert not torch.equal(first[3], sample(0)[3])
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert bm.set_training_seed().initial_seed() == 42


def test_cuda_seed_is_set_when_available(monkeypatch, preserve_rng_state):
    calls = []
    monkeypatch.setattr(
        torch, "manual_seed", lambda seed: calls.append(("torch", seed))
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda seed: calls.append(("cuda", seed))
    )
    bm.set_training_seed()
    assert calls == [("torch", 42), ("cuda", 42)]


@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.5])
def test_invalid_training_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="training seed"):
        bm.set_training_seed(seed)


def test_lf_and_crlf_checksums_match_without_rewriting_files(tmp_path):
    lf = b"smiles,pIC50_BACE1\nCCO,6.0\nCCN,5.0\n"
    expected = hashlib.sha256(lf).hexdigest()
    for name, contents in (("lf.csv", lf), ("crlf.csv", lf.replace(b"\n", b"\r\n"))):
        path = tmp_path / name
        path.write_bytes(contents)
        assert results.file_sha256(path) == expected
        assert path.read_bytes() == contents
    changed = tmp_path / "changed.csv"
    changed.write_bytes(lf.replace(b"6.0", b"7.0"))
    assert results.file_sha256(changed) != expected


def test_original_recorded_checksum_remains_compatible(frozen_benchmark):
    recorded = pd.read_csv(results.RESULTS_PATH)
    rf = recorded[recorded["model"].isin(["rf_bioactivity", "rf_toxicity"])]
    assert len(rf) == 46
    original = "b340ddf73f38e10f8b6a2948156417661a269ffc9a6d9b097b538c4690e1b041"
    assert set(rf["data_sha256"]) == {original}
    assert results.file_sha256(COMBINED_CSV_PATH) == original
    assert bm.benchmark_metadata(frozen_benchmark)["data_sha256"] == original
    assert recorded.columns.tolist() == results.COLUMNS


def test_current_shared_partitions_match_existing_rf_label_counts(frozen_benchmark):
    recorded = pd.read_csv(results.RESULTS_PATH)
    endpoints = recorded[
        recorded["model"].isin(["rf_bioactivity", "rf_toxicity"])
        & recorded["n_labeled"].notna()
    ]
    for row in endpoints.itertuples(index=False):
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


def test_existing_recorder_preserves_all_rf_rows_on_a_copy(tmp_path, frozen_benchmark):
    original = results.RESULTS_PATH.read_bytes()
    path = tmp_path / "results.csv"
    path.write_bytes(original)
    before = pd.read_csv(path)
    rows = [
        {
            "encoder": "fixture",
            "dataset": "bace1",
            "endpoint": "pIC50_BACE1",
            "split": "val",
            "metric": "rmse",
            "value": 0.5,
            "n_labeled": 10,
            "n_positive": 5,
            "params": json.dumps(
                {"benchmark": bm.benchmark_metadata(frozen_benchmark)}
            ),
        }
    ]
    bm.record_results(rows, model="benchmark_fixture", path=path)
    written = pd.read_csv(path)
    preserved = written[written["model"] != "benchmark_fixture"].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, preserved, rtol=1e-14, atol=1e-14)
    assert len(written) == len(before) + 1
    assert written.columns.tolist() == results.COLUMNS
    assert results.RESULTS_PATH.read_bytes() == original
