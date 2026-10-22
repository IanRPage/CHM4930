import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import benchmark as bm
from baseline import results
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu import benchmark as mb
from mufu import contract as c
from mufu.model import MuFu
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.preprocess import featurize


@pytest.fixture(autouse=True)
def preserve_files_and_rng(monkeypatch):
    protected = {p: p.read_bytes() for p in (COMBINED_CSV_PATH, results.RESULTS_PATH)}
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    record = bm.record_results

    def copy_only(rows, **kwargs):
        assert (
            kwargs.get("path", results.RESULTS_PATH).resolve()
            != results.RESULTS_PATH.resolve()
        )
        return record(rows, **kwargs)

    monkeypatch.setattr(bm, "record_results", copy_only)
    yield
    for path, contents in protected.items():
        assert path.read_bytes() == contents
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def snapshot(tmp_path):
    frame = pd.DataFrame(
        {
            "smiles": ["C1" + "C" * (n - 1) + "1" for n in range(3, 23)],
            "pIC50_BACE1": [5.0 + 2 * (i % 2) for i in range(20)],
            "pIC50_EGFR": [8.0 - 4 * (i % 2) for i in range(20)],
            **{
                flag: True for flag in ("in_bace1", "in_egfr", "in_tox21", "in_clintox")
            },
        }
    )
    frame.loc[19, "pIC50_EGFR"] = np.nan
    for j, task in enumerate(c.TOX_COLUMNS):
        frame[task] = [(i + j) % 2 for i in range(20)]
    frame.loc[17, c.TOX_COLUMNS[0]] = np.nan
    frame.loc[16:, c.TOX_COLUMNS[1]] = np.nan
    frame.loc[16:, c.TOX_COLUMNS[2]] = 0
    path = tmp_path / "combined.csv"
    frame.to_csv(path, index=False)
    full = bm.load_benchmark(path)
    metadata = bm.benchmark_metadata(full, path)
    return path, full, metadata


def provenance(metadata):
    return {
        "trained": True,
        "training_steps": 10,
        "training_seed": 42,
        "selection_split": "val",
        "benchmark": copy.deepcopy(metadata),
        "run_id": "mock-test-only",
    }


def synthetic_predictions(cohort):
    labels = cohort.set_index("smiles")
    pic50 = labels[list(c.PIC50_COLUMNS)].fillna(-100.0) + np.array([1.0, 2.0])
    active = 1.0 - 2.0 * labels[list(c.ACTIVE_COLUMNS)].fillna(0.0)
    tox = 4.0 * labels[list(c.TOX_COLUMNS)].fillna(1000.0) - 2.0
    return {"pic50": pic50, "active_logits": active, "tox_logits": tox}


class LookupMuFu(nn.Module):
    """Synthetic endpoint outputs keyed by molecule, not loader position."""

    def __init__(self, frame):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(0.9)
        self.lookup = synthetic_predictions(frame)
        self.calls = []

    def forward(self, inputs):
        assert set(inputs) == set(c.MuFuInputs.__annotations__)
        assert not self.training and not self.dropout.training
        assert not torch.is_grad_enabled() and torch.is_inference_mode_enabled()
        self.calls.append(list(inputs["smiles"]))
        # Verify fingerprints and graph order against their molecule identities.
        graphs = inputs["graph"].to_data_list()
        for i, smiles in enumerate(inputs["smiles"]):
            expected = featurize(smiles)
            assert torch.equal(inputs["fingerprint"][i].cpu(), expected.fp[0].float())
            assert torch.equal(graphs[i].x.cpu(), expected.x)
            assert torch.equal(graphs[i].edge_index.cpu(), expected.edge_index)
        return {
            key: torch.tensor(
                table.loc[inputs["smiles"]].to_numpy(), dtype=torch.float64
            )
            + self.anchor * 0
            for key, table in self.lookup.items()
        }


def evaluate(snapshot, model=None, **kwargs):
    path, full, metadata = snapshot
    return mb.evaluate_mufu(
        model if model is not None else LookupMuFu(full),
        data_path=path,
        training=provenance(metadata),
        **kwargs,
    )


def test_complete_sequential_coverage_alignment_eval_mode_and_rng(snapshot):
    _, full, _ = snapshot
    model = LookupMuFu(full).train()
    model.dropout.eval()  # Restore mixed component modes, not just the root flag.
    modes = {m: m.training for m in model.modules()}
    before = torch.get_rng_state().clone()
    result = evaluate(snapshot, model, splits=("test", "val"), batch_size=1)
    assert all(m.training == mode for m, mode in modes.items())
    assert torch.equal(torch.get_rng_state(), before)
    assert model.anchor.grad is None
    expected_smiles = full.loc[full["split"].isin(["val", "test"]), "smiles"].tolist()
    assert [smiles for batch in model.calls for smiles in batch] == expected_smiles
    assert len(model.calls) == 4
    assert result.training_attested
    assert len(result.rows) == 50
    for split in ("val", "test"):
        expected = synthetic_predictions(full.loc[full["split"] == split])
        for key, table in result.predictions[split].items():
            pd.testing.assert_frame_equal(table, expected[key], check_dtype=False)


def test_output_mapping_metrics_masks_and_result_identities(snapshot):
    result = evaluate(snapshot, splits=("val", "test"))
    table = pd.DataFrame(result.rows)
    val = table[table["split"] == "val"].set_index(["endpoint", "metric"])
    for pic50, rmse in zip(c.PIC50_COLUMNS, (1.0, 2.0)):
        assert val.loc[(pic50, "rmse"), "value"] == rmse
        assert val.loc[(pic50, "r2"), "value"] == 0
        assert val.loc[(pic50, "roc_auc"), "value"] == 1
        assert val.loc[(pic50, "roc_auc"), "n_positive"] == 1
    for active in c.ACTIVE_COLUMNS:
        assert val.loc[(active, "roc_auc"), "value"] == 0
    assert not table.duplicated(["dataset", "endpoint", "split", "metric"]).any()
    for j, task in enumerate(c.TOX_COLUMNS):
        row = val.loc[(task, "roc_auc")]
        if j < 3:
            assert np.isnan(row["value"])
        else:
            assert row["value"] == 1
        assert row["n_labeled"] == (1 if j == 0 else 0 if j == 1 else 2)
    for name in ("overall", "tox21", "clintox"):
        assert val.loc[(f"macro_{name}", "roc_auc"), "value"] == 1
        assert np.isnan(val.loc[(f"macro_{name}", "roc_auc"), "n_labeled"])
    test_egfr = table.loc[
        (table["split"] == "test") & (table["endpoint"] == "pIC50_EGFR")
    ].set_index("metric")
    assert test_egfr.loc["rmse", "n_labeled"] == 1
    assert np.isnan(test_egfr.loc["roc_auc", "value"])
    params = table["params"].map(json.loads)
    assert all(
        p["score_source"] == "predicted_pic50"
        for p in params[table["endpoint"].isin(c.PIC50_COLUMNS)]
    )
    assert all(
        p["score_source"] == "active_logits"
        for p in params[table["endpoint"].isin(c.ACTIVE_COLUMNS)]
    )


def test_real_integrated_mufu_accepts_structure_only_batches(snapshot):
    class TinySmiles(nn.Module):
        out_dim = 4

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(128, 4)

        def forward(self, smiles):
            return torch.stack(
                [
                    self.embedding(torch.tensor(list(map(ord, s)))).mean(0)
                    for s in smiles
                ]
            )

    model = MuFu(
        TinySmiles(), FingerprintEncoder(hidden_dims=(8,)), GNNEncoder(hidden_dims=(8,))
    )
    result = mb.evaluate_mufu(model, data_path=snapshot[0], batch_size=1)
    assert model.training
    assert not result.training_attested
    assert len(result.rows) == 25
    assert {row["split"] for row in result.rows} == {"val"}
    assert all(torch.isfinite(tensor).all() for tensor in model.state_dict().values())


@pytest.mark.parametrize("change", ["dropped", "reordered", "mask", "labels"])
def test_incompatible_datasets_fail_and_restore_modes(snapshot, monkeypatch, change):
    original = mb.MuFuDataset

    def bad_dataset(frame):
        dataset = original(frame)
        if change == "dropped":
            dataset.smiles.pop()
        elif change == "reordered":
            dataset.smiles.reverse()
        elif change == "mask":
            dataset.masks["mask_bio"][0, 0] = False
        else:
            dataset.labels["y_pic50"][0, 0] += 100
        return dataset

    monkeypatch.setattr(mb, "MuFuDataset", bad_dataset)
    model = LookupMuFu(snapshot[1]).train()
    with pytest.raises(ValueError, match="cohort|order|mask|misaligned"):
        evaluate(snapshot, model)
    assert model.training
    assert not model.calls


def test_incomplete_loader_is_not_silently_scored(snapshot, monkeypatch):
    original = mb.DataLoader

    def incomplete(*args, **kwargs):
        loader = original(*args, **kwargs)
        return [next(iter(loader))]

    monkeypatch.setattr(mb, "DataLoader", incomplete)
    with pytest.raises(ValueError, match="incomplete evaluation"):
        evaluate(snapshot, batch_size=1)


def test_collate_reordering_is_rejected(snapshot, monkeypatch):
    original = mb.collate
    monkeypatch.setattr(mb, "collate", lambda examples: original(examples[::-1]))
    with pytest.raises(ValueError, match="SMILES coverage/order"):
        evaluate(snapshot)


def test_filtered_full_table_is_rejected(snapshot, monkeypatch):
    monkeypatch.setattr(bm, "load_benchmark", lambda path: snapshot[1].iloc[:-1])
    with pytest.raises(ValueError, match="full, unchanged snapshot"):
        evaluate(snapshot)


@pytest.mark.parametrize("change", ["keys", "shape", "dtype", "nonfinite"])
def test_bad_model_outputs_are_rejected(snapshot, change):
    class BadOutput(LookupMuFu):
        def forward(self, inputs):
            outputs = super().forward(inputs)
            if change == "keys":
                del outputs["active_logits"]
            elif change == "shape":
                outputs["tox_logits"] = outputs["tox_logits"][:, :12]
            elif change == "dtype":
                outputs["pic50"] = outputs["pic50"].long()
            else:
                outputs["pic50"][0, 0] = float("nan")
            return outputs

    with pytest.raises(ValueError, match="keys|shape|dtype|finite"):
        evaluate(snapshot, BadOutput(snapshot[1]))


@pytest.mark.parametrize(
    "case",
    [
        "trained",
        "steps",
        "seed",
        "selection",
        "data",
        "split",
        "missing_metadata",
        "missing_state",
        "keys",
        "shape",
        "dtype",
        "nonfinite",
    ],
)
def test_invalid_checkpoints_rejected_before_prediction(snapshot, tmp_path, case):
    model = LookupMuFu(snapshot[1])
    info = provenance(snapshot[2])
    state = copy.deepcopy(model.state_dict())
    packet = {"model_state_dict": state, "training": info}
    if case == "trained":
        info["trained"] = False
    elif case == "steps":
        info["training_steps"] = 0
    elif case == "seed":
        info["training_seed"] = 0
    elif case == "selection":
        info["selection_split"] = "test"
    elif case in ("data", "split"):
        info["benchmark"][f"{case}_sha256"] = "old"
    elif case == "missing_metadata":
        del packet["training"]
    elif case == "missing_state":
        del packet["model_state_dict"]
    elif case == "keys":
        state["other"] = state.pop("anchor")
    elif case == "shape":
        state["anchor"] = torch.ones(3)
    elif case == "dtype":
        state["anchor"] = state["anchor"].double()
    else:
        state["anchor"].fill_(float("nan"))
    path = tmp_path / "bad.pt"
    torch.save(packet, path)
    with pytest.raises(ValueError):
        mb.evaluate_mufu(
            model, checkpoint=path, data_path=snapshot[0], splits=("val", "test")
        )
    assert model.anchor.item() == 0
    assert not model.calls


def test_checkpoint_loading_is_safe_strict_and_identified(
    snapshot, tmp_path, monkeypatch
):
    model = LookupMuFu(snapshot[1])
    path = tmp_path / "selected.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "training": provenance(snapshot[2])},
        path,
    )
    with torch.no_grad():
        model.anchor.fill_(2)
    load, calls = torch.load, []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    result = mb.evaluate_mufu(
        model, checkpoint=path, data_path=snapshot[0], splits=("val", "test")
    )
    assert model.anchor.item() == 0
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert (
        result.checkpoint["file_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result.checkpoint["state_sha256"] == mb._state_sha256(model)
    assert len(result.checkpoint["state_sha256"]) == 64


def test_unattested_weights_are_validation_diagnostics_only(snapshot, tmp_path):
    model = LookupMuFu(snapshot[1])
    result = mb.evaluate_mufu(model, data_path=snapshot[0])
    assert not result.training_attested
    assert all(
        not json.loads(row["params"])["training_attested"] for row in result.rows
    )
    path = tmp_path / "results.csv"
    for export in (False, True):
        with pytest.raises(ValueError, match="requires attested trained"):
            mb.evaluate_mufu(
                model,
                data_path=snapshot[0],
                splits=("val", "test"),
                write_results=export,
                results_path=path,
            )
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"splits": ("train",)},
        {"splits": ("val", "val")},
        {"splits": ()},
        {"write_results": True},
        {"checkpoint": "x", "training": {}},
    ],
)
def test_invalid_evaluation_configuration_fails_before_loading(
    snapshot, monkeypatch, kwargs
):
    monkeypatch.setattr(
        bm,
        "load_benchmark",
        lambda *args: pytest.fail("invalid request must fail before dataset loading"),
    )
    with pytest.raises(ValueError):
        mb.evaluate_mufu(LookupMuFu(snapshot[1]), **kwargs)


@pytest.fixture(scope="module")
def real_benchmark():
    return bm.load_benchmark()


def test_one_complete_export_preserves_rf_rows_and_schema(
    real_benchmark, tmp_path, monkeypatch
):
    metadata = bm.benchmark_metadata(real_benchmark)
    monkeypatch.setattr(bm, "load_benchmark", lambda path: real_benchmark)
    monkeypatch.setattr(
        mb,
        "_predict_split",
        lambda model, cohort, batch_size, device: synthetic_predictions(cohort),
    )
    path = tmp_path / "results.csv"
    path.write_bytes(results.RESULTS_PATH.read_bytes())
    before = pd.read_csv(path)
    record, calls = bm.record_results, []

    def observed_record(rows, **kwargs):
        calls.append((len(rows), kwargs["model"]))
        return record(rows, **kwargs)

    monkeypatch.setattr(bm, "record_results", observed_record)
    model = LookupMuFu(real_benchmark)
    result = mb.evaluate_mufu(
        model,
        training=provenance(metadata),
        splits=("val", "test"),
        write_results=True,
        results_path=path,
    )
    assert calls == [(50, "mufu_full")]
    after = pd.read_csv(path)
    assert after.columns.tolist() == results.COLUMNS
    assert len(after) == 96
    rf = after.loc[after["model"] != mb.MODEL_ID].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, rf, rtol=1e-14, atol=1e-14)
    for row in result.rows:
        if row["endpoint"] == "pIC50_BACE1":
            assert row["n_labeled"] == (808 if row["split"] == "val" else 810)
        if row["endpoint"] in c.TOX_COLUMNS or row["endpoint"] in c.PIC50_COLUMNS:
            original = before.loc[
                (before["endpoint"] == row["endpoint"])
                & (before["split"] == row["split"])
                & (before["metric"] == row["metric"])
            ]
            assert original["n_labeled"].iloc[0] == row["n_labeled"]
            assert original["n_positive"].iloc[0] == row["n_positive"]
    saved = path.read_bytes()
    with pytest.raises(ValueError, match="complete val and test"):
        mb.evaluate_mufu(
            model, training=provenance(metadata), write_results=True, results_path=path
        )
    assert path.read_bytes() == saved


def test_snapshot_mismatch_cannot_discard_rf_rows(snapshot, tmp_path):
    path = tmp_path / "results.csv"
    path.write_bytes(results.RESULTS_PATH.read_bytes())
    before = path.read_bytes()
    with pytest.raises(ValueError, match="discard RF results"):
        evaluate(
            snapshot, splits=("val", "test"), write_results=True, results_path=path
        )
    assert path.read_bytes() == before
