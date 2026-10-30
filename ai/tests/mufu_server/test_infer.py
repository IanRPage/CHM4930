import dataclasses

import pandas as pd
import pytest
import torch
from conftest import SEEDS, settings

from baseline.results import file_sha256
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.data.dataset import MuFuDataset
from mufu.ensemble import predict_with_ensemble_intervals
from mufu_server import infer
from mufu_server.config import COVERAGE
from mufu_server.infer import MC_SEED, InvalidSmiles, check_response, load_predictor


# infer() runs check_response, so only the server-set values are checked here
def test_infer_reports_canonical_smiles_and_metadata(predictor, ensemble_dir):
    response = predictor.infer("OCC")

    sha8 = file_sha256(ensemble_dir / "mufu_ensemble.json")[:8]
    assert response["smiles"] == "CCO"
    assert response["model_version"] == f"mufu_ensemble@{sha8}"
    assert response["n_samples"] == 4
    assert response["coverage"] == COVERAGE


def test_infer_matches_the_ensemble_prediction(predictor):
    frame = pd.DataFrame({"smiles": ["CCCN"]})
    for column in (*PIC50_COLUMNS, *ACTIVE_COLUMNS, *TOX_COLUMNS):
        frame[column] = float("nan")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(MC_SEED)
        expected = predict_with_ensemble_intervals(
            predictor.members, MuFuDataset(frame), predictor.params
        )

    response = predictor.infer("CCCN")
    egfr = response["bioactivity"]["EGFR"]
    assert egfr["pic50"] == expected["pIC50_EGFR"][0]
    assert egfr["pic50_lo"] == expected["pIC50_EGFR_lo95"][0]
    assert egfr["pic50_hi"] == expected["pIC50_EGFR_hi95"][0]
    assert egfr["p_active"] == expected["active_EGFR"][0]
    assert response["toxicity"]["SR-p53"] == expected["SR-p53"][0]


@pytest.mark.parametrize(
    ("break_files", "error", "match"),
    [
        (lambda d: (d / "mufu_seed_1.pt").unlink(), FileNotFoundError, "seed_1"),
        (lambda d: (d / "mufu_ensemble.json").unlink(), FileNotFoundError, "ensemble"),
        (
            lambda d: (d / "mufu_seed_2.pt").open("ab").write(b"0"),
            ValueError,
            "refit",
        ),
    ],
)
def test_load_predictor_rejects_missing_or_stale_files(
    ensemble_copy, encoder, break_files, error, match
):
    break_files(ensemble_copy)
    with pytest.raises(error, match=match):
        load_predictor(settings(ensemble_copy), smiles_encoder=encoder, seeds=SEEDS)


def test_fetch_version_downloads_only_a_missing_version(monkeypatch, tmp_path):
    downloads = []

    def load(*_):
        if not downloads:
            raise FileNotFoundError

    monkeypatch.setattr(infer, "load_tagged_version", load)
    monkeypatch.setattr(infer, "download_version", lambda *a: downloads.append(a))
    pinned = settings(tmp_path, hf_repo="org/mufu", model_version="v1.0.0")

    infer.fetch_version(pinned)
    infer.fetch_version(pinned)
    assert [d[:3] for d in downloads] == [("org/mufu", "v1.0.0", tmp_path)]


def test_infer_is_reproducible(predictor):
    assert predictor.infer("c1ccccc1O") == predictor.infer("c1ccccc1O")


@pytest.mark.parametrize("smiles", ["not-a-smiles", "C1CC", "[Xx]"])
def test_invalid_smiles_raise(predictor, smiles):
    with pytest.raises(InvalidSmiles):
        predictor.infer(smiles)


def test_predictor_needs_the_served_coverage(predictor):
    conformal = dataclasses.replace(predictor.params.conformal, coverages=(0.9, 0.8))
    params = dataclasses.replace(predictor.params, conformal=conformal)
    with pytest.raises(ValueError, match="coverage"):
        dataclasses.replace(predictor, params=params)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda r: r["bioactivity"]["EGFR"].update(pic50=None),
        lambda r: r["bioactivity"]["EGFR"].update(
            pic50_lo=r["bioactivity"]["EGFR"]["pic50"] + 1
        ),
        lambda r: r["bioactivity"]["EGFR"].update(pic50_lo=-400.0),
        lambda r: r["bioactivity"]["BACE1"].update(p_active=1.5),
        lambda r: r["toxicity"].update({"NR-AR": float("nan")}),
        lambda r: r["toxicity"].pop("CT_TOX"),
        lambda r: r.update(n_samples=1),
        lambda r: r.update(coverage=1.0),
        lambda r: r.update(smiles=""),
    ],
)
def test_check_response_rejects_contract_breaks(predictor, corrupt):
    response = predictor.infer("CCO")
    corrupt(response)
    with pytest.raises(ValueError, match="invalid response"):
        check_response(response)


def test_check_response_allows_null_outputs(predictor):
    response = predictor.infer("CCO")
    response["bioactivity"]["EGFR"].update(pic50=None, pic50_lo=None, pic50_hi=None)
    response["bioactivity"]["EGFR"]["p_active"] = None
    response["toxicity"]["CT_TOX"] = None
    check_response(response)
