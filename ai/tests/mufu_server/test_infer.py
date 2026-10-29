import dataclasses

import pytest

from mufu_server.config import COVERAGE
from mufu_server.infer import InvalidSmiles, check_response


# infer() runs check_response, so only the server-set values are checked here
def test_infer_reports_canonical_smiles_and_metadata(predictor):
    response = predictor.infer("OCC")

    assert response["smiles"] == "CCO"
    assert response["model_version"] == "mufu_seed_0@test"
    assert response["n_samples"] == 4
    assert response["coverage"] == COVERAGE


def test_infer_is_reproducible(predictor):
    assert predictor.infer("c1ccccc1O") == predictor.infer("c1ccccc1O")


@pytest.mark.parametrize("smiles", ["not-a-smiles", "C1CC", "[Xx]"])
def test_invalid_smiles_raise(predictor, smiles):
    with pytest.raises(InvalidSmiles):
        predictor.infer(smiles)


def test_predictor_needs_the_served_coverage(predictor):
    conformal = dataclasses.replace(predictor.conformal, coverages=(0.9, 0.8))
    with pytest.raises(ValueError, match="coverage"):
        dataclasses.replace(predictor, conformal=conformal)


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
