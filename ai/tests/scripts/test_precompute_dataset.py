import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

precompute = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts" / "precompute_dataset.py")
)


@pytest.fixture
def artifact():
    empty = dict.fromkeys(
        ["pIC50_BACE1", "pIC50_EGFR", *precompute["TOX21"], *precompute["CLINTOX"]]
    )
    empty.update({f"in_{source}": False for source in precompute["SOURCES"]})
    rows = [
        {
            **empty,
            "smiles": "CCO",
            "pIC50_BACE1": 6.0,
            "NR-AR": 1,
            "in_bace1": True,
            "in_tox21": True,
        },
        {
            **empty,
            "smiles": "CC",
            "pIC50_BACE1": 5.9,
            "NR-AR": 0,
            "in_bace1": True,
            "in_tox21": True,
        },
        {**empty, "smiles": "N", "FDA_APPROVED": 1, "in_clintox": True},
    ]
    raw = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    return json.loads(precompute["canonical_json"](precompute["build_artifact"](raw)))


def test_molecule_properties_threshold_and_missing_labels(artifact):
    molecules = {record["smiles"]: record for record in artifact["molecules"]}
    ethanol = molecules["CCO"]
    assert ethanol["elementCounts"] == {"C": 2, "H": 6, "O": 1}
    assert ethanol["atomCount"] == 9
    assert ethanol["heavyAtomCount"] == 3
    assert ethanol["molecularWeight"] == pytest.approx(46.069, abs=0.001)
    assert ethanol["bioactivity"]["BACE1"] == {"pic50": 6.0, "active": True}
    assert ethanol["bioactivity"]["EGFR"] == {"pic50": None, "active": None}
    assert ethanol["activity"] == "active"
    assert ethanol["risk"] == "positive"
    assert molecules["CC"]["activity"] == "inactive"
    assert molecules["CC"]["risk"] == "negative"
    assert molecules["N"]["activity"] == "unknown"
    assert molecules["N"]["risk"] == "unknown"
    assert molecules["N"]["riskLabelsObserved"] == 0


def test_summary_counts_rates_and_distributions(artifact):
    summary = artifact["summary"]["all"]
    assert summary["counts"] == {
        "total": 3,
        "withBioactivityLabels": 2,
        "withToxicityLabels": 3,
    }
    assert summary["tox21"]["NR-AR"] == {
        "family": "NR",
        "positive": 1,
        "negative": 1,
        "labeled": 2,
        "missing": 1,
        "positiveRate": 0.5,
    }
    assert summary["clintox"]["FDA_APPROVED"]["positive"] == 1
    assert summary["clintox"]["CT_TOX"]["positiveRate"] is None
    target = summary["bioactivity"]["BACE1"]
    assert (target["active"], target["inactive"], target["missing"]) == (1, 1, 1)
    assert target["histograms"][0] == {
        "binWidth": 0.25,
        "edges": [5.75, 6.0, 6.25],
        "counts": [1, 1],
    }
    for histogram in summary["molecularWeight"]["histograms"]:
        assert sum(histogram["counts"]) == 3
    assert summary["elements"]["O"]["percentageOfMolecules"] == pytest.approx(100 / 3)
    assert summary["elements"]["O"]["averageAtomsPerMolecule"] == pytest.approx(1 / 3)
    assert artifact["summary"]["bace1"]["counts"]["total"] == 2
    assert artifact["summary"]["tox21"]["counts"]["total"] == 2
    assert (
        artifact["summary"]["egfr"]["bioactivity"]["BACE1"]["statistics"]["count"] == 0
    )
