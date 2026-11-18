import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

AI_DIR = Path(__file__).resolve().parents[1]
TARGETS = ("BACE1", "EGFR")
ACTIVE_THRESHOLD = 6.0
TOX21 = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]
CLINTOX = ("FDA_APPROVED", "CT_TOX")
SOURCES = ("bace1", "egfr", "tox21", "clintox")
WIDTHS = {"pic50": (0.25, 0.5, 1.0), "molecularWeight": (25, 50, 100)}


def read_snapshot(raw: bytes) -> list[dict]:
    frame = pd.read_csv(io.BytesIO(raw))
    rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    records = [molecule_record(row) for row in rows]
    return sorted(records, key=lambda record: record["id"])


def known(values) -> list:
    return [value for value in values if value is not None]


def molecule_record(row: dict) -> dict:
    smiles = row["smiles"]
    mol = Chem.MolFromSmiles(smiles)
    expanded = Chem.AddHs(mol)
    atoms = list(expanded.GetAtoms())
    weight = float(Descriptors.MolWt(mol))
    binary = {
        key: None if row[key] is None else int(row[key]) for key in (*TOX21, *CLINTOX)
    }
    pic50 = {target: row[f"pIC50_{target}"] for target in TARGETS}
    activity = known(pic50.values())
    risk = known(binary[key] for key in (*TOX21, "CT_TOX"))
    elements = Counter(atom.GetSymbol() for atom in atoms if atom.GetAtomicNum())
    return {
        "id": "mol_" + hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
        "smiles": smiles,
        "sources": [source for source in SOURCES if row[f"in_{source}"]],
        "bioactivity": {
            target: {
                "pic50": value,
                "active": None if value is None else value >= ACTIVE_THRESHOLD,
            }
            for target, value in pic50.items()
        },
        "tox21": {key: binary[key] for key in TOX21},
        "clintox": {key: binary[key] for key in CLINTOX},
        "molecularWeight": weight,
        "elementCounts": dict(elements),
        "atomCount": len(atoms),
        "heavyAtomCount": sum(atom.GetAtomicNum() > 1 for atom in atoms),
        "wildcardAtomCount": sum(atom.GetAtomicNum() == 0 for atom in atoms),
        "activity": ("active" if max(activity) >= ACTIVE_THRESHOLD else "inactive")
        if activity
        else "unknown",
        "risk": ("positive" if any(risk) else "negative") if risk else "unknown",
        "riskLabelsObserved": len(risk),
    }


def statistics(values: list) -> dict:
    keys = ("min", "p25", "median", "p75", "max")
    quantiles = (
        np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist() if values else [None] * 5
    )
    return {
        "count": len(values),
        "mean": fmean(values) if values else None,
        **dict(zip(keys, quantiles)),
    }


def histograms(values: list, bounds: tuple, widths: tuple) -> list[dict]:
    result = []
    for width in widths:
        first, last = math.floor(bounds[0] / width), math.floor(bounds[1] / width) + 1
        counts = Counter(math.floor(value / width) for value in values)
        result.append(
            {
                "binWidth": width,
                "edges": [index * width for index in range(first, last + 1)],
                "counts": [counts[index] for index in range(first, last)],
            }
        )
    return result


def binary_summary(labels: list) -> dict:
    observed = known(labels)
    positive = sum(observed)
    return {
        "positive": positive,
        "negative": len(observed) - positive,
        "labeled": len(observed),
        "missing": len(labels) - len(observed),
        "positiveRate": positive / len(observed) if observed else None,
    }


def target_values(records: list[dict], target: str) -> list:
    return known(record["bioactivity"][target]["pic50"] for record in records)


def summarize(records: list[dict], bounds: dict) -> dict:
    total, bioactivity = len(records), {}
    for target in TARGETS:
        values = target_values(records, target)
        active = sum(value >= ACTIVE_THRESHOLD for value in values)
        bioactivity[target] = {
            "labeled": len(values),
            "missing": total - len(values),
            "active": active,
            "inactive": len(values) - active,
            "statistics": statistics(values),
            "histograms": histograms(values, bounds[target], WIDTHS["pic50"]),
        }
    totals, occurrences = Counter(), Counter()
    for record in records:
        totals.update(record["elementCounts"])
        occurrences.update(record["elementCounts"].keys())
    weights = [record["molecularWeight"] for record in records]
    return {
        "counts": {
            "total": total,
            "withBioactivityLabels": sum(r["activity"] != "unknown" for r in records),
            "withToxicityLabels": sum(
                bool(known([*r["tox21"].values(), *r["clintox"].values()]))
                for r in records
            ),
        },
        "bioactivity": bioactivity,
        "tox21": {
            assay: {
                "family": assay.split("-")[0],
                **binary_summary([r["tox21"][assay] for r in records]),
            }
            for assay in TOX21
        },
        "clintox": {
            key: binary_summary([r["clintox"][key] for r in records]) for key in CLINTOX
        },
        "elements": {
            element: {
                "moleculesContaining": occurrences[element],
                "percentageOfMolecules": 100 * occurrences[element] / total,
                "totalAtoms": totals[element],
                "averageAtomsPerMolecule": totals[element] / total,
            }
            for element in sorted(totals)
        },
        **{
            key + "s": statistics([r[key] for r in records])
            for key in ("atomCount", "heavyAtomCount", "wildcardAtomCount")
        },
        "moleculesWithWildcardAtoms": sum(r["wildcardAtomCount"] > 0 for r in records),
        "molecularWeight": {
            "statistics": statistics(weights),
            "histograms": histograms(
                weights, bounds["molecularWeight"], WIDTHS["molecularWeight"]
            ),
        },
    }


def canonical_json(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def build_artifact(raw: bytes) -> dict:
    records = read_snapshot(raw)
    populations = {
        target: [*target_values(records, target), ACTIVE_THRESHOLD]
        for target in TARGETS
    }
    populations["molecularWeight"] = [r["molecularWeight"] for r in records]
    bounds = {key: (min(values), max(values)) for key, values in populations.items()}
    groups = {"all": records}
    for source in SOURCES:
        groups[source] = [r for r in records if source in r["sources"]]
    artifact = {
        "schemaVersion": 1,
        "metadata": {
            "sourceFile": "combined.csv",
            "sourceSha256": hashlib.sha256(raw).hexdigest(),
            "sourceRowCount": len(records),
            "rdkitVersion": rdBase.rdkitVersion,
            "activeThreshold": ACTIVE_THRESHOLD,
            "sources": list(SOURCES),
            "targets": list(TARGETS),
            "tox21Assays": [
                {"name": assay, "family": assay.split("-")[0]} for assay in TOX21
            ],
            "clintoxLabels": {
                "FDA_APPROVED": {"0": "not approved", "1": "FDA approved"},
                "CT_TOX": {
                    "0": "not labeled clinically toxic",
                    "1": "clinical toxicity",
                },
            },
            "clintoxClassImbalanceNote": (
                "ClinTox labels are imbalanced; show class counts and rates over observed labels."
            ),
            "histogramBinWidths": WIDTHS,
            "riskLabelCount": len(TOX21) + 1,
        },
        "summary": {name: summarize(group, bounds) for name, group in groups.items()},
        "molecules": records,
    }
    artifact["datasetVersion"] = (
        "sha256:" + hashlib.sha256(canonical_json(artifact)).hexdigest()
    )
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=AI_DIR / "data/baseline/combined.csv"
    )
    parser.add_argument(
        "--output", type=Path, default=AI_DIR / "runs/dataset-explorer.v1.json"
    )
    args = parser.parse_args()
    artifact = build_artifact(args.input.read_bytes())
    payload = canonical_json(artifact) + b"\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    size = args.output.write_bytes(payload)
    print(
        f"Wrote {args.output}: {len(artifact['molecules']):,} compounds, {size:,} bytes"
    )
    print(f"Dataset version: {artifact['datasetVersion']}")
    print(f"Counts: {artifact['summary']['all']['counts']}")


if __name__ == "__main__":
    main()
