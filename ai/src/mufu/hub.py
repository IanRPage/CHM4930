"""
Publish and download versions of MuFu on the HuggingFace Hub.

A version is every file the served model needs from one training run, tagged
`vMAJOR.MINOR.PATCH`, with a `version.json` naming the tag and the training commit.
Each upload replaces every file in one commit, so a tag never mixes versions.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    EvalResult,
    HfApi,
    ModelCard,
    ModelCardData,
    snapshot_download,
)

from baseline.benchmark import SEEDS
from baseline.results import DEFAULT_SPLIT_ID, file_sha256, summarize_seeds
from mufu.ensemble import load_ensemble_params, member_paths, sidecar_path
from mufu.mc_dropout import load_conformal
from mufu.mc_dropout import sidecar_path as conformal_path

SMILES_ENCODER = "ibm-research/MoLFormer-XL-both-10pct"
CODE_URL = "https://github.com/IanRPage/CHM4930"
VERSION_FILE = "version.json"
TAG_PATTERN = re.compile(r"v\d+\.\d+\.\d+")

# (dataset, endpoint, metric, label), all on the test split
CARD_METRICS = (
    ("bace1", "pIC50_BACE1", "rmse", "BACE1 pIC50"),
    ("bace1", "pIC50_BACE1", "r2", "BACE1 pIC50"),
    ("egfr", "pIC50_EGFR", "rmse", "EGFR pIC50"),
    ("egfr", "pIC50_EGFR", "r2", "EGFR pIC50"),
    ("tox21", "macro_tox21", "roc_auc", "Tox21 (macro)"),
    ("clintox", "CT_TOX", "roc_auc", "ClinTox CT_TOX"),
    ("clintox", "FDA_APPROVED", "roc_auc", "ClinTox FDA_APPROVED"),
)
HF_METRIC_TYPES = {"rmse": "rmse", "r2": "r_squared", "roc_auc": "roc_auc"}
METRIC_LABELS = {"rmse": "RMSE", "r2": "R²", "roc_auc": "ROC AUC"}


@dataclass(frozen=True)
class Version:
    model_name: str
    git_commit: str
    data_sha256: str
    ensemble_sha256: str
    files: tuple[Path, ...]


def load_version(
    checkpoint_dir: Path, model_name: str = "mufu", seeds: Sequence[int] = SEEDS
) -> Version:
    """
    Raises unless the files are one committed run's complete, matching set.
    """
    paths = member_paths(checkpoint_dir, model_name, seeds)
    for path in paths:
        load_conformal(path)
    load_ensemble_params(paths, checkpoint_dir, model_name)

    runs = set()
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "model" not in ckpt or ckpt.get("calibration") is None:
            raise ValueError(f"{path.name} has no model weights or calibration")
        runs.add((ckpt.get("git_commit"), ckpt.get("data_sha256")))
    if len(runs) != 1:
        raise ValueError(f"checkpoints come from different runs: {sorted(runs)}")
    ((commit, data_sha256),) = runs
    if not commit or commit == "unknown" or commit.endswith("-dirty"):
        raise ValueError(f"checkpoints were trained on an uncommitted tree ({commit})")

    ensemble = sidecar_path(checkpoint_dir, model_name)
    files = (*paths, *(conformal_path(p) for p in paths), ensemble)
    return Version(model_name, commit, data_sha256, file_sha256(ensemble), files)


def _test_value(df: pd.DataFrame, dataset: str, endpoint: str, metric: str):
    rows = df[
        (df["dataset"] == dataset)
        & (df["endpoint"] == endpoint)
        & (df["split"] == "test")
        & (df["metric"] == metric)
    ]
    return rows.iloc[0] if len(rows) else None


def model_card(
    version: Version,
    tag: str,
    results: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    repo_id: str,
) -> ModelCard:
    """
    `results` as `load_results()` returns it; `ensemble_metrics` as `fit_ensemble.py`
    writes it.
    """
    name = version.model_name
    fitted_on = set(ensemble_metrics.get("ensemble_sha256", []))
    if fitted_on != {version.ensemble_sha256}:
        raise ValueError(
            "ensemble metrics aren't from this ensemble file; rerun fit_ensemble.py"
        )
    results = results[results["split_id"] == DEFAULT_SPLIT_ID]
    run = results[
        (results["model"] == name)
        & (results["git_commit"] == version.git_commit)
        & (results["data_sha256"] == version.data_sha256)
    ]
    if run.empty:
        raise ValueError(f"results.csv has no {name} rows for {version.git_commit}")
    single = summarize_seeds(run)
    baseline = summarize_seeds(
        results[
            results["model"].str.startswith("rf_")
            & (results["data_sha256"] == version.data_sha256)
        ]
    )

    eval_results = []
    table = [
        "| Task | Metric | Ensemble | Single model (mean ± std) | RF baseline |",
        "| --- | --- | --- | --- | --- |",
    ]
    for dataset, endpoint, metric, label in CARD_METRICS:
        ens = _test_value(ensemble_metrics, dataset, endpoint, metric)
        one = _test_value(single, dataset, endpoint, metric)
        rf = _test_value(baseline, dataset, endpoint, metric)
        if ens is None or one is None:
            raise ValueError(f"no test {endpoint} {metric} for {tag}")
        kind = "classification" if metric == "roc_auc" else "regression"
        eval_results.append(
            EvalResult(
                task_type=f"tabular-{kind}",
                task_name=label,
                dataset_type=dataset,
                dataset_config=endpoint,
                dataset_name=f"{label.split()[0]} (CHM4930 test split)",
                dataset_split="test",
                metric_type=HF_METRIC_TYPES[metric],
                metric_name=f"Test {METRIC_LABELS[metric]}",
                metric_value=round(float(ens["value"]), 4),
            )
        )
        rf_value = "—" if rf is None else f"{rf['value']:.3f}"
        table.append(
            f"| {label} | {METRIC_LABELS[metric]} | {ens['value']:.3f} "
            f"| {one['value']:.3f} ± {one['std']:.3f} | {rf_value} |"
        )

    data = ModelCardData(
        model_name=name,
        license="apache-2.0",
        library_name="pytorch",
        tags=["chemistry", "molecular-property-prediction", "toxicity", "ensemble"],
        base_model=SMILES_ENCODER,
        eval_results=eval_results,
    )
    n_seeds = run["seed"].nunique()
    metrics_table = "\n".join(table)
    body = f"""
# {name} {tag}

MuFu, from [CHM4930]({CODE_URL}), fuses a frozen MoLFormer SMILES embedding, an ECFP4
fingerprint and a GINE graph encoder. For one SMILES it predicts EGFR and BACE1 pIC50
(with a conformal interval) and P(active), the 12 Tox21 assays and the 2 ClinTox tasks.

This is a {n_seeds}-seed ensemble from commit `{version.git_commit}`, trained on data
with sha256 `{version.data_sha256}`. pIC50 is the members' mean, and probabilities are
averaged after each member's own calibration. The interval is
`μ ± q · max(σ, floor)`, where `σ²` adds the mean MC-Dropout variance and the
variance between members, and `q` is fitted on validation.

The files only work as a set: each `_mc_dropout.json` is fitted to its checkpoint, and
`{name}_ensemble.json` to exactly these checkpoints. Don't mix files across tags.

## Test metrics

{metrics_table}

Single-model numbers are over the {n_seeds} seeds; RF is a random forest on ECFP4.

## Usage

With [the code]({CODE_URL}) and `ai/src` on the path (MolFormer is downloaded from
`{SMILES_ENCODER}` on first use):

```python
from pathlib import Path

from huggingface_hub import snapshot_download

from encoders.smiles_encoder import SmilesEncoder
from mufu.ensemble import load_ensemble_params, load_members, member_paths

path = Path(snapshot_download("{repo_id}", revision="{tag}"))
paths = member_paths(path, "{name}", range({n_seeds}))
params = load_ensemble_params(paths, path, "{name}")
members = load_members(paths, SmilesEncoder(freeze=True))
```

Predict with `mufu.ensemble.predict_with_ensemble_intervals`, or serve it with
`MUFU_HF_REPO={repo_id} MUFU_MODEL_VERSION={tag}` (see `ai/README.md`). For a
single seed, use `mufu.experiment.load_mufu_checkpoint`.
"""
    return ModelCard(f"---\n{data.to_yaml()}\n---\n{body}")


def upload_version(
    api: HfApi, repo_id: str, version: Version, tag: str, card: ModelCard
) -> None:
    if not TAG_PATTERN.fullmatch(tag):
        raise ValueError(f"tag {tag!r} isn't vMAJOR.MINOR.PATCH")
    api.create_repo(repo_id, exist_ok=True)
    if tag in {ref.name for ref in api.list_repo_refs(repo_id).tags}:
        raise ValueError(f"{repo_id} already has {tag}")

    info = {"tag": tag, "git_commit": version.git_commit}
    keep = {
        *(p.name for p in version.files),
        "README.md",
        VERSION_FILE,
        ".gitattributes",
    }
    stale = sorted(set(api.list_repo_files(repo_id)) - keep)
    operations = [
        *(CommitOperationAdd(p.name, p) for p in version.files),
        CommitOperationAdd(VERSION_FILE, (json.dumps(info) + "\n").encode()),
        CommitOperationAdd("README.md", str(card).encode()),
        *(CommitOperationDelete(name) for name in stale),
    ]
    commit = api.create_commit(repo_id, operations, commit_message=f"Publish {tag}")
    api.create_tag(repo_id, tag=tag, revision=commit.oid)


def load_tagged_version(
    directory: Path, tag: str, model_name: str = "mufu", seeds: Sequence[int] = SEEDS
) -> Version:
    """
    `load_version`, and checks `version.json` names `tag` and the checkpoints' commit.
    """
    version = load_version(directory, model_name, seeds)
    info = json.loads((directory / VERSION_FILE).read_text())
    if info != {"tag": tag, "git_commit": version.git_commit}:
        raise ValueError(f"{directory} holds {info}, not {tag} of {version.git_commit}")
    return version


def download_version(
    repo_id: str,
    tag: str,
    dest: Path,
    model_name: str = "mufu",
    seeds: Sequence[int] = SEEDS,
) -> Version:
    snapshot_download(repo_id, revision=tag, local_dir=dest)
    return load_tagged_version(dest, tag, model_name, seeds)
