"""
Publish and download versions of MuFu on the HuggingFace Hub.

A version is every file the served model needs from one training run, tagged
`vMAJOR.MINOR.PATCH`, with a `version.json` naming the tag and code provenance.
Each upload replaces every file in one commit, so a tag never mixes versions.
"""

import json
import re
import shutil
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

from baseline.benchmark import PROMOTED_PATH, RUNS_DIR, SEEDS
from baseline.results import (
    DEFAULT_SPLIT_ID,
    RESULTS_PATH,
    file_sha256,
    runs_path,
    summarize_seeds,
    verify_code_identity,
)
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
    base_commit: str
    ai_tree: str
    patch_sha256: str
    data_sha256: str
    ensemble_sha256: str
    files: tuple[Path, ...]
    run_id: str = ""
    extra_files: tuple[tuple[str, Path], ...] = ()

    @property
    def git_commit(self) -> str:
        return self.base_commit


def _find_run_json(checkpoint_dir: Path) -> Path | None:
    for candidate in (checkpoint_dir / "run.json", checkpoint_dir.parent / "run.json"):
        if candidate.exists():
            return candidate
    return None


def load_version(
    checkpoint_dir: Path,
    model_name: str = "mufu",
    seeds: Sequence[int] = SEEDS,
    *,
    verify_git: bool = False,
) -> Version:
    """
    Raises unless the files are one committed run's complete, matching set.
    """
    if (checkpoint_dir / "checkpoints").is_dir() and not (
        checkpoint_dir / f"{model_name}_seed_{seeds[0]}.pt"
    ).exists():
        checkpoint_dir = checkpoint_dir / "checkpoints"

    paths = member_paths(checkpoint_dir, model_name, seeds)
    for path in paths:
        load_conformal(path)
    load_ensemble_params(paths, checkpoint_dir, model_name)

    runs = set()
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if "model" not in ckpt or ckpt.get("calibration") is None:
            raise ValueError(f"{path.name} has no model weights or calibration")
        ckpt_base = ckpt.get("base_commit") or ckpt.get("git_commit")
        ckpt_tree = ckpt.get("ai_tree", "")
        ckpt_patch = ckpt.get("patch_sha256", "")
        runs.add((ckpt_base, ckpt_tree, ckpt_patch, ckpt.get("data_sha256")))
    if len(runs) != 1:
        raise ValueError(f"checkpoints come from different runs: {sorted(runs)}")
    ((base_commit, ai_tree, patch_sha256, data_sha256),) = runs
    if (
        not base_commit
        or base_commit == "unknown"
        or str(base_commit).endswith("-dirty")
    ):
        raise ValueError(
            f"checkpoints were trained on an uncommitted tree ({base_commit})"
        )

    run_id = ""
    extra_files: list[tuple[str, Path]] = []
    run_json_path = _find_run_json(checkpoint_dir)
    patch_path: Path | None = None
    if run_json_path is not None:
        run_meta = json.loads(run_json_path.read_text())
        if run_meta.get("status", "complete") != "complete":
            raise ValueError(
                f"run {run_meta.get('run_id')} is not complete (status={run_meta.get('status')!r})"
            )
        if run_meta.get("data_sha256") != data_sha256:
            raise ValueError("run.json data_sha256 does not match checkpoints")
        meta_base = str(run_meta.get("base_commit", ""))
        if not meta_base.startswith(str(base_commit)):
            raise ValueError(
                f"run.json base_commit ({meta_base}) does not match checkpoint ({base_commit})"
            )
        base_commit = meta_base
        ai_tree = str(run_meta.get("ai_tree", ai_tree))
        patch_sha256 = str(run_meta.get("patch_sha256", patch_sha256))
        run_id = str(run_meta.get("run_id", run_json_path.parent.name))
        run_root = run_json_path.parent
        extra_files.append(("run.json", run_json_path))
        candidate_patch = run_root / "patch.diff"
        if candidate_patch.exists():
            patch_path = candidate_patch
            if patch_sha256:
                extra_files.append(("patch.diff", candidate_patch))
        for metric_csv in ("val_metrics.csv", "test_metrics.csv"):
            m_path = run_root / metric_csv
            if m_path.exists():
                extra_files.append((metric_csv, m_path))
        preds_dir = run_root / "predictions"
        if preds_dir.is_dir():
            for p in sorted(preds_dir.glob("*.csv")):
                extra_files.append((f"predictions/{p.name}", p))

        if (run_root / "baselines").is_dir():
            rf_candidates = sorted((run_root / "baselines").glob("rf_*"))
            promoted_rf_ids = None
        else:
            rf_candidates = sorted(run_root.parent.glob("rf_*"))
            promoted_rf_ids = None
            if (
                run_root.parent.resolve() == RUNS_DIR.resolve()
                and PROMOTED_PATH.exists()
            ):
                raw_promoted = json.loads(PROMOTED_PATH.read_text())
                promoted_rf_ids = {
                    str(rid)
                    for model_key, splits in raw_promoted.items()
                    if model_key.startswith("rf_") and isinstance(splits, dict)
                    for rid in splits.values()
                }

        for rf_dir in rf_candidates:
            if not rf_dir.is_dir():
                continue
            if promoted_rf_ids is not None and rf_dir.name not in promoted_rf_ids:
                continue
            rf_json = rf_dir / "run.json"
            if not rf_json.exists():
                continue
            rf_meta = json.loads(rf_json.read_text())
            if (
                rf_meta.get("status") != "complete"
                or rf_meta.get("data_sha256") != data_sha256
            ):
                continue
            prefix = f"baselines/{rf_dir.name}"
            extra_files.append((f"{prefix}/run.json", rf_json))
            for metric_csv in ("val_metrics.csv", "test_metrics.csv"):
                m_path = rf_dir / metric_csv
                if m_path.exists():
                    extra_files.append((f"{prefix}/{metric_csv}", m_path))
            rf_preds = rf_dir / "predictions"
            if rf_preds.is_dir():
                for p in sorted(rf_preds.glob("*.csv")):
                    extra_files.append((f"{prefix}/predictions/{p.name}", p))

    if verify_git:
        verify_code_identity(
            str(base_commit),
            str(ai_tree),
            str(patch_sha256),
            patch_path=patch_path,
        )

    ensemble = sidecar_path(checkpoint_dir, model_name)
    files = (*paths, *(conformal_path(p) for p in paths), ensemble)
    return Version(
        model_name=model_name,
        base_commit=str(base_commit),
        ai_tree=str(ai_tree),
        patch_sha256=str(patch_sha256),
        data_sha256=str(data_sha256),
        ensemble_sha256=file_sha256(ensemble),
        files=files,
        run_id=run_id,
        extra_files=tuple(extra_files),
    )


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
    `results` as `load_results()` returns it; `ensemble_metrics` as `score_mufu_run_test`
    writes it. Selects single-model rows by `version.run_id` when present.
    """
    name = version.model_name
    fitted_on = set(ensemble_metrics.get("ensemble_sha256", []))
    if fitted_on != {version.ensemble_sha256}:
        raise ValueError(
            "ensemble metrics aren't from this ensemble file; rerun fit_ensemble.py"
        )
    results = results[results["split_id"] == DEFAULT_SPLIT_ID]
    if version.run_id and "run_id" in results.columns:
        run = results[
            (results["model"] == name)
            & (results["run_id"] == version.run_id)
            & (results["data_sha256"] == version.data_sha256)
        ]
    else:
        commit_col = "base_commit" if "base_commit" in results.columns else "git_commit"
        run = results[
            (results["model"] == name)
            & (results[commit_col].astype(str).str.startswith(str(version.base_commit)))
            & (results["data_sha256"] == version.data_sha256)
        ]
    if run.empty:
        target_desc = version.run_id or version.base_commit
        raise ValueError(f"results.csv has no {name} rows for {target_desc}")
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
    tree_note = f" (`ai_tree` `{version.ai_tree[:12]}`)" if version.ai_tree else ""
    body = f"""
# {name} {tag}

MuFu, from [CHM4930]({CODE_URL}), fuses a frozen MoLFormer SMILES embedding, an ECFP4
fingerprint and a GINE graph encoder. For one SMILES it predicts EGFR and BACE1 pIC50
(with a conformal interval) and P(active), the 12 Tox21 assays and the 2 ClinTox tasks.

This is a {n_seeds}-seed ensemble from commit `{version.base_commit}`{tree_note}, trained on data
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

    info = {
        "tag": tag,
        "run_id": version.run_id,
        "base_commit": version.base_commit,
        "ai_tree": version.ai_tree,
        "patch_sha256": version.patch_sha256,
    }
    keep = {
        *(p.name for p in version.files),
        *(rel for rel, _ in version.extra_files),
        "README.md",
        VERSION_FILE,
        ".gitattributes",
    }
    stale = sorted(set(api.list_repo_files(repo_id)) - keep)
    operations = [
        *(CommitOperationAdd(p.name, p) for p in version.files),
        *(CommitOperationAdd(rel, p) for rel, p in version.extra_files),
        CommitOperationAdd(VERSION_FILE, (json.dumps(info) + "\n").encode()),
        CommitOperationAdd("README.md", str(card).encode()),
        *(CommitOperationDelete(name) for name in stale),
    ]
    commit = api.create_commit(repo_id, operations, commit_message=f"Publish {tag}")
    api.create_tag(repo_id, tag=tag, revision=commit.oid)


def load_tagged_version(
    directory: Path, tag: str, model_name: str = "mufu", seeds: Sequence[int] = SEEDS
) -> Version:
    """`load_version`, plus check `version.json` matches `tag` and checkpoint provenance."""
    version = load_version(directory, model_name, seeds)
    info = json.loads((directory / VERSION_FILE).read_text())
    if info.get("tag") != tag:
        raise ValueError(
            f"{directory} holds {info}, not {tag} of {version.base_commit}"
        )
    if "base_commit" in info:
        if (
            info.get("base_commit") != version.base_commit
            or info.get("ai_tree", "") != version.ai_tree
            or info.get("patch_sha256", "") != version.patch_sha256
        ):
            raise ValueError(
                f"{directory} holds {info}, not {tag} of {version.base_commit}"
            )
    elif "git_commit" in info:
        legacy_commit = str(info.get("git_commit") or "")
        if not legacy_commit or not version.base_commit.startswith(legacy_commit):
            raise ValueError(
                f"{directory} holds {info}, not {tag} of {version.base_commit}"
            )
    else:
        raise ValueError(
            f"{directory} version.json is missing commit provenance: {info}"
        )
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


def populate_run_from_download(
    downloaded_dir: Path,
    tag: str,
    *,
    runs_dir: Path = RUNS_DIR,
    results_path: Path = RESULTS_PATH,
    model_name: str = "mufu",
    seeds: Sequence[int] = SEEDS,
) -> Path:
    """Unpack a downloaded HF snapshot into `runs/<run_id>/` (and bundled `runs/rf_*/`)."""
    version = load_tagged_version(downloaded_dir, tag, model_name, seeds)
    dl_run_json = downloaded_dir / "run.json"
    if dl_run_json.exists():
        meta = json.loads(dl_run_json.read_text())
        run_id = str(meta["run_id"])
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dl_run_json, run_dir / "run.json")
        if (downloaded_dir / "patch.diff").exists():
            shutil.copy2(downloaded_dir / "patch.diff", run_dir / "patch.diff")
        for metric_csv in ("val_metrics.csv", "test_metrics.csv"):
            src_m = downloaded_dir / metric_csv
            dst_m = run_dir / metric_csv
            if src_m.exists() and not dst_m.exists():
                shutil.copy2(src_m, dst_m)
    else:
        r_path = runs_path(results_path)
        if not r_path.exists():
            raise FileNotFoundError(
                f"downloaded {tag} has no run.json and {r_path} is missing"
            )
        runs_df = pd.read_csv(r_path, dtype={"patch_sha256": str})
        runs_df["patch_sha256"] = runs_df["patch_sha256"].fillna("").astype(str)
        match = runs_df[
            (runs_df["model"] == model_name)
            & (runs_df["split_id"] == DEFAULT_SPLIT_ID)
            & (runs_df["data_sha256"] == version.data_sha256)
            & (
                runs_df["base_commit"]
                .astype(str)
                .str.startswith(str(version.base_commit))
            )
        ]
        if match.empty:
            raise ValueError(
                f"could not find a tracked {model_name} row in {r_path} matching "
                f"{version.base_commit}"
            )
        row = match.iloc[0]
        run_id = str(row["run_id"])
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": run_id,
            "model": str(row["model"]),
            "encoder": str(row["encoder"]),
            "split_id": str(row["split_id"]),
            "seeds": [int(s) for s in seeds],
            "completed_seeds": [int(s) for s in seeds],
            "status": "complete",
            "params": str(row["params"]),
            "data_sha256": str(row["data_sha256"]),
            "split_sha256": str(row["split_sha256"]),
            "base_commit": str(row["base_commit"]),
            "ai_tree": str(row["ai_tree"]),
            "patch_sha256": str(row["patch_sha256"]),
            "date": str(row["date"]),
        }
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for src in version.files:
        dst = ckpt_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    dl_preds = downloaded_dir / "predictions"
    if dl_preds.is_dir():
        dst_preds = run_dir / "predictions"
        dst_preds.mkdir(parents=True, exist_ok=True)
        for src in sorted(dl_preds.glob("*.csv")):
            dst = dst_preds / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    dl_baselines = downloaded_dir / "baselines"
    if dl_baselines.is_dir():
        for rf_src in sorted(dl_baselines.glob("rf_*")):
            if not rf_src.is_dir():
                continue
            rf_dst = runs_dir / rf_src.name
            rf_dst.mkdir(parents=True, exist_ok=True)
            for fname in ("run.json", "val_metrics.csv", "test_metrics.csv"):
                s = rf_src / fname
                d = rf_dst / fname
                if s.exists() and not d.exists():
                    shutil.copy2(s, d)
            rf_preds_src = rf_src / "predictions"
            if rf_preds_src.is_dir():
                rf_preds_dst = rf_dst / "predictions"
                rf_preds_dst.mkdir(parents=True, exist_ok=True)
                for src in sorted(rf_preds_src.glob("*.csv")):
                    dst = rf_preds_dst / src.name
                    if not dst.exists():
                        shutil.copy2(src, dst)

    return run_dir
