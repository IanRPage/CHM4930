"""
Read, write, and summarize benchmark metrics in `ai/data/baseline/results.csv`.

Each row in `results.csv` is one seed's measurement on one scaffold split (`split_id`).
What is constant per promoted `(model, encoder, split_id)` run (run_id, n_seeds, params,
data/split hashes, code provenance, date) is stored once per run in `results_runs.csv`
next to it, and `load_results()` joins the two.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.combined_loader import COMBINED_CSV_PATH

log = logging.getLogger(__name__)

AI_ROOT = Path(__file__).resolve().parents[2]
CODE_SUBTREES = ("src", "drivers", "environment.yml")
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

RESULTS_PATH = COMBINED_CSV_PATH.with_name("results.csv")
DEFAULT_SPLIT_ID = "default"
COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "seed",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_labeled",
    "n_positive",
    "n_tasks",
    "dropped_tasks",
    "run_id",
    "n_seeds",
    "params",
    "data_sha256",
    "split_sha256",
    "base_commit",
    "ai_tree",
    "patch_sha256",
    "date",
]
RUN_KEYS = ["model", "encoder", "split_id"]
RUN_INFO = [
    "run_id",
    "n_seeds",
    "params",
    "data_sha256",
    "split_sha256",
    "base_commit",
    "ai_tree",
    "patch_sha256",
    "date",
]
RUN_COLUMNS = RUN_KEYS + RUN_INFO
METRIC_COLUMNS = [c for c in COLUMNS if c not in RUN_INFO]
SORT_KEYS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "seed",
    "metric",
]
SUMMARY_GROUP_KEYS = [
    c
    for c in COLUMNS
    if c
    not in (
        "seed",
        "value",
        "ci_low",
        "ci_high",
        "run_id",
        "n_seeds",
        "base_commit",
        "ai_tree",
        "patch_sha256",
        "date",
    )
]
SUMMARY_COLUMNS = [
    "model",
    "encoder",
    "dataset",
    "endpoint",
    "split_id",
    "split",
    "metric",
    "value",
    "std",
    "ci_low",
    "ci_high",
    "n_seeds",
    "n_labeled",
    "n_positive",
    "n_tasks",
    "dropped_tasks",
    "params",
    "data_sha256",
    "split_sha256",
]
SPLIT_GROUP_KEYS = ["model", "encoder", "dataset", "endpoint", "split", "metric"]
SPLIT_SUMMARY_COLUMNS = [
    *SPLIT_GROUP_KEYS,
    "value",
    "split_std",
    "seed_std",
    "n_splits",
    "n_seeds",
    "data_sha256",
]
LOWER_IS_BETTER = frozenset({"rmse"})


@dataclass(frozen=True)
class CodeIdentity:
    base_commit: str
    ai_tree: str
    patch: str
    patch_sha256: str


def runs_path(path: Path = RESULTS_PATH) -> Path:
    return path.with_name(path.stem + "_runs.csv")


def load_results(path: Path = RESULTS_PATH) -> pd.DataFrame:
    """
    Metric rows joined with their promoted run's metadata from `results_runs.csv`.
    """
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    metrics = pd.read_csv(path, float_precision="round_trip")
    runs = pd.read_csv(runs_path(path), dtype={"patch_sha256": str})
    runs["patch_sha256"] = runs["patch_sha256"].fillna("").astype(str)
    runs["n_seeds"] = runs["n_seeds"].astype("Int64")
    metrics["seed"] = metrics["seed"].astype("Int64")
    return metrics.merge(runs, on=RUN_KEYS, how="left", validate="many_to_one")[COLUMNS]


def write_results(out: pd.DataFrame, path: Path = RESULTS_PATH) -> None:
    """
    Split joined rows into the metric and run files. Both are written to temp files and
    the old files backed up before either is replaced. If a replace fails, the old pair
    is restored, so the two files never disagree.
    """
    out = out.reindex(columns=COLUMNS).sort_values(SORT_KEYS, ignore_index=True)
    out["seed"] = out["seed"].astype("Int64")
    out["n_seeds"] = out["n_seeds"].astype("Int64")
    out["patch_sha256"] = out["patch_sha256"].fillna("").astype(str)
    runs = out[RUN_COLUMNS].drop_duplicates()
    if runs.duplicated(RUN_KEYS).any():
        raise ValueError(
            "a run has more than one metadata record for the same (model, encoder, split_id)"
        )
    pairs = [
        (out[METRIC_COLUMNS], path),
        (runs.sort_values(RUN_KEYS, ignore_index=True), runs_path(path)),
    ]
    targets = [target for _, target in pairs]
    tmps = [t.with_name(t.name + ".tmp") for t in targets]
    baks = [t.with_name(t.name + ".bak") for t in targets]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for (df, _), tmp in zip(pairs, tmps):
            df.to_csv(tmp, index=False, lineterminator="\n")
        for target, bak in zip(targets, baks):
            if target.exists():
                shutil.copy2(target, bak)
        try:
            for target, tmp in zip(targets, tmps):
                tmp.replace(target)
        except BaseException:
            for target, bak in zip(targets, baks):
                if bak.exists():
                    bak.replace(target)
                else:
                    target.unlink(missing_ok=True)
            raise
    finally:
        for leftover in (*tmps, *baks):
            leftover.unlink(missing_ok=True)
    log.info("wrote %d rows to %s", len(out), path)


def params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def file_sha256(path: Path) -> str:
    """Hash a dataset CSV using LF bytes, including in an existing CRLF checkout.

    This matches the committed snapshot and historical results without rewriting
    the file. All bytes other than CRLF line endings remain significant.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git_stdout(
    args: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> str:
    return subprocess.run(
        ["git", *args],
        input=input_text,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _git_repo_and_prefix(ai_dir: Path = AI_ROOT) -> tuple[Path, str, list[str]]:
    top = Path(
        _git_stdout(["rev-parse", "--show-toplevel"], cwd=ai_dir).strip()
    ).resolve()
    rel = ai_dir.resolve().relative_to(top).as_posix()
    rel_prefix = "" if rel in ("", ".") else rel
    code_paths = [f"{rel_prefix}/{sub}" if rel_prefix else sub for sub in CODE_SUBTREES]
    return top, rel_prefix, code_paths


def _mktree_scoped(
    full_tree: str,
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> str:
    entries = _git_stdout(["ls-tree", full_tree, *CODE_SUBTREES], cwd=cwd, env=env)
    return _git_stdout(
        ["mktree"],
        cwd=cwd,
        env=env,
        input_text=entries if entries.endswith("\n") or not entries else entries + "\n",
    ).strip()


def _write_scoped_index_tree(top: Path, rel_prefix: str, env: dict[str, str]) -> str:
    write_args = ["write-tree", *([f"--prefix={rel_prefix}/"] if rel_prefix else [])]
    full_tree = _git_stdout(write_args, cwd=top, env=env).strip()
    return _mktree_scoped(full_tree, cwd=top, env=env)


def commit_ai_tree(rev: str = "HEAD", ai_dir: Path = AI_ROOT) -> str:
    """Return the 40-char git tree SHA for `CODE_SUBTREES` under `ai/` at `rev`."""
    top, rel_prefix, _ = _git_repo_and_prefix(ai_dir)
    tree_spec = f"{rev}:{rel_prefix}" if rel_prefix else rev
    return _mktree_scoped(tree_spec, cwd=top)


def code_identity(ai_dir: Path = AI_ROOT) -> CodeIdentity:
    """Compute reproducible code provenance for `CODE_SUBTREES` under `ai/`."""
    top, rel_prefix, code_paths = _git_repo_and_prefix(ai_dir)
    base_commit = _git_stdout(["rev-parse", "HEAD"], cwd=top).strip()
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git_stdout(["read-tree", "HEAD"], cwd=top, env=env)
        _git_stdout(["add", "-A", "--", *code_paths], cwd=top, env=env)
        patch = _git_stdout(
            ["diff", "--cached", "--binary", "HEAD", "--", *code_paths],
            cwd=top,
            env=env,
        )
        ai_tree = _write_scoped_index_tree(top, rel_prefix, env)
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest() if patch else ""
    return CodeIdentity(
        base_commit=base_commit,
        ai_tree=ai_tree,
        patch=patch,
        patch_sha256=patch_sha256,
    )


def verify_code_identity(
    base_commit: str,
    ai_tree: str,
    patch_sha256: str = "",
    patch_path: Path | None = None,
    ai_dir: Path = AI_ROOT,
) -> None:
    """Verify that `(base_commit, ai_tree, patch_sha256)` is reproducible in git."""
    if not _HEX40_RE.match(base_commit or ""):
        raise ValueError(f"base_commit must be a 40-char hex SHA, got {base_commit!r}")
    if not _HEX40_RE.match(ai_tree or ""):
        raise ValueError(f"ai_tree must be a 40-char hex SHA, got {ai_tree!r}")

    top, rel_prefix, _ = _git_repo_and_prefix(ai_dir)
    has_commit = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{base_commit}^{{commit}}"],
            cwd=top,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )

    if not patch_sha256:
        if has_commit:
            expected_tree = commit_ai_tree(base_commit, ai_dir=ai_dir)
            if expected_tree != ai_tree:
                raise ValueError(
                    f"ai_tree {ai_tree} does not match {base_commit} tree {expected_tree}"
                )
            return
        head_tree = commit_ai_tree("HEAD", ai_dir=ai_dir)
        if head_tree == ai_tree:
            return
        raise ValueError(
            f"base_commit {base_commit} is not in local git and ai_tree {ai_tree} "
            f"does not match HEAD ({head_tree})"
        )

    if commit_ai_tree("HEAD", ai_dir=ai_dir) == ai_tree:
        return
    if not has_commit:
        raise ValueError(f"patched run requires base_commit {base_commit} in local git")
    if patch_path is None or not patch_path.exists():
        raise ValueError(
            f"run has patch_sha256={patch_sha256[:12]}... but patch.diff is missing"
        )
    actual_patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if actual_patch_sha != patch_sha256:
        raise ValueError(
            f"patch.diff sha256 ({actual_patch_sha}) != patch_sha256 ({patch_sha256})"
        )
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        _git_stdout(["read-tree", base_commit], cwd=top, env=env)
        _git_stdout(["apply", "--cached", str(patch_path.resolve())], cwd=top, env=env)
        reconstructed = _write_scoped_index_tree(top, rel_prefix, env)
    if reconstructed != ai_tree:
        raise ValueError(
            f"applying patch.diff to {base_commit} produced {reconstructed}, expected {ai_tree}"
        )


def summarize_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per split: mean and std over seeds, and the mean per-seed bootstrap CI.
    """
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = df.reindex(columns=COLUMNS)
    grouped = df.groupby(SUMMARY_GROUP_KEYS, dropna=False, as_index=False).agg(
        value=("value", "mean"),
        std=("value", lambda s: s.std(ddof=1)),
        ci_low=("ci_low", "mean"),
        ci_high=("ci_high", "mean"),
        n_seeds=("value", "count"),
    )
    sort_cols = [k for k in SORT_KEYS if k != "seed"]
    return grouped[SUMMARY_COLUMNS].sort_values(sort_cols, ignore_index=True)


def summarize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    `split_std` is the std of the split means, `seed_std` the mean within-split std.
    """
    if df.empty:
        return pd.DataFrame(columns=SPLIT_SUMMARY_COLUMNS)
    per_split = summarize_seeds(df)
    grouped = per_split.groupby(
        [*SPLIT_GROUP_KEYS, "data_sha256"], dropna=False, as_index=False
    ).agg(
        value=("value", "mean"),
        split_std=("value", lambda s: s.std(ddof=1)),
        seed_std=("std", "mean"),
        n_splits=("split_id", "nunique"),
        n_seeds=("n_seeds", "sum"),
    )
    return grouped[SPLIT_SUMMARY_COLUMNS].sort_values(
        SPLIT_GROUP_KEYS, ignore_index=True
    )
