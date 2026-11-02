"""
Paired bootstrap of one run minus another on a split (`val` or `test`), from the per-seed
predictions saved in each run's `predictions/` directory. Each resample draws compounds once
and scores both models on them, with the delta averaged over seeds.
"""

import json
import logging
import multiprocessing as mp
import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import benchmark
from baseline.results import DEFAULT_SPLIT_ID, LOWER_IS_BETTER, RESULTS_PATH
from pipeline.combined_loader import TOX_TASKS
from pipeline.toxicity_loader import DATASETS

log = logging.getLogger(__name__)

PAIRED_PATH = RESULTS_PATH.with_name("results_paired.csv")
PAIRED_COLUMNS = [
    "a",
    "b",
    "split_id",
    "split",
    "endpoint",
    "metric",
    "delta",
    "ci_low",
    "ci_high",
    "p_a_better",
    "n_seeds",
    "n_boot",
]
HEADLINE = {
    "pIC50_BACE1": ("rmse", "spearman"),
    "pIC50_EGFR": ("rmse", "spearman"),
    "macro_tox21": ("roc_auc", "pr_auc"),
    "CT_TOX": ("roc_auc", "pr_auc"),
    "FDA_APPROVED": ("roc_auc",),
}


def _columns(endpoint: str) -> list[str]:
    group = endpoint.removeprefix("macro_")
    if group == endpoint:
        return [endpoint]
    return list(TOX_TASKS) if group == "overall" else list(DATASETS[group]["tasks"])


def _run_label(run_dir: Path) -> str:
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return str(meta.get("model") or meta.get("run_id") or run_dir.name)
    return run_dir.name


def _predictions(run_dir: Path, split: str, seed: int) -> pd.DataFrame:
    path = run_dir / "predictions" / f"{split}_seed_{int(seed)}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing prediction file {path}")
    # round_trip parsing so tied scores stay tied, as when they were scored
    return pd.read_csv(path, float_precision="round_trip")


_FORK_PAIRS: list[tuple] = []
_FORK_ENDPOINT: str = ""
_FORK_METRICS: tuple[str, ...] = ()


def _paired_values(
    tables_pair, idx: np.ndarray, endpoint: str, metrics: Sequence[str]
) -> np.ndarray:
    rows = benchmark._score(*tables_pair, idx)
    found = {r["metric"]: r["value"] for r in rows if r["endpoint"] == endpoint}
    return np.array([found[m] for m in metrics], dtype=float)


def _paired_boot_chunk(chunk: list[np.ndarray]) -> np.ndarray:
    return np.array(
        [
            np.mean(
                [
                    _paired_values(ta, idx, _FORK_ENDPOINT, _FORK_METRICS)
                    - _paired_values(tb, idx, _FORK_ENDPOINT, _FORK_METRICS)
                    for ta, tb in _FORK_PAIRS
                ],
                axis=0,
            )
            for idx in chunk
        ],
        dtype=float,
    )


def paired_bootstrap(
    df: pd.DataFrame,
    run_a: Path,
    run_b: Path,
    endpoint: str,
    metrics: Sequence[str],
    *,
    split: str = "test",
    split_id: str = DEFAULT_SPLIT_ID,
    seeds: Sequence[int] = benchmark.SEEDS,
    n_boot: int = benchmark.N_BOOT,
    boot_seed: int = 0,
    label_a: str | None = None,
    label_b: str | None = None,
) -> list[dict]:
    """
    `run_a` minus `run_b` per metric on `split`: the delta, its 2.5/97.5 percentiles over
    resamples, and the share of resamples where `run_a` is better. `df` must already be
    split as `split_id`.
    """
    global _FORK_PAIRS, _FORK_ENDPOINT, _FORK_METRICS
    split_df = df[df["split"] == split]
    columns = _columns(endpoint)
    name_a = label_a if label_a is not None else _run_label(run_a)
    name_b = label_b if label_b is not None else _run_label(run_b)

    def tables(run_dir: Path, label: str, seed: int):
        preds = _predictions(run_dir, split, seed)
        if preds["smiles"].tolist() != split_df["smiles"].tolist():
            raise ValueError(
                f"{label} seed {seed}: SMILES do not match the {split} split"
            )
        scores = {c: preds[c].to_numpy() for c in columns}
        return benchmark._split_tables(split_df, scores, split)

    pairs = [(tables(run_a, name_a, s), tables(run_b, name_b, s)) for s in seeds]

    def delta(idx):
        return np.mean(
            [
                _paired_values(ta, idx, endpoint, metrics)
                - _paired_values(tb, idx, endpoint, metrics)
                for ta, tb in pairs
            ],
            axis=0,
        )

    n = len(split_df)
    rng = np.random.default_rng(boot_seed)
    sels = [rng.integers(0, n, n) for _ in range(n_boot)]
    if n_boot >= 200 and "fork" in mp.get_all_start_methods():
        _FORK_PAIRS = pairs
        _FORK_ENDPOINT = endpoint
        _FORK_METRICS = tuple(metrics)
        chunk_size = max(1, (n_boot + 19) // 20)
        chunks = [sels[i : i + chunk_size] for i in range(0, n_boot, chunk_size)]
        with mp.get_context("fork").Pool() as pool:
            boots = np.vstack(pool.map(_paired_boot_chunk, chunks))
    else:
        boots = np.array([delta(idx) for idx in sels], dtype=float)
    point = delta(np.arange(n))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(boots, [2.5, 97.5], axis=0)

    rows = []
    for j, metric in enumerate(metrics):
        draws = boots[:, j][~np.isnan(boots[:, j])]
        better = draws < 0 if metric in LOWER_IS_BETTER else draws > 0
        rows.append(
            {
                "a": name_a,
                "b": name_b,
                "split_id": split_id,
                "split": split,
                "endpoint": endpoint,
                "metric": metric,
                "delta": point[j],
                "ci_low": low[j],
                "ci_high": high[j],
                "p_a_better": better.mean(),
                "n_seeds": len(seeds),
                "n_boot": len(draws),
            }
        )
    return rows


def _has_split_predictions(run_dir: Path, split: str, seeds: Sequence[int]) -> bool:
    return all(
        (run_dir / "predictions" / f"{split}_seed_{int(s)}.csv").exists() for s in seeds
    )


def compare_run_against_baselines(
    run_dir: Path,
    df: pd.DataFrame | None = None,
    *,
    split: str = "val",
    promoted_path: Path | None = None,
    runs_dir: Path | None = None,
    include_promoted_mufu: bool = True,
    seeds: Sequence[int] | None = None,
    n_boot: int = benchmark.N_BOOT,
) -> pd.DataFrame:
    """Paired bootstrap of `run_dir` against promoted baselines on `split`, saved to `<split>_paired.csv`."""
    meta = json.loads((run_dir / "run.json").read_text())
    split_id = str(meta.get("split_id", DEFAULT_SPLIT_ID))
    if seeds is None:
        seeds = tuple(meta.get("seeds", benchmark.SEEDS))
    if df is None:
        df = benchmark.load_benchmark_split(split_id=split_id)

    prom_path = promoted_path or benchmark.PROMOTED_PATH
    base_runs_dir = runs_dir or benchmark.RUNS_DIR
    if not prom_path.exists():
        log.warning(
            "promoted.json not found at %s; skipping paired comparison", prom_path
        )
        return pd.DataFrame(columns=PAIRED_COLUMNS)

    promoted = json.loads(prom_path.read_text())
    rows: list[dict] = []

    mufu_id = promoted.get("mufu", {}).get(split_id)
    if include_promoted_mufu and mufu_id and mufu_id != run_dir.name:
        mufu_dir = base_runs_dir / mufu_id
        if _has_split_predictions(mufu_dir, split, seeds):
            for endpoint, metrics in HEADLINE.items():
                rows += paired_bootstrap(
                    df,
                    run_dir,
                    mufu_dir,
                    endpoint,
                    metrics,
                    split=split,
                    split_id=split_id,
                    seeds=seeds,
                    n_boot=n_boot,
                    label_a=run_dir.name,
                    label_b=mufu_id,
                )
        else:
            log.warning(
                "skipping %s comparison against promoted mufu (%s): missing predictions in %s "
                "(populate via `python drivers/download_mufu.py`)",
                split,
                mufu_id,
                mufu_dir,
            )

    for endpoint, metrics in HEADLINE.items():
        rf_model = "rf_bioactivity" if endpoint.startswith("pIC50_") else "rf_toxicity"
        rf_id = promoted.get(rf_model, {}).get(split_id)
        if not rf_id:
            continue
        rf_dir = base_runs_dir / rf_id
        if not _has_split_predictions(rf_dir, split, seeds):
            log.warning(
                "skipping %s comparison on %s against %s (%s): missing predictions in %s "
                "(populate via `PYTHONPATH=src python -m baseline.%s`)",
                split,
                endpoint,
                rf_model,
                rf_id,
                rf_dir,
                "baseline_bioactivity"
                if rf_model == "rf_bioactivity"
                else "baseline_toxicity",
            )
            continue
        rows += paired_bootstrap(
            df,
            run_dir,
            rf_dir,
            endpoint,
            metrics,
            split=split,
            split_id=split_id,
            seeds=seeds,
            n_boot=n_boot,
            label_a=run_dir.name if mufu_id != run_dir.name else "mufu",
            label_b=rf_model,
        )

    out = pd.DataFrame(rows, columns=PAIRED_COLUMNS)
    out_path = run_dir / f"{split}_paired.csv"
    out.to_csv(out_path, index=False, lineterminator="\n")
    return out
