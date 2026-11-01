"""
Paired bootstrap of one model minus another on a split, from the test predictions
`run_benchmark` saves. Each resample draws test compounds once and scores both models on
them, with the delta averaged over seeds. The CI covers test-compound sampling for these
trained seeds; it doesn't resample training, so it says nothing about a retrain, though
seed spread is small next to it. Writes a model minus RF on the default split to
`data/baseline/results_paired.csv`, replacing that model's earlier rows.

From `ai/`, with `PYTHONPATH=src` (the model defaults to `mufu`):

    python -m baseline.compare [model]
"""

import logging
import sys
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


def _predictions(model, split_id, seed, predictions_dir) -> pd.DataFrame:
    directory = predictions_dir or benchmark.PREDICTIONS_DIR
    paths = list(directory.glob(f"{model}__*__{split_id}__seed{seed}.csv"))
    if len(paths) != 1:
        raise FileNotFoundError(
            f"expected one {model} {split_id} seed {seed} prediction file, "
            f"found {len(paths)}"
        )
    # round_trip parsing so tied scores stay tied, as when they were scored
    return pd.read_csv(paths[0], float_precision="round_trip")


def paired_bootstrap(
    df: pd.DataFrame,
    a: str,
    b: str,
    endpoint: str,
    metrics: Sequence[str],
    *,
    split_id: str = DEFAULT_SPLIT_ID,
    seeds: Sequence[int] = benchmark.SEEDS,
    n_boot: int = benchmark.N_BOOT,
    boot_seed: int = 0,
    predictions_dir: Path | None = None,
) -> list[dict]:
    """
    `a` minus `b` per metric: the delta, its 2.5/97.5 percentiles over resamples, and
    the share of resamples where `a` is better. `df` must already be split as `split_id`.
    """
    test = df[df["split"] == "test"]
    columns = _columns(endpoint)

    def tables(model, seed):
        preds = _predictions(model, split_id, seed, predictions_dir)
        if preds["smiles"].tolist() != test["smiles"].tolist():
            raise ValueError(f"{model} seed {seed}: SMILES do not match the test split")
        scores = {c: preds[c].to_numpy() for c in columns}
        return benchmark._split_tables(test, scores, "test")

    def values(tables, idx):
        rows = benchmark._score(*tables, idx)
        found = {r["metric"]: r["value"] for r in rows if r["endpoint"] == endpoint}
        return np.array([found[m] for m in metrics], dtype=float)

    pairs = [(tables(a, s), tables(b, s)) for s in seeds]

    def delta(idx):
        return np.mean([values(ta, idx) - values(tb, idx) for ta, tb in pairs], axis=0)

    n = len(test)
    rng = np.random.default_rng(boot_seed)
    boots = np.array([delta(rng.integers(0, n, n)) for _ in range(n_boot)])
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
                "a": a,
                "b": b,
                "split_id": split_id,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 2:
        raise SystemExit("usage: python -m baseline.compare [model]")
    model = sys.argv[1] if len(sys.argv) == 2 else "mufu"
    df = benchmark.load_benchmark_split()
    rows = []
    for endpoint, metrics in HEADLINE.items():
        rf = "rf_bioactivity" if endpoint.startswith("pIC50_") else "rf_toxicity"
        log.info("bootstrapping %s - %s on %s", model, rf, endpoint)
        rows += paired_bootstrap(df, model, rf, endpoint, metrics)
    new = pd.DataFrame(rows)
    if PAIRED_PATH.exists():
        old = pd.read_csv(PAIRED_PATH)
        new = pd.concat([old[old["a"] != model], new], ignore_index=True)
    new.to_csv(PAIRED_PATH, index=False)
    log.info("wrote %d rows to %s", len(rows), PAIRED_PATH)
    print(new[new["a"] == model].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
