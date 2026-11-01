"""
Default-split test metrics bucketed by each compound's nearest-neighbour ECFP4 Tanimoto
similarity to train, from the predictions `run_benchmark` saves. Writes the mean and std
over seeds to `data/baseline/results_by_similarity.csv`.

From `ai/`, with `PYTHONPATH=src`:

    python -m baseline.similarity
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import benchmark
from baseline.results import RESULTS_PATH
from featurize.fingerprint import FINGERPRINT_BITS, mol_to_ecfp4
from pipeline.preprocess import to_mol

log = logging.getLogger(__name__)

SIMILARITY_PATH = RESULTS_PATH.with_name("results_by_similarity.csv")
BUCKET_EDGES = (0.3, 0.4, 0.5, 0.6)
BUCKETS = ("[0.0, 0.3)", "[0.3, 0.4)", "[0.4, 0.5)", "[0.5, 0.6)", "[0.6, 1.0]")
REPORTED = {
    "pIC50_BACE1": ("rmse", "spearman", "roc_auc"),
    "pIC50_EGFR": ("rmse", "spearman", "roc_auc"),
    "macro_tox21": ("roc_auc",),
    "macro_clintox": ("roc_auc",),
    "CT_TOX": ("roc_auc",),
}
KEYS = ["model", "encoder", "dataset", "endpoint", "metric", "bucket", "n_compounds"]


def nearest_train_similarity(df: pd.DataFrame, split: str = "test") -> np.ndarray:
    """
    Max Tanimoto to any train compound, aligned with `df[df["split"] == split]`.
    """
    fps = np.stack(
        [mol_to_ecfp4(to_mol(s), FINGERPRINT_BITS) for s in df["smiles"]]
    ).astype(np.float32)
    splits = df["split"].to_numpy()
    train, query = fps[splits == "train"], fps[splits == split]
    inter = query @ train.T
    union = query.sum(axis=1)[:, None] + train.sum(axis=1)[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return sim.max(axis=1)


def bucket_labels(similarity: np.ndarray) -> np.ndarray:
    return np.array(BUCKETS)[np.digitize(similarity, BUCKET_EDGES)]


def bucket_rows(
    df: pd.DataFrame, predictions: pd.DataFrame, buckets: np.ndarray
) -> list[dict]:
    """
    `predictions` holds `smiles` and score columns aligned with the test rows.
    """
    test = df[df["split"] == "test"].reset_index(drop=True)
    if predictions["smiles"].tolist() != test["smiles"].tolist():
        raise ValueError("prediction SMILES do not match the test split")
    rows = []
    for bucket in sorted(set(buckets)):
        in_bucket = buckets == bucket
        scores = {c: predictions.loc[in_bucket, c].to_numpy() for c in predictions}
        scored = benchmark.evaluate_split_predictions(test[in_bucket], {"test": scores})
        rows += [
            {**row, "bucket": bucket, "n_compounds": int(in_bucket.sum())}
            for row in scored
            if row["metric"] in REPORTED.get(row["endpoint"], ())
        ]
    return rows


def similarity_table(
    df: pd.DataFrame, predictions_dir: Path | None = None
) -> pd.DataFrame:
    directory = predictions_dir or benchmark.PREDICTIONS_DIR
    files = sorted(directory.glob("*__default__seed*.csv"))
    if not files:
        raise FileNotFoundError(f"no default-split predictions in {directory}")
    buckets = bucket_labels(nearest_train_similarity(df))

    rows = []
    for path in files:
        model, encoder, _, seed = path.stem.split("__")
        rows += [
            {**row, "model": model, "encoder": encoder, "seed": seed}
            for row in bucket_rows(
                df, pd.read_csv(path, float_precision="round_trip"), buckets
            )
        ]
    return (
        pd.DataFrame(rows)
        .groupby(KEYS, dropna=False, as_index=False)
        .agg(
            n_labeled=("n_labeled", "first"),
            value=("value", "mean"),
            std=("value", lambda s: s.std(ddof=1)),
            n_seeds=("value", "count"),
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    table = similarity_table(benchmark.load_benchmark_split())
    table.to_csv(SIMILARITY_PATH, index=False)
    log.info("wrote %d rows to %s", len(table), SIMILARITY_PATH)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
