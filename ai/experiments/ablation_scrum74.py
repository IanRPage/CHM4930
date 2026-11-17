"""
SCRUM-74: eval-time modality ablation on promoted MuFu checkpoints.

Runs all six mask patterns (three drop-one, three keep-one) over every
promoted mufu run (6 splits × 5 seeds), with and without mask rescaling.
CPU only, no retraining.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

torch.set_num_threads(12)

import pandas as pd

from baseline.benchmark import (
    SEEDS,
    evaluate_split_predictions,
    load_benchmark_split,
    resolve_run_dir,
    save_split_predictions,
)
from baseline.results import DEFAULT_SPLIT_ID
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.ensemble import member_paths
from mufu.experiment import load_mufu_checkpoint, predict_mufu_splits

MODEL_NAME = "mufu"
BATCH_SIZE = 64
DEVICE = "cpu"

MASK_VARIANTS: dict[str, tuple[bool, ...]] = {
    "ecfp4-gine": (False, True, True),
    "molformer-gine": (True, False, True),
    "molformer-ecfp4": (True, True, False),
    "molformer": (True, False, False),
    "ecfp4": (False, True, False),
    "gine": (False, False, True),
}

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_DIR / "scrum74_ablation.csv"
RESULTS_COLUMNS = [
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
]


def mufu_run_ids() -> list[tuple[str, str]]:
    runs_csv = (
        Path(__file__).resolve().parents[1] / "data" / "baseline" / "results_runs.csv"
    )
    df = pd.read_csv(runs_csv)
    mufu = df[df["model"] == MODEL_NAME]
    return list(zip(mufu["run_id"], mufu["split_id"]))


def run_ablation() -> pd.DataFrame:
    all_rows: list[dict] = []
    runs = mufu_run_ids()
    print(f"Found {len(runs)} mufu runs", flush=True)

    for run_id, split_id in runs:
        run_dir = resolve_run_dir(run_id, model=MODEL_NAME)
        if not (run_dir / "run.json").exists():
            print(f"[{split_id}] skipping {run_id} — run directory missing", flush=True)
            continue
        meta = json.loads((run_dir / "run.json").read_text())
        seeds = tuple(int(s) for s in meta.get("seeds", SEEDS))
        split_id = str(meta.get("split_id", DEFAULT_SPLIT_ID))

        df = load_benchmark_split(split_id=split_id)
        datasets = split_datasets(df)
        smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

        ckpt_dir = run_dir / "checkpoints"
        paths = dict(zip(seeds, member_paths(ckpt_dir, MODEL_NAME, seeds)))

        for seed in seeds:
            model, loaded_meta = load_mufu_checkpoint(
                paths[seed],
                smiles_encoder,
                device=DEVICE,
            )
            for rescale in (False, True):
                model.fusion.rescale = rescale
                for variant_name, mask_pattern in MASK_VARIANTS.items():
                    encoder_name = variant_name + ("-rescaled" if rescale else "")
                    mask_tensor = torch.tensor(mask_pattern, dtype=torch.bool)

                    preds = predict_mufu_splits(
                        model,
                        datasets,
                        seed=seed,
                        batch_size=BATCH_SIZE,
                        family_states=loaded_meta["family_states"],
                        calibration=None,
                        conformal=None,
                        mask=mask_tensor,
                    )

                    ablation_dir = run_dir / "predictions" / "ablation" / encoder_name
                    for split in ("val", "test"):
                        save_split_predictions(
                            df,
                            split,
                            preds[split],
                            ablation_dir / f"{split}_seed_{seed}.csv",
                        )

                    rows = evaluate_split_predictions(df, preds, n_boot=2000)
                    for row in rows:
                        row["model"] = MODEL_NAME
                        row["encoder"] = encoder_name
                        row["split_id"] = split_id
                        row["seed"] = seed
                    all_rows.extend(rows)
                    print(f"[{split_id}] seed {seed} | {encoder_name}", flush=True)

    return pd.DataFrame(all_rows)


def print_summary(results: pd.DataFrame) -> None:
    test = results[results["split"] == "test"]
    headline = test[
        test["metric"].isin(["roc_auc", "rmse"])
        & test["endpoint"].isin(
            ["active_BACE1", "active_EGFR", "pIC50_BACE1", "pIC50_EGFR", "tox"]
        )
    ]
    if headline.empty:
        print("No headline metrics found.")
        return
    summary = (
        headline.groupby(["encoder", "endpoint", "metric"])["value"]
        .agg(["mean", "std"])
        .round(4)
    )
    print("\n=== Ablation summary (test, mean ± std across splits × seeds) ===")
    print(summary.to_string())


if __name__ == "__main__":
    results = run_ablation()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results[RESULTS_COLUMNS].to_csv(OUTPUT_PATH, index=False, lineterminator="\n")
    print(f"\nResults written to {OUTPUT_PATH}")
    print_summary(results)
