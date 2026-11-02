"""
Train a candidate MuFu run on the shared benchmark seeds.

From ai/: python drivers/train_mufu.py
Writes checkpoints, predictions (`val` and `test`), loss logs, `val_metrics.csv`, and
`val_paired.csv` into `ai/runs/<run_id>/`. Does not score `test` or modify
`ai/data/baseline/results.csv`; promote a winning run with `drivers/promote_mufu.py`.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baseline.benchmark import (
    RUNS_DIR,
    SEEDS,
    load_benchmark_split,
    make_run_id,
    run_benchmark,
)
from baseline.compare import compare_run_against_baselines
from baseline.results import DEFAULT_SPLIT_ID, code_identity
from encoders.smiles_encoder import SmilesEncoder
from fit_ensemble import fit_and_report
from mufu.data.dataset import split_datasets
from mufu.experiment import (
    DEFAULT_CONFIG,
    ENCODER,
    benchmark_params,
    make_fit_predict,
)

# training settings
model_name = "mufu"
run_name = ""  # optional label -> ai/runs/mufu_<run_name>_<split_id>/
seeds = SEEDS  # 0, 1, 2, 3, 4
split_id = "default"  # or "scaffold-0" ... "scaffold-4"
ensemble_after_training = True
# override fields here
config = replace(DEFAULT_CONFIG, device="cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(
            "Set training options at the top of this file, not as arguments."
        )

    identity = code_identity()
    run_id = make_run_id(model_name, split_id, run_name=run_name)
    run_dir = RUNS_DIR / run_id
    if (run_dir / "run.json").exists():
        raise SystemExit(
            f"Run directory {run_dir} already exists; set a new `run_name` or delete it."
        )

    print(f"Starting run {run_id} (ai_tree={identity.ai_tree[:12]})...", flush=True)
    print("Loading and featurizing the shared split...", flush=True)
    df = load_benchmark_split(split_id=split_id)
    datasets = split_datasets(df)
    for split in datasets:
        print(f"{split}: {len(datasets[split])} molecules")
    print(config, flush=True)

    will_fit_ensemble = (
        ensemble_after_training and split_id == DEFAULT_SPLIT_ID and len(seeds) >= 2
    )
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))
    fit_predict = make_fit_predict(
        datasets,
        smiles_encoder,
        config,
        run_dir=run_dir,
        model_name=model_name,
        identity=identity,
        verbose=True,
    )
    val_summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder=ENCODER,
        params=benchmark_params(config, smiles_encoder),
        seeds=seeds,
        split_id=split_id,
        df=df,
        run_dir=run_dir,
        identity=identity,
        mark_complete=not will_fit_ensemble,
    )
    columns = ["dataset", "endpoint", "split", "metric", "value", "std", "n_seeds"]
    print("\nValidation summary (per-seed mean +/- std):")
    print(val_summary[columns].round(4).to_string(index=False))

    if will_fit_ensemble:
        fit_and_report(
            df,
            datasets,
            smiles_encoder,
            run_dir=run_dir,
            model_name=model_name,
            seeds=seeds,
        )
        meta_path = run_dir / "run.json"
        meta = json.loads(meta_path.read_text())
        meta["status"] = "complete"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    val_paired = compare_run_against_baselines(run_dir, df, split="val", seeds=seeds)
    if not val_paired.empty:
        print("\nValidation paired bootstrap CIs (candidate - reference):")
        print(val_paired.round(4).to_string(index=False))

    print(
        f"\nRun complete: {run_dir}\n"
        f"To promote this run: set `run_id = {run_id!r}` in drivers/promote_mufu.py"
    )
