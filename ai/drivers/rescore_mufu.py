"""
Recompute predictions and metrics from saved checkpoints in `ai/runs/<run_id>/`.

From ai/: python drivers/rescore_mufu.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baseline.benchmark import (
    SEEDS,
    is_promoted_run,
    load_benchmark_split,
    rebuild_tracked_results,
    resolve_run_dir,
    save_split_predictions,
    score_run_predictions,
)
from baseline.results import DEFAULT_SPLIT_ID, file_sha256, summarize_seeds
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.ensemble import member_paths
from mufu.experiment import load_mufu_checkpoint
from mufu.mc_dropout import (
    load_conformal,
    predict_with_intervals,
)
from mufu.mc_dropout import (
    sidecar_path as conformal_sidecar_path,
)
from mufu.train import predict
from pipeline.combined_loader import COMBINED_CSV_PATH

run_id = ""  # empty -> promoted default run
model_name = "mufu"
batch_size = 64
overwrite_predictions = False
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    run_dir = resolve_run_dir(run_id, model=model_name)
    meta = json.loads((run_dir / "run.json").read_text())
    seeds = tuple(int(s) for s in meta.get("seeds", SEEDS))
    split_id = str(meta.get("split_id", DEFAULT_SPLIT_ID))

    paths = dict(zip(seeds, member_paths(run_dir / "checkpoints", model_name, seeds)))
    ckpts = {
        s: torch.load(p, map_location="cpu", weights_only=True)
        for s, p in paths.items()
    }
    if any(c["seed"] != s for s, c in ckpts.items()):
        raise SystemExit("a checkpoint's seed doesn't match its file name")
    runs = {
        (c.get("base_commit") or c.get("git_commit"), c.get("data_sha256"))
        for c in ckpts.values()
    }
    if len(runs) != 1:
        raise SystemExit(f"checkpoints come from different runs: {runs}")
    ((_, data_sha256),) = runs
    if data_sha256 != file_sha256(COMBINED_CSV_PATH):
        raise SystemExit("checkpoints were trained on a different data snapshot")

    df = load_benchmark_split(split_id=split_id)
    datasets = split_datasets(df)
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

    preds_dir = run_dir / "predictions"
    for seed in seeds:
        ckpt_path = paths[seed]
        model, loaded_meta = load_mufu_checkpoint(
            ckpt_path, smiles_encoder, device=device
        )
        val_preds = predict(
            model,
            datasets["val"],
            batch_size=batch_size,
            family_states=loaded_meta["family_states"],
            calibration=loaded_meta["calibration"],
        )
        test_preds = dict(
            predict(
                model,
                datasets["test"],
                batch_size=batch_size,
                family_states=loaded_meta["family_states"],
                calibration=loaded_meta["calibration"],
            )
        )
        if conformal_sidecar_path(ckpt_path).exists():
            conformal = load_conformal(ckpt_path)
            param = next(model.parameters(), None)
            devices = (
                [param.device]
                if param is not None and param.device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                intervals = predict_with_intervals(
                    model, datasets["test"], conformal, batch_size=batch_size
                )
            for key, val in intervals.items():
                if key not in test_preds:
                    test_preds[key] = val

        save_split_predictions(
            df,
            "val",
            val_preds,
            preds_dir / f"val_seed_{seed}.csv",
            overwrite=overwrite_predictions,
        )
        save_split_predictions(
            df,
            "test",
            test_preds,
            preds_dir / f"test_seed_{seed}.csv",
            overwrite=overwrite_predictions,
        )

    if is_promoted_run(run_dir, model=model_name, split_id=split_id):
        all_rows = rebuild_tracked_results()
        scored = all_rows[all_rows["run_id"] == run_dir.name]
    else:
        scored = score_run_predictions(run_dir, df, include_test=False)

    summary = summarize_seeds(scored)
    columns = ["endpoint", "split", "metric", "value", "std", "ci_low", "ci_high"]
    print(summary[columns].round(4).to_string(index=False))
