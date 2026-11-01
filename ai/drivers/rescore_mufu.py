"""
Re-score saved MuFu seed checkpoints on the default split without retraining, e.g.
after a metric change. Rows keep the checkpoints' training commit in `git_commit`.

From ai/: python drivers/rescore_mufu.py

Use the published ensemble's checkpoints (drivers/download_mufu.py, copied into
checkpoints/). Refuses checkpoints from another data snapshot or from a different commit
than the recorded MuFu rows, unless `allow_new_commit = True`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baseline.benchmark import SEEDS, load_benchmark_split, run_benchmark
from baseline.results import file_sha256, load_results
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import split_datasets
from mufu.ensemble import member_paths
from mufu.experiment import ENCODER, load_mufu_checkpoint
from mufu.train import predict
from pipeline.combined_loader import COMBINED_CSV_PATH

AI_DIR = Path(__file__).resolve().parents[1]

model_name = "mufu"
batch_size = 64
allow_new_commit = False
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    paths = dict(zip(SEEDS, member_paths(AI_DIR / "checkpoints", model_name, SEEDS)))
    ckpts = {
        s: torch.load(p, map_location="cpu", weights_only=True)
        for s, p in paths.items()
    }
    if any(c["seed"] != s for s, c in ckpts.items()):
        raise SystemExit("a checkpoint's seed doesn't match its file name")
    runs = {(c["git_commit"], c["data_sha256"]) for c in ckpts.values()}
    if len(runs) != 1:
        raise SystemExit(f"checkpoints come from different runs: {runs}")
    ((commit, data_sha256),) = runs
    if data_sha256 != file_sha256(COMBINED_CSV_PATH):
        raise SystemExit("checkpoints were trained on a different data snapshot")
    recorded = load_results()
    recorded = set(recorded.loc[recorded["model"] == model_name, "git_commit"])
    if recorded and commit not in recorded and not allow_new_commit:
        raise SystemExit(f"checkpoints are from {commit}; results.csv has {recorded}")

    df = load_benchmark_split()
    datasets = split_datasets(df)
    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

    def fit_predict(df, seed):
        model, meta = load_mufu_checkpoint(paths[seed], smiles_encoder, device=device)
        return {
            split: predict(
                model,
                datasets[split],
                batch_size=batch_size,
                family_states=meta["family_states"],
                calibration=meta["calibration"],
            )
            for split in ("val", "test")
        }

    summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder=ENCODER,
        params=ckpts[SEEDS[0]]["params"],
        df=df,
        commit=commit,
    )
    columns = ["endpoint", "split", "metric", "value", "std", "ci_low", "ci_high"]
    print(summary[columns].round(4).to_string(index=False))
