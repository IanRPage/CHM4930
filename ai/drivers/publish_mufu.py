"""
Publish the promoted MuFu model in `ai/runs/<run_id>/` to the HuggingFace Hub.

From ai/: python drivers/publish_mufu.py
Needs a write token (`hf auth login`) and the run promoted into `data/baseline/`. With
`dry_run = True`, only writes the model card to `ai/runs/<run_id>/README.md`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from huggingface_hub import HfApi

from baseline.benchmark import PROMOTED_PATH, RUNS_DIR
from baseline.results import DEFAULT_SPLIT_ID, load_results
from mufu.hub import load_version, model_card, upload_version

repo_id = "CHM4930/mufu"
run_id = ""  # empty -> promoted default run
tag = ""  # empty -> promoted["tag"]
model_name = "mufu"
dry_run = True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    if not PROMOTED_PATH.exists():
        raise SystemExit(f"Missing {PROMOTED_PATH}; run promote_mufu.py first.")
    promoted = json.loads(PROMOTED_PATH.read_text())
    target_run_id = run_id or promoted.get(model_name, {}).get(DEFAULT_SPLIT_ID, "")
    target_tag = tag or promoted.get("tag", "v1.0.0")
    if not target_run_id:
        raise SystemExit("No promoted MuFu run found in promoted.json.")

    run_dir = RUNS_DIR / target_run_id
    version = load_version(run_dir / "checkpoints", model_name, verify_git=True)
    card = model_card(
        version,
        target_tag,
        load_results(),
        pd.read_csv(run_dir / "ensemble_metrics.csv"),
        repo_id,
    )
    if dry_run:
        card.save(run_dir / "README.md")
        print(f"dry run: wrote {run_dir / 'README.md'} as {target_tag}")
    else:
        upload_version(HfApi(), repo_id, version, target_tag, card)
        print(f"published https://huggingface.co/{repo_id}/tree/{target_tag}")
