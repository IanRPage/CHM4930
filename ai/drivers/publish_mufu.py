"""
Publish the promoted MuFu model in `ai/runs/<run_id>/` to the HuggingFace Hub.

From ai/: python drivers/publish_mufu.py <major|minor|patch>
Needs a write token (`hf auth login`) and the run promoted into `data/baseline/`. With
`dry_run = True`, only writes the model card to `ai/runs/<run_id>/README.md`.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from huggingface_hub import HfApi

from baseline.benchmark import PROMOTED_PATH, resolve_run_dir
from baseline.results import load_results
from mufu.hub import BUMP_PARTS, load_version, model_card, next_tag, upload_version

repo_id = "CHM4930/mufu"
run_id = ""  # empty -> promoted default run
model_name = "mufu"
dry_run = True


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in BUMP_PARTS:
        raise SystemExit(
            f"Usage: python drivers/publish_mufu.py <{'|'.join(BUMP_PARTS)}>"
        )

    if not PROMOTED_PATH.exists():
        raise SystemExit(f"Missing {PROMOTED_PATH}; run promote_mufu.py first.")
    run_dir = resolve_run_dir(run_id, model=model_name)
    api = HfApi()
    target_tag = next_tag(api, repo_id, sys.argv[1])
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
        upload_version(api, repo_id, version, target_tag, card)
        promoted = json.loads(PROMOTED_PATH.read_text())
        promoted["tag"] = target_tag
        PROMOTED_PATH.write_text(json.dumps(promoted, indent=2) + "\n")
        print(f"published https://huggingface.co/{repo_id}/tree/{target_tag}")
