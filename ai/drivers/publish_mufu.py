"""
Publish the MuFu ensemble in checkpoints/ to the HuggingFace Hub.

From ai/: python drivers/publish_mufu.py
Needs a write token (`hf auth login`) and the run's rows committed to results.csv. With
`dry_run`, only writes the model card to outputs/<model_name>/README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from huggingface_hub import HfApi

from baseline.results import load_results
from mufu.hub import load_version, model_card, upload_version

AI_DIR = Path(__file__).resolve().parents[1]

repo_id = "CHM4930/mufu"
tag = "v1.0.0"  # vMAJOR.MINOR.PATCH, see ai/README.md
model_name = "mufu"
dry_run = True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    outputs_dir = AI_DIR / "outputs" / model_name
    version = load_version(AI_DIR / "checkpoints", model_name)
    card = model_card(
        version,
        tag,
        load_results(),
        pd.read_csv(outputs_dir / "ensemble_metrics.csv"),
        repo_id,
    )
    if dry_run:
        card.save(outputs_dir / "README.md")
        print(f"dry run: wrote {outputs_dir / 'README.md'} as {tag}")
    else:
        upload_version(HfApi(), repo_id, version, tag, card)
        print(f"published https://huggingface.co/{repo_id}/tree/{tag}")
