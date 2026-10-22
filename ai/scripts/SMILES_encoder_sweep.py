"""Standalone frozen MolFormer plus a trainable BACE-1 regression head.

Use an immutable Hugging Face commit SHA in PRETRAINED_REVISION. All splitting,
selection, scoring, and explicit export use the shared benchmark workflow.

    python scripts/SMILES_encoder_sweep.py --stage-config pretrained.json
    python scripts/SMILES_encoder_sweep.py --benchmark --config chosen.json --dry-run
    python scripts/SMILES_encoder_sweep.py --benchmark --select-from outputs/encoders/SMILES/sweep/1-head.csv --write-results

Imports and dry runs do not load Transformers or pretrained weights.
"""

import copy
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from benchmark import config_json, evaluate_predictions, set_training_seed
from encoders.benchmark import run_cli
from pipeline.cache import DATA_DIR

OUT = DATA_DIR.parent / "outputs" / "encoders" / "SMILES" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT = {
    "MODEL_NAME": "ibm-research/MoLFormer-XL-both-10pct",
    "PRETRAINED_REVISION": None,
    "FREEZE": True,
    "MAX_LENGTH": 202,
    "ENCODER_BATCH_SIZE": 32,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "LR": 1e-3,
    "WEIGHT_DECAY": 0.0,
}
STAGES = {"1-head": ([DEFAULT | {"LR": lr} for lr in (1e-4, 1e-3, 1e-2)], [0, 1, 2])}


def normalize_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT):
        raise ValueError(
            "SMILES configuration must contain only the documented training keys"
        )
    cfg = DEFAULT | config
    revision = cfg["PRETRAINED_REVISION"]
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError(
            "PRETRAINED_REVISION must be an immutable 40-character commit SHA"
        )
    cfg["PRETRAINED_REVISION"] = revision.lower()
    if cfg["FREEZE"] is not True:
        raise ValueError("this benchmark requires FREEZE=true")
    if not isinstance(cfg["MODEL_NAME"], str) or not cfg["MODEL_NAME"].strip():
        raise ValueError("MODEL_NAME must be a nonempty pretrained model identifier")
    for key in ("MAX_LENGTH", "ENCODER_BATCH_SIZE", "BATCH_SIZE", "EPOCHS"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("LR", "WEIGHT_DECAY"):
        value = cfg[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{key} must be finite and numeric")
        if value < 0 or (key == "LR" and value == 0):
            raise ValueError(
                f"{key} must be {'positive' if key == 'LR' else 'nonnegative'}"
            )
    return cfg


def make_encoder(cfg):
    # Keep the benchmark CLI and mocked tests usable without this optional dependency.
    from encoders.smiles_encoder import SmilesEncoder

    return SmilesEncoder(
        model_name=cfg["MODEL_NAME"],
        freeze=cfg["FREEZE"],
        max_length=cfg["MAX_LENGTH"],
        revision=cfg["PRETRAINED_REVISION"],
    )


def train_head(cfg, seed, data, *, evaluate_test=False):
    generator = set_training_seed(seed)
    encoder = make_encoder(cfg).to(DEVICE).eval()
    resolved = getattr(encoder.model.config, "_commit_hash", None)
    if resolved is not None and resolved != cfg["PRETRAINED_REVISION"]:
        raise ValueError(
            "loaded pretrained revision does not match the requested commit"
        )
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise ValueError("the pretrained SMILES encoder must remain frozen")
    train, val = data.indices["train"], data.indices["val"]

    @torch.no_grad()
    def embeddings(indices):
        chunks = []
        for start in range(0, len(indices), cfg["ENCODER_BATCH_SIZE"]):
            idx = indices[start : start + cfg["ENCODER_BATCH_SIZE"]]
            smiles = data.frame.iloc[idx]["smiles"].tolist()
            encoded = encoder(smiles)
            if (
                encoded.shape != (len(idx), encoder.out_dim)
                or not torch.isfinite(encoded).all()
            ):
                raise ValueError(
                    "SMILES embeddings must be finite and aligned with the requested rows"
                )
            chunks.append(encoded.detach())
        return torch.cat(chunks)

    # Frozen embeddings use no labels or fitted preprocessing. Test is encoded only
    # after checkpoint selection when the shared scorer explicitly requests it.
    X_train, X_val = embeddings(train), embeddings(val)
    y_train = data.y[train].to(DEVICE)
    head = nn.Linear(encoder.out_dim, 1).to(DEVICE)
    nn.init.constant_(head.bias, y_train.mean().item())
    optimizer = torch.optim.Adam(
        head.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    loader = DataLoader(
        range(len(train)),
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        generator=generator,
    )

    @torch.no_grad()
    def predict(indices):
        head.eval()
        if np.array_equal(indices, train):
            features = X_train
        elif np.array_equal(indices, val):
            features = X_val
        else:
            features = embeddings(indices)
        return head(features).squeeze(-1).cpu()

    best_rmse, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, cfg["EPOCHS"] + 1):
        head.train()
        for idx in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(
                head(X_train[idx]).squeeze(-1), y_train[idx]
            ).backward()
            optimizer.step()
        rmse = evaluate_predictions(
            data.frame.iloc[val]["pIC50_BACE1"], predict(val), "regression"
        )["rmse"]
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = (
                rmse,
                copy.deepcopy(head.state_dict()),
                epoch,
            )
    if best_state is None:
        raise ValueError("head training produced no finite validation checkpoint")
    head.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "pretrained": {
            "model_name": encoder.model_name,
            "revision": cfg["PRETRAINED_REVISION"],
            "resolved_revision": resolved or cfg["PRETRAINED_REVISION"],
            "freeze": True,
            "head": "linear",
        },
        **data.score(predict, evaluate_test=evaluate_test),
    }


def run(cfg, seed, data, *, evaluate_test=False):
    cfg = normalize_config(cfg)
    start = time.time()
    result = train_head(cfg, seed, data, evaluate_test=evaluate_test)
    return {
        "cfg": config_json(cfg),
        "seed": seed,
        **cfg,
        **result,
        "secs": round(time.time() - start, 1),
        "device": str(DEVICE),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def main(argv=None):
    return run_cli(
        argv,
        stages=STAGES,
        output_dir=OUT,
        experiment="smiles_encoder",
        encoder=lambda cfg: "smiles_encoder",
        normalize_config=normalize_config,
        run=run,
    )


if __name__ == "__main__":
    main()
