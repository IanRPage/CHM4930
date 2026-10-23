"""
Staged FingerprintEncoder experiments on the shared frozen BACE-1 cohort.

Stage CSVs/logs retain their existing paths. Exploration scores train/val only;
test is reserved for a selected canonical seed-42 run. Legacy stage files without
matching benchmark metadata cannot be resumed.

    python scripts/ECFP4_encoder_sweep.py 3 4
    python scripts/ECFP4_encoder_sweep.py --benchmark --config chosen.json --dry-run
    python scripts/ECFP4_encoder_sweep.py --benchmark --config chosen.json --write-results

Stages can still be run independently, including with NVIDIA MPS.
"""

import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader

from benchmark import config_json, evaluate_predictions, set_training_seed
from encoders.benchmark import grid, run_cli
from encoders.fingerprint_encoder import FingerprintEncoder
from featurize.fingerprint import FINGERPRINT_BITS
from pipeline.cache import DATA_DIR

OUT = DATA_DIR.parent / "outputs" / "encoders" / "ECFP4" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mlp(cfg, seed, data, *, evaluate_test=False):
    train, val = data.indices["train"], data.indices["val"]
    y = data.y
    X = data.fingerprints(cfg["N_BITS"]).to(DEVICE)
    X_train, y_train = X[train], y[train].to(DEVICE)
    generator = set_training_seed(seed)
    encoder = FingerprintEncoder(
        cfg["N_BITS"],
        cfg["HIDDEN_DIMS"],
        cfg["DROPOUT"],
        final_activation=cfg["FINAL_ACTIVATION"],
    )
    head = nn.Linear(encoder.out_dim, 1)
    nn.init.constant_(head.bias, y[train].mean().item())
    model = nn.Sequential(encoder, head).to(DEVICE)
    optim = torch.optim.AdamW if cfg["OPT"] == "adamw" else torch.optim.Adam
    optimizer = optim(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    loader = DataLoader(
        range(len(train)),
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        generator=generator,
    )

    @torch.no_grad()
    def predict(idx):
        model.eval()
        return model(X[idx]).squeeze(-1).cpu()

    best_rmse, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        for idx in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(
                model(X_train[idx]).squeeze(-1), y_train[idx]
            ).backward()
            optimizer.step()
        rmse = evaluate_predictions(
            data.frame.iloc[val]["pIC50_BACE1"], predict(val), "regression"
        )["rmse"]
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = (
                rmse,
                copy.deepcopy(model.state_dict()),
                epoch,
            )

    if best_state is None:
        raise ValueError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        **data.score(predict, evaluate_test=evaluate_test),
    }


def fit_baseline(cfg, seed, data, *, evaluate_test=False):
    train = data.indices["train"]
    X = data.fingerprints(cfg["N_BITS"]).numpy()
    if cfg["MODEL"] == "ridge":
        model = Ridge(alpha=cfg["ALPHA"])
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_features=cfg["MAX_FEATURES"],
            n_jobs=-1,
            random_state=seed,
        )
    model.fit(X[train], data.y[train].numpy())
    return data.score(lambda idx: model.predict(X[idx]), evaluate_test=evaluate_test)


DEFAULT = {
    "N_BITS": FINGERPRINT_BITS,
    "HIDDEN_DIMS": (1024, 512),
    "DROPOUT": 0.2,
    "FINAL_ACTIVATION": True,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-5,
    "BATCH_SIZE": 64,
    "EPOCHS": 50,
    "OPT": "adam",
}
PICK = DEFAULT | {
    "HIDDEN_DIMS": (512, 256),
    "DROPOUT": 0.6,
    "WEIGHT_DECAY": 1e-3,
    "EPOCHS": 100,
}
SEEDS, FRESH_SEEDS = [0, 1, 2], [3, 4, 5, 6, 7]

# stages (configs, seeds) in run order. holdout-seed stages rerun top picks on seeds no earlier stage chose
STAGES = {
    "1-regularization": (
        grid(
            DEFAULT,
            DROPOUT=[0.2, 0.4, 0.6],
            WEIGHT_DECAY=[1e-5, 1e-4, 1e-3],
            HIDDEN_DIMS=[(1024, 512), (512, 256), (256,)],
        ),
        SEEDS,
    ),
    "2-regularization-wider": (
        grid(
            DEFAULT | {"EPOCHS": 100},
            DROPOUT=[0.6, 0.7],
            HIDDEN_DIMS=[(1024, 512), (512, 256)],
            WEIGHT_DECAY=[1e-3, 3e-3, 1e-2],
        )
        + grid(
            DEFAULT | {"EPOCHS": 100, "OPT": "adamw"},
            DROPOUT=[0.6, 0.7],
            HIDDEN_DIMS=[(1024, 512), (512, 256)],
            WEIGHT_DECAY=[1e-2, 3e-2, 1e-1],
        ),
        SEEDS,
    ),
    "3-training-dynamics": (
        grid(
            PICK, LR=[3e-4, 1e-3, 3e-3], BATCH_SIZE=[32, 64, 128], N_BITS=[1024, 2048]
        ),
        SEEDS,
    ),
    "4-holdout-seeds": (
        [
            DEFAULT,
            PICK,
            PICK | {"HIDDEN_DIMS": (1024, 512), "DROPOUT": 0.7, "WEIGHT_DECAY": 3e-3},
            PICK | {"HIDDEN_DIMS": (1024, 512)},
            PICK | {"LR": 3e-3, "BATCH_SIZE": 128},
        ],
        FRESH_SEEDS,
    ),
    "5-final-activation": (
        grid(PICK, FINAL_ACTIVATION=[True, False], DROPOUT=[0.5, 0.6, 0.7]),
        SEEDS,
    ),
    "6-final-activation-holdout-seeds": (
        [
            PICK | {"FINAL_ACTIVATION": False},
            PICK | {"FINAL_ACTIVATION": False, "DROPOUT": 0.7},
            PICK | {"FINAL_ACTIVATION": False, "DROPOUT": 0.7, "EPOCHS": 150},
        ],
        FRESH_SEEDS,
    ),
    "7-baselines": (
        grid({"N_BITS": 2048, "MODEL": "ridge"}, ALPHA=[10, 30, 100, 300])
        + grid(
            {"N_BITS": 2048, "MODEL": "rf"}, MAX_FEATURES=["sqrt", "log2", 0.1, 0.3]
        ),
        FRESH_SEEDS,
    ),
}


def run(cfg, seed, data, *, evaluate_test=False):
    start = time.time()
    res = (fit_baseline if "MODEL" in cfg else train_mlp)(
        cfg, seed, data, evaluate_test=evaluate_test
    )
    return {
        "cfg": config_json(cfg),
        "seed": seed,
        **cfg,
        **res,
        "secs": round(time.time() - start, 1),
        "device": str(DEVICE),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def normalize_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT):
        raise ValueError(
            "canonical ECFP4 configuration must contain only encoder training keys"
        )
    return DEFAULT | config


def main(argv=None):
    return run_cli(
        argv,
        stages=STAGES,
        output_dir=OUT,
        experiment="ecfp4_encoder",
        encoder=lambda cfg: f"fingerprint_encoder-{cfg['N_BITS']}",
        normalize_config=normalize_config,
        run=run,
    )


if __name__ == "__main__":
    main()
