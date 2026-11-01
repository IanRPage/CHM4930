"""
Staged GNNEncoder experiments on the shared frozen BACE-1 cohort.

    python experiments/gnn_encoder_sweep.py 1
"""

import copy
import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from torch.utils.data import BatchSampler, RandomSampler
from torch_geometric.loader import DataLoader

from baseline.benchmark import (
    evaluate_split_predictions,
    params_json,
    seed_everything,
)
from encoder_benchmark import grid, run_cli, validation_rmse
from encoders.gnn_encoder import GNNEncoder
from pipeline.cache import DATA_DIR

OUT = DATA_DIR.parent / "outputs" / "encoders" / "GNN" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingBatchSampler(BatchSampler):
    """Keep every training example, merging a singleton tail for BatchNorm."""

    def __iter__(self):
        previous = None
        for batch in super().__iter__():
            if previous is not None:
                if len(batch) == 1:
                    yield previous + batch
                    return
                yield previous
            previous = batch
        if previous is not None:
            yield previous

    def __len__(self):
        count = super().__len__()
        return count - int(
            not self.drop_last
            and count > 1
            and len(self.sampler) % self.batch_size == 1
        )


@torch.no_grad()
def predict(model, indices, batch_size, graphs):
    loader = DataLoader(
        [graphs[i] for i in indices],
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions = []
    for batch in loader:
        batch = batch.to(DEVICE)
        predictions.append(model(batch).squeeze(-1).cpu())
    return torch.cat(predictions)


@contextmanager
def _deterministic_training():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@_deterministic_training()
def train_gnn(cfg, seed, data, *, evaluate_test=False):
    generator = seed_everything(seed)
    train, val = data.indices["train"], data.indices["val"]
    if cfg["BATCH_SIZE"] < 2 or len(train) < 2:
        raise ValueError(
            "GNN training needs a batch size and training cohort of at least 2"
        )
    y, graphs = data.y, data.graphs()
    encoder = GNNEncoder(
        hidden_dims=cfg["HIDDEN_DIMS"],
        dropout=cfg["DROPOUT"],
    )
    head = nn.Linear(encoder.out_dim, 1)
    nn.init.constant_(head.bias, y[train].mean().item())
    model = nn.Sequential(encoder, head).to(DEVICE)
    optimizer_cls = torch.optim.AdamW if cfg["OPT"] == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
    )
    train_graphs = [graphs[i] for i in train]
    train_loader = DataLoader(
        train_graphs,
        batch_sampler=TrainingBatchSampler(
            RandomSampler(train_graphs, generator=generator),
            cfg["BATCH_SIZE"],
            drop_last=False,
        ),
        generator=generator,
    )
    best_rmse, best_state, best_epoch = float("inf"), None, 0
    early_stop = 15
    epochs_without_improve = 0
    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(batch).squeeze(-1)
            nn.functional.mse_loss(pred, batch.y).backward()
            optimizer.step()
        val_pred = predict(model, val, cfg["BATCH_SIZE"], graphs)
        val_rmse = validation_rmse(data, val_pred, evaluate=evaluate_split_predictions)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
        if epochs_without_improve >= early_stop:
            break
    if best_state is None:
        raise ValueError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        **data.score(
            lambda idx: predict(model, idx, cfg["BATCH_SIZE"], graphs),
            evaluate_test=evaluate_test,
        ),
    }


DEFAULT = {
    "HIDDEN_DIMS": (256, 128),
    "DROPOUT": 0.2,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 128,
    "EPOCHS": 100,
    "OPT": "adam",
}

PICK = DEFAULT | {
    "HIDDEN_DIMS": (256, 256, 128),
    "DROPOUT": 0.1,
    "WEIGHT_DECAY": 1e-3,
    "LR": 1e-3,
    "BATCH_SIZE": 32,
    "OPT": "adam",
    "EPOCHS": 100,
}

FINAL_BASE = {
    "DROPOUT": 0.1,
    "LR": 3e-4,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "OPT": "adamw",
}

FINAL = FINAL_BASE | {
    "HIDDEN_DIMS": (256, 256, 128),
    "WEIGHT_DECAY": 1e-4,
}

SEEDS = [0, 1, 2]
FRESH_SEEDS = [3, 4, 5, 6, 7]
FINAL_SEEDS = list(range(8, 18))


# Stages

STAGES = {
    "1-architecture": (
        grid(
            DEFAULT,
            HIDDEN_DIMS=[
                (128,),
                (128, 128),
                (256, 128),
                (256, 256),
                (512, 256),
                (256, 256, 128),
            ],
        ),
        SEEDS,
    ),
    "2-regularization": (
        grid(
            PICK,
            DROPOUT=[
                0.1,
                0.2,
                0.4,
            ],
            WEIGHT_DECAY=[
                0.0,
                1e-4,
                1e-3,
            ],
        ),
        SEEDS,
    ),
    "3-training-dynamics": (
        grid(
            PICK,
            LR=[
                3e-4,
                1e-3,
            ],
            BATCH_SIZE=[
                32,
                64,
            ],
        ),
        SEEDS,
    ),
    "4-optimizer": (
        grid(
            PICK,
            OPT=[
                "adam",
                "adamw",
            ],
            WEIGHT_DECAY=[
                1e-4,
                1e-3,
            ],
        ),
        SEEDS,
    ),
    "5-holdout-seeds": (
        [
            PICK,
            PICK
            | {
                "DROPOUT": 0.2,
            },
            PICK
            | {
                "HIDDEN_DIMS": (128, 128),
            },
            PICK
            | {
                "BATCH_SIZE": 64,
            },
            PICK
            | {
                "OPT": "adamw",
            },
        ],
        FRESH_SEEDS,
    ),
    "6-final-check": (
        [
            FINAL_BASE
            | {
                "HIDDEN_DIMS": (128, 128),
                "WEIGHT_DECAY": 0.0,
            },
            FINAL_BASE
            | {
                "HIDDEN_DIMS": (128, 128),
                "WEIGHT_DECAY": 1e-4,
            },
            FINAL_BASE
            | {
                "HIDDEN_DIMS": (256, 256, 128),
                "WEIGHT_DECAY": 0.0,
            },
            FINAL_BASE
            | {
                "HIDDEN_DIMS": (256, 256, 128),
                "WEIGHT_DECAY": 1e-4,
            },
            FINAL_BASE
            | {
                "HIDDEN_DIMS": (128, 128),
                "WEIGHT_DECAY": 1e-4,
                "OPT": "adam",
                "LR": 1e-3,
            },
        ],
        FINAL_SEEDS,
    ),
}


def run(cfg, seed, data, *, evaluate_test=False):
    start = time.time()
    result = train_gnn(cfg, seed, data, evaluate_test=evaluate_test)
    return {
        "cfg": params_json(cfg),
        "seed": seed,
        **cfg,
        **result,
        "secs": round(time.time() - start, 1),
        "device": str(DEVICE),
    }


def normalize_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT):
        raise ValueError(
            "canonical GNN configuration must contain only encoder training keys"
        )
    return DEFAULT | config


def main(argv=None):
    return run_cli(
        argv,
        stages=STAGES,
        output_dir=OUT,
        experiment="gnn_encoder",
        normalize_config=normalize_config,
        run=run,
    )


if __name__ == "__main__":
    main()
