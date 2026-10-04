"""
Staged hyperparameter sweep for molecular graph encoder (GNNEncoder).

Uses the same BACE1 data and scaffold split strategy as the ECFP4 encoder
sweep. Molecules are featurized through pipeline.preprocess.featurize_many(),
which generates standardized molecular graphs plus ECFP and SMILES modalities.

Each (config, seed) run adds a row to:

   outputs/encoders/GNN/sweep/<stage>.csv

and a line to:

   outputs/encoders/GNN/sweep/<stage>.log

Runs already present in the CSV are skipped.
"""

import copy
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn
from torch_geometric.loader import DataLoader

from encoders.gnn_encoder import GNNEncoder
from evaluation import evaluation_metrics
from pipeline import bioactivity_loader as bl
from pipeline.cache import DATA_DIR
from pipeline.preprocess import featurize_many

OUT = DATA_DIR.parent / "outputs" / "encoders" / "GNN" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load and featurize data

df = bl.load_bace1()
graphs, failed = featurize_many(df["smiles"], 2048)
df = df.drop(failed).reset_index(drop=True)
y = torch.tensor(
    df["pIC50"].to_numpy(),
    dtype=torch.float32,
)
active = df["active"].to_numpy()

# Attach graph-level regression targets to PyTorch Geometric Data objects for batching
for graph, target in zip(graphs, y):
    graph.y = target.unsqueeze(0)


# Scaffold split

# Keep whole molecular scaffolds within one split.
scaffolds = df["smiles"].map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s))
train, val, test = [], [], []
for idx in sorted(
    df.groupby(scaffolds).indices.values(),
    key=lambda i: (-len(i), i[0]),
):
    if len(train) + len(idx) <= 0.8 * len(df):
        train += idx.tolist()

    elif len(val) + len(idx) <= 0.1 * len(df):
        val += idx.tolist()

    else:
        test += idx.tolist()

if not val or not test:
    raise ValueError(
        f"scaffold split left val ({len(val)}) or test ({len(test)}) empty"
    )


# Prediction / scoring


@torch.no_grad()
def predict(model, indices, batch_size):
    loader = DataLoader(
        [graphs[i] for i in indices],
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    predictions = []

    for batch in loader:
        batch = batch.to(DEVICE)

        pred = model(batch).squeeze(-1)

        predictions.append(pred.cpu())

    return torch.cat(predictions)


def score(model, cfg):
    out = {}

    for name, idx in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        pred = predict(
            model,
            idx,
            cfg["BATCH_SIZE"],
        )

        reg = evaluation_metrics(
            y[idx],
            pred,
            "regression",
        )

        cls = evaluation_metrics(
            active[idx],
            pred,
            "classification",
        )

        out |= {
            f"{name}_RMSE": reg["rmse"],
            f"{name}_R2": reg["r2"],
            f"{name}_ROC-AUC": cls["roc_auc"],
        }

    return out


# Training


def train_gnn(cfg, seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)

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

    train_loader = DataLoader(
        [graphs[i] for i in train],
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    best_rmse = float("inf")
    best_state = None
    best_epoch = 0

    early_stop = 15
    epochs_without_improve = 0

    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()

        for batch in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            pred = model(batch).squeeze(-1)

            nn.functional.mse_loss(
                pred,
                batch.y,
            ).backward()

            optimizer.step()

        val_pred = predict(
            model,
            val,
            cfg["BATCH_SIZE"],
        )

        val_rmse = evaluation_metrics(
            y[val],
            val_pred,
            "regression",
        )["rmse"]

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= early_stop:
            break

    model.load_state_dict(best_state)

    return {
        "best_epoch": best_epoch,
        **score(model, cfg),
    }


def grid(base, **values):
    return [base | dict(zip(values, v)) for v in itertools.product(*values.values())]


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
    "LR": 1e-3,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "OPT": "adam",
}

SEEDS = [0, 1, 2]
FRESH_SEEDS = [3, 4, 5, 6, 7]


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
        ],
        [0, 1, 2, 3, 4],
    ),
}


def run(cfg, seed):
    start = time.time()

    result = train_gnn(
        cfg,
        seed,
    )

    return {
        "cfg": json.dumps(cfg),
        "seed": seed,
        **cfg,
        **result,
        "secs": round(
            time.time() - start,
            1,
        ),
    }


if __name__ == "__main__":
    by_number = {name.split("-")[0]: name for name in STAGES}

    names = [by_number.get(arg, arg) for arg in sys.argv[1:]] or list(STAGES)

    if unknown := set(names) - set(STAGES):
        sys.exit(f"unknown stages {sorted(unknown)}, pick from {list(STAGES)}")

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in names:
        configs, seeds = STAGES[name]

        csv = OUT / f"{name}.csv"

        done = (
            set(pd.read_csv(csv)[["cfg", "seed"]].itertuples(index=False))
            if csv.exists()
            else set()
        )

        runs = list(
            itertools.product(
                configs,
                seeds,
            )
        )

        for i, (cfg, seed) in enumerate(
            runs,
            1,
        ):
            if (json.dumps(cfg), seed) in done:
                continue

            row = pd.DataFrame([run(cfg, seed)])

            pd.concat([pd.read_csv(csv), row] if csv.exists() else [row]).to_csv(
                csv,
                index=False,
            )

            line = (
                f"{name} "
                f"[{i}/{len(runs)}] "
                f"val {row['val_RMSE'][0]:.3f} "
                f"{row['secs'][0]}s "
                f"{row['cfg'][0]}"
            )

            print(
                line,
                flush=True,
            )

            with open(
                OUT / f"{name}.log",
                "a",
            ) as log:
                print(
                    line,
                    file=log,
                )
