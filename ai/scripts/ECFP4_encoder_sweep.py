"""
Staged hyperparameter sweep for ECFP4 encoder (FingerprintEncoder)

Same data, scaffold split, and training as ai/notebooks/ECFP4-encoder.ipynb.
Each (config, seed) run adds a row to outputs/encoders/ECFP4/sweep/<stage>.csv
and a line to <stage>.log next to it, and runs already there are skipped. Pick
by mean val_RMSE across seeds (test only for reporting).
notebooks/ECFP4-encoder-sweep-results.ipynb plots the CSVs generated from this
script

    python scripts/ECFP4_encoder_sweep.py  # every stage, in order
    python scripts/ECFP4_encoder_sweep.py 3 4  # just these, by number or full name

One run can't fill the GPU, and stages are independent, so we can speed each
stage's run using NVIDIA MPS (about ~2.5x faster):

    nvidia-cuda-mps-control -d
    for s in 1 2 3 4 5 6 7; do
        python scripts/ECFP4_encoder_sweep.py $s &
    done; wait
    echo quit | nvidia-cuda-mps-control
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader

from encoders.fingerprint_encoder import FingerprintEncoder
from evaluation import evaluation_metrics
from pipeline import bioactivity_loader as bl
from pipeline.cache import DATA_DIR
from pipeline.preprocess import featurize_many

OUT = DATA_DIR.parent / "outputs" / "encoders" / "ECFP4" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = bl.load_bace1()
fps = {}
data, failed = featurize_many(df["smiles"], 2048)
df = df.drop(failed).reset_index(drop=True)
fps[2048] = torch.cat([d.fp for d in data])
y = torch.tensor(df["pIC50"].to_numpy(), dtype=torch.float32)
active = df["active"].to_numpy()

# whole scaffolds go to one split, largest first
scaffolds = df["smiles"].map(lambda s: MurckoScaffold.MurckoScaffoldSmiles(smiles=s))
train, val, test = [], [], []
for idx in sorted(
    df.groupby(scaffolds).indices.values(), key=lambda i: (-len(i), i[0])
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


def fingerprints(n_bits):
    if n_bits not in fps:
        fps[n_bits] = torch.cat([d.fp for d in featurize_many(df["smiles"], n_bits)[0]])
    return fps[n_bits]


def score(predict):
    out = {}
    for name, idx in [("train", train), ("val", val), ("test", test)]:
        pred = predict(idx)
        reg = evaluation_metrics(y[idx], pred, "regression")
        cls = evaluation_metrics(active[idx], pred, "classification")
        out |= {
            f"{name}_RMSE": reg["rmse"],
            f"{name}_R2": reg["r2"],
            f"{name}_ROC-AUC": cls["roc_auc"],
        }
    return out


def train_mlp(cfg, seed):
    X = fingerprints(cfg["N_BITS"]).to(DEVICE)
    X_train, y_train = X[train], y[train].to(DEVICE)
    torch.manual_seed(seed)
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
        generator=torch.Generator().manual_seed(seed),
    )

    @torch.no_grad()
    def predict(idx):
        model.eval()
        return model(X[idx]).squeeze(-1).cpu()

    # keep best val epoch's weights
    best_rmse, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, cfg["EPOCHS"] + 1):
        model.train()
        for idx in loader:
            optimizer.zero_grad()
            nn.functional.mse_loss(
                model(X_train[idx]).squeeze(-1), y_train[idx]
            ).backward()
            optimizer.step()
        rmse = evaluation_metrics(y[val], predict(val), "regression")["rmse"]
        if rmse < best_rmse:
            best_rmse, best_state, best_epoch = (
                rmse,
                copy.deepcopy(model.state_dict()),
                epoch,
            )

    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, **score(predict)}


def fit_baseline(cfg, seed):
    X = fingerprints(cfg["N_BITS"]).numpy()
    if cfg["MODEL"] == "ridge":
        model = Ridge(alpha=cfg["ALPHA"])
    else:
        model = RandomForestRegressor(
            n_estimators=500,
            max_features=cfg["MAX_FEATURES"],
            n_jobs=-1,
            random_state=seed,
        )
    model.fit(X[train], y[train].numpy())
    return score(lambda idx: model.predict(X[idx]))


def grid(base, **values):
    return [base | dict(zip(values, v)) for v in itertools.product(*values.values())]


DEFAULT = {
    "N_BITS": 2048,
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


def run(cfg, seed):
    start = time.time()
    res = (fit_baseline if "MODEL" in cfg else train_mlp)(cfg, seed)
    return {
        "cfg": json.dumps(cfg),
        "seed": seed,
        **cfg,
        **res,
        "secs": round(time.time() - start, 1),
    }


if __name__ == "__main__":
    by_number = {name.split("-")[0]: name for name in STAGES}
    names = [by_number.get(arg, arg) for arg in sys.argv[1:]] or list(STAGES)
    if unknown := set(names) - set(STAGES):
        sys.exit(f"unknown stages {sorted(unknown)}, pick from {list(STAGES)}")
    OUT.mkdir(parents=True, exist_ok=True)

    for name in names:
        configs, seeds = STAGES[name]
        csv = OUT / f"{name}.csv"
        done = (
            set(pd.read_csv(csv)[["cfg", "seed"]].itertuples(index=False))
            if csv.exists()
            else set()
        )
        runs = list(itertools.product(configs, seeds))
        for i, (cfg, seed) in enumerate(runs, 1):
            if (json.dumps(cfg), seed) in done:
                continue
            row = pd.DataFrame([run(cfg, seed)])
            pd.concat([pd.read_csv(csv), row] if csv.exists() else [row]).to_csv(
                csv, index=False
            )
            line = f"{name} [{i}/{len(runs)}] val {row['val_RMSE'][0]:.3f} {row['secs'][0]}s {row['cfg'][0]}"
            print(line, flush=True)
            with open(OUT / f"{name}.log", "a") as log:
                print(line, file=log)
