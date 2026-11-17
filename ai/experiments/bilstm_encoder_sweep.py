"""
biLSTM sweep using the shared encoder benchmark.

from /ai:
    python experiments/bilstm_encoder_sweep.py 1 #initial test
    python experiments/bilstm_encoder_sweep.py 2 #architecture stage
    python experiments/bilstm_encoder_sweep.py 3 #other exploratory stages

"""

import copy
import hashlib
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.benchmark import params_json, seed_everything
from encoder_benchmark import run_cli, validation_rmse
from encoders.bilstm_smiles_encoder import (
    UNK_TOKEN,
    biLSTMEncoder,
    build_vocabulary,
    tokenize_smiles,
)
from pipeline.cache import DATA_DIR

OUT = DATA_DIR.parent / "outputs" / "encoders" / "BiLSTM" / "sweep"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT = "bilstm_encoder"

DEFAULT = {
    "EMBEDDING_DIM": 128,
    "HIDDEN_DIM": 128,
    "NUM_LAYERS": 2,
    "DROPOUT": 0.2,
    "LR": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "PATIENCE": 15,
    "GRAD_CLIP": 1.0,
    "OPT": "adamw",
}


# fixed settings carried forward between sequential tuning stages
# update these after reviewing the preceding stage's validation results


SEARCH_BASE = DEFAULT | {
    "EPOCHS": 40,
    "PATIENCE": 8,
}

# each tuple: embedding_dim, hidden_dim, layers, dropout,
#             learning_rate, batch_size
JOINT_CONFIGS = [
    (64, 128, 2, 0.2, 1e-3, 64),  # best candidate from previous sweep
    (128, 128, 2, 0.2, 1e-3, 64),
    (64, 64, 2, 0.2, 1e-3, 64),
    (64, 256, 2, 0.3, 3e-4, 64),
    (128, 256, 2, 0.3, 3e-4, 64),
    (128, 128, 1, 0.1, 1e-3, 64),
    (64, 128, 1, 0.2, 3e-4, 32),
    (128, 128, 2, 0.3, 3e-4, 32),
    (64, 128, 2, 0.1, 3e-3, 64),
    (128, 128, 2, 0.2, 3e-4, 128),
    (64, 256, 1, 0.2, 1e-3, 128),
    (128, 64, 2, 0.3, 1e-3, 32),
]

joint_search = [
    SEARCH_BASE
    | {
        "EMBEDDING_DIM": embedding,
        "HIDDEN_DIM": hidden,
        "NUM_LAYERS": layers,
        "DROPOUT": dropout,
        "LR": lr,
        "BATCH_SIZE": batch,
    }
    for embedding, hidden, layers, dropout, lr, batch in JOINT_CONFIGS
]

STAGES = {
    "1-smoke": (
        [SEARCH_BASE | {"EPOCHS": 2, "PATIENCE": 2}],
        [0],
    ),
    "2-joint-search": (joint_search, [0]),
}


def normalize_config(config):
    """Apply defaults and validate sweep settings"""
    cfg = DEFAULT | config

    if set(config) - set(DEFAULT):
        raise ValueError("Unknown configuration parameters")

    for key in (
        "EMBEDDING_DIM",
        "HIDDEN_DIM",
        "NUM_LAYERS",
        "BATCH_SIZE",
        "EPOCHS",
        "PATIENCE",
    ):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")

    if not 0 <= cfg["DROPOUT"] < 1:
        raise ValueError("DROPOUT must be between 0 and 1")

    if cfg["LR"] <= 0 or cfg["WEIGHT_DECAY"] < 0 or cfg["GRAD_CLIP"] <= 0:
        raise ValueError("Invalid optimizer parameters")

    if cfg["OPT"] not in ("adam", "adamw"):
        raise ValueError("OPT must be adam or adamw")

    for key in ("LR", "WEIGHT_DECAY", "GRAD_CLIP"):
        if not math.isfinite(cfg[key]):
            raise ValueError(f"{key} must be finite")

    return cfg


def save_epoch_history(cfg, seed, data, history):
    """Save per-epoch metrics for one completed training run"""
    history_dir = OUT / "epoch_history"
    history_dir.mkdir(parents=True, exist_ok=True)

    # identify the run using its configuration, seed, and benchmark data
    run_info = f"{params_json(cfg)}:{seed}:{data.data_sha256}:{data.split_sha256}"
    run_id = hashlib.sha256(run_info.encode("utf-8")).hexdigest()[:16]
    path = history_dir / f"run_{run_id}.csv"

    records = pd.DataFrame(history)
    records["cfg"] = params_json(cfg)
    records["seed"] = seed
    records["data_sha256"] = data.data_sha256
    records["split_sha256"] = data.split_sha256

    # atomic replacement prevents a partially written final CSV
    temp_path = path.with_suffix(".tmp")
    try:
        records.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)

    return path


def train_bilstm(cfg, seed, data, *, evaluate_test=False):
    """Train encoder & linear head, select epoch by validation RMSE"""
    cfg = normalize_config(cfg)
    generator = seed_everything(seed)
    train = data.indices["train"]
    val = data.indices["val"]

    smiles = data.frame["smiles"].tolist()
    vocab = build_vocabulary([smiles[int(i)] for i in train])

    # prepare token ID tensors once, instead of tokenizing on every epoch
    # the vocabulary is built exclusively from the training split
    tokenized_smiles = [
        torch.tensor(
            [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize_smiles(smi)],
            dtype=torch.long,
        )
        for smi in smiles
    ]
    vocab_coverage = {
        name: sum(
            token not in vocab
            for i in data.indices[name]
            for token in tokenize_smiles(smiles[int(i)])
        )
        / max(
            1,
            sum(len(tokenize_smiles(smiles[int(i)])) for i in data.indices[name]),
        )
        for name in ("train", "val", "test")
        if name != "test" or evaluate_test
    }

    encoder = biLSTMEncoder(
        vocab=vocab,
        embedding_dim=cfg["EMBEDDING_DIM"],
        hidden_dim=cfg["HIDDEN_DIM"],
        num_layers=cfg["NUM_LAYERS"],
        dropout=cfg["DROPOUT"],
    )

    head = nn.Linear(encoder.out_dim, 1)
    y = data.y
    nn.init.constant_(head.bias, y[train].mean().item())

    class Model(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, sequences):
            return self.head(self.encoder.forward_token_ids(sequences)).squeeze(-1)

    model = Model(encoder, head).to(DEVICE)
    optimizer_type = torch.optim.AdamW if cfg["OPT"] == "adamw" else torch.optim.Adam
    optimizer = optimizer_type(
        model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"]
    )
    loader = DataLoader(
        list(map(int, train)),
        batch_size=cfg["BATCH_SIZE"],
        shuffle=True,
        generator=generator,
    )

    @torch.no_grad()
    def predict(indices):
        model.eval()
        predictions = []
        for start in range(0, len(indices), cfg["BATCH_SIZE"]):
            idx = indices[start : start + cfg["BATCH_SIZE"]]
            predictions.append(model([tokenized_smiles[int(i)] for i in idx]).cpu())
        return torch.cat(predictions)

    best_rmse, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []

    for epoch in range(1, cfg["EPOCHS"] + 1):
        epoch_start = time.perf_counter()
        model.train()

        total_loss = 0.0
        total_samples = 0

        for idx in loader:
            idx_list = idx.tolist()
            target = y[idx].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            prediction = model([tokenized_smiles[i] for i in idx_list])

            loss = nn.functional.mse_loss(prediction, target)

            if not torch.isfinite(loss):
                raise ValueError(f"non-finite training loss at epoch {epoch}")

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg["GRAD_CLIP"],
                error_if_nonfinite=True,
            )
            optimizer.step()

            # accumulate the existing batch losses without
            # running another full training-set evaluation
            batch_size = len(idx_list)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples
        rmse = validation_rmse(data, predict(val))

        if rmse < best_rmse:
            best_rmse = rmse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        epoch_secs = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_RMSE": rmse,
                "epoch_secs": round(epoch_secs, 2),
                "best_val_RMSE": best_rmse,
            }
        )

        print(
            f"Epoch {epoch}/{cfg['EPOCHS']} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val RMSE: {rmse:.4f} | "
            f"Time: {epoch_secs:.1f}s",
            flush=True,
        )

        if stale >= cfg["PATIENCE"]:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise ValueError("training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    result = {
        "best_epoch": best_epoch,
        "vocab_size": len(vocab),
        "train_unk_fraction": vocab_coverage["train"],
        "val_unk_fraction": vocab_coverage["val"],
        **({"test_unk_fraction": vocab_coverage["test"]} if evaluate_test else {}),
        **data.score(
            predict,
            evaluate_test=evaluate_test,
        ),
        "epochs_trained": len(history),
    }
    return result, history


def run(cfg, seed, data, *, evaluate_test=False):
    """Run one BiLSTM experiment and save its epoch history."""
    cfg = normalize_config(cfg)
    start = time.time()

    result, history = train_bilstm(
        cfg,
        seed,
        data,
        evaluate_test=evaluate_test,
    )

    history_path = save_epoch_history(cfg, seed, data, history)

    return {
        "cfg": params_json(cfg),
        "seed": seed,
        **cfg,
        **result,
        "secs": round(time.time() - start, 1),
        "device": str(DEVICE),
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "history_file": str(history_path),
    }


def main(argv=None):
    return run_cli(
        argv,
        stages=STAGES,
        output_dir=OUT,
        experiment=EXPERIMENT,
        normalize_config=normalize_config,
        run=run,
    )


if __name__ == "__main__":
    main()
