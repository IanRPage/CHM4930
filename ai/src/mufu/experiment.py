"""
Run MuFu through the shared benchmark.

`make_fit_predict` builds the `fit_predict` that `run_benchmark` calls per seed, reusing
one SMILES encoder so its embedding cache carries across seeds. `DEFAULT_CONFIG` and
`ENCODER` are the script and notebook defaults.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from baseline.benchmark import (
    SEEDS,
    evaluate_split_predictions,
    load_benchmark_split,
    load_split_predictions,
)
from baseline.results import CodeIdentity, code_identity, file_sha256
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.calibration import CalibrationParams, fit_calibration
from mufu.contract import D_MODEL, MODALITY_DROPOUT_P, PIC50_COLUMNS, Encoder
from mufu.data.dataset import MuFuDataset
from mufu.loss.task_loss import dataset_pos_weights
from mufu.mc_dropout import (
    DEFAULT_COVERAGES,
    fit_mc_dropout,
    predict_with_intervals,
)
from mufu.model import MuFu
from mufu.train import (
    TrainConfig,
    cpu_state,
    fit,
    load_non_smiles_state,
    predict,
    predict_logits,
)
from pipeline.bioactivity_loader import PIC50_ACTIVE_THRESHOLD
from pipeline.combined_loader import COMBINED_CSV_PATH

ENCODER = "molformer-ecfp4-gine"
DEFAULT_CONFIG = TrainConfig(
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    epochs=50,
    patience=10,
    min_delta=1e-3,
    sampler="mixed",
    bio_fraction=0.8,
    checkpoint_mode="task_family",
    pos_weight="raw",
    stopping="metric",
    log_var_lr=None,
    task_weights=(1.0, 0.5, 0.5),
    fp_hidden_dims=(512, 256),
    fp_dropout=0.6,
    calibration="platt",
)
RUNTIME_FIELDS = ("device", "num_workers")


def benchmark_params(cfg: TrainConfig, smiles_encoder: Encoder) -> dict:
    name = getattr(smiles_encoder, "model_name", type(smiles_encoder).__name__)
    revision = getattr(smiles_encoder, "revision", None)
    train = {k: v for k, v in asdict(cfg).items() if k not in RUNTIME_FIELDS}
    return {
        **train,
        "d_model": D_MODEL,
        "modality_dropout_p": MODALITY_DROPOUT_P,
        "active_threshold": PIC50_ACTIVE_THRESHOLD,
        "smiles_encoder": f"{name}@{revision}" if revision else name,
        "smiles_freeze": getattr(smiles_encoder, "freeze", None),
    }


# checkpoints from before gnn_hidden_dims was recorded in params
def infer_gnn_hidden_dims(state: Mapping[str, torch.Tensor]) -> tuple[int, ...]:
    dims = []
    while (key := f"graph_encoder.mp_layers.{len(dims)}.nn.0.weight") in state:
        dims.append(state[key].shape[0])
    if not dims:
        raise ValueError("checkpoint has no graph_encoder.mp_layers weights")
    return tuple(dims)


def load_mufu_checkpoint(
    path: Path,
    smiles_encoder: Encoder,
    *,
    device: torch.device | str = "cpu",
) -> tuple[MuFu, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    params = ckpt.get("params", {})
    fp_hidden_dims = tuple(params.get("fp_hidden_dims", (1024, 512)))
    fp_dropout = float(params.get("fp_dropout", 0.2))
    gnn_hidden_dims = tuple(
        params.get("gnn_hidden_dims") or infer_gnn_hidden_dims(ckpt["model"])
    )
    gnn_dropout = float(params.get("gnn_dropout", 0.1))

    model = MuFu(
        smiles_encoder,
        FingerprintEncoder(hidden_dims=fp_hidden_dims, dropout=fp_dropout),
        GNNEncoder(hidden_dims=gnn_hidden_dims, dropout=gnn_dropout),
    ).to(device)
    load_non_smiles_state(model, ckpt["model"])
    model.eval()

    cal_state = ckpt.get("calibration")
    calibration = cal_state and CalibrationParams.from_state_dict(cal_state)
    return model, {
        "family_states": ckpt.get("family_states"),
        "best_epochs": ckpt.get("best_epochs"),
        "calibration": calibration,
        "checkpoint": ckpt,
    }


def make_fit_predict(
    datasets: Mapping[str, MuFuDataset],
    smiles_encoder: Encoder,
    cfg: TrainConfig,
    *,
    run_dir: Path,
    model_name: str = "mufu",
    identity: CodeIdentity | None = None,
    data_path: Path = COMBINED_CSV_PATH,
    verbose: bool = False,
) -> Callable[[pd.DataFrame, int], dict]:
    if any(p.requires_grad for p in smiles_encoder.parameters()):
        raise ValueError("smiles_encoder must be frozen; it's shared across seeds")
    params = benchmark_params(cfg, smiles_encoder)
    if identity is None:
        identity = code_identity()
    checkpoint_dir = run_dir / "checkpoints"
    logs_dir = run_dir / "logs"

    # df unused, datasets already built from it and run_benchmark checks smiles
    def fit_predict(df: pd.DataFrame, seed: int) -> dict:
        ckpt_path = checkpoint_dir / f"{model_name}_seed_{seed}.pt"
        loss_path = logs_dir / f"seed_{seed}_loss.csv"
        for p in (ckpt_path, loss_path):
            if p.exists():
                raise FileExistsError(f"refusing to overwrite write-once file {p}")

        model = MuFu(
            smiles_encoder,
            FingerprintEncoder(hidden_dims=cfg.fp_hidden_dims, dropout=cfg.fp_dropout),
            GNNEncoder(hidden_dims=cfg.gnn_hidden_dims, dropout=cfg.gnn_dropout),
        )
        result = fit(model, datasets, cfg, verbose=verbose)
        if verbose:
            print(
                f"seed {seed}: best epoch {result.best_epochs or result.best_epoch} | "
                f"val loss {result.best_val_loss:.4f}",
                flush=True,
            )

        calibration = None
        if cfg.calibration != "none":
            calibration = fit_calibration(
                cfg.calibration,
                predict_logits(
                    model,
                    datasets["val"],
                    batch_size=cfg.batch_size,
                    family_states=result.family_states,
                ),
                datasets["val"],
                pos_weights=dataset_pos_weights(datasets["train"], mode=cfg.pos_weight),
            )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": cpu_state(model),
                "family_states": result.family_states,
                "best_epochs": result.best_epochs,
                "calibration": calibration and calibration.state_dict(),
                "weighting": cpu_state(result.weighting),
                "epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "seed": int(seed),
                "params": params,
                "base_commit": identity.base_commit,
                "ai_tree": identity.ai_tree,
                "patch_sha256": identity.patch_sha256,
                "data_sha256": file_sha256(data_path),
            },
            ckpt_path,
        )

        conformal = fit_mc_dropout(
            model, datasets["val"], ckpt_path, batch_size=cfg.batch_size
        )

        logs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "epoch": range(1, len(result.train_losses) + 1),
                "train_loss": result.train_losses,
                "val_loss": result.val_losses,
                "val_pic50_loss": result.val_task_losses["pic50"],
                "val_bio_cls_loss": result.val_task_losses["bio_cls"],
                "val_tox_loss": result.val_task_losses["tox"],
            }
        ).to_csv(loss_path, index=False, lineterminator="\n")

        val_preds = predict(
            model,
            datasets["val"],
            batch_size=cfg.batch_size,
            family_states=result.family_states,
            calibration=calibration,
        )
        test_preds = dict(
            predict(
                model,
                datasets["test"],
                batch_size=cfg.batch_size,
                family_states=result.family_states,
                calibration=calibration,
            )
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            test_intervals = predict_with_intervals(
                model, datasets["test"], conformal, batch_size=cfg.batch_size
            )
        for key, val in test_intervals.items():
            if key not in test_preds:
                test_preds[key] = val

        return {"val": val_preds, "test": test_preds}

    return fit_predict


def coverage_from_predictions_df(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
) -> pd.DataFrame:
    """
    Compute interval empirical coverage and mean width on `test_df` directly from a
    saved prediction DataFrame containing `<col>_lo<pct>` and `<col>_hi<pct>` columns.
    """
    rows = []
    for col in PIC50_COLUMNS:
        target = col.removeprefix("pIC50_")
        mask = test_df[col].notna().to_numpy()
        y = test_df.loc[mask, col].to_numpy(dtype=float)
        for cov in coverages:
            pct = round(cov * 100)
            lo = pred_df.loc[mask, f"{col}_lo{pct}"].to_numpy(dtype=float)
            hi = pred_df.loc[mask, f"{col}_hi{pct}"].to_numpy(dtype=float)
            rows.append(
                {
                    "target": target,
                    "nominal": cov,
                    "empirical": (
                        float(((y >= lo) & (y <= hi)).mean())
                        if len(y)
                        else float("nan")
                    ),
                    "n": len(y),
                    "mean_width": float((hi - lo).mean()) if len(y) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def score_mufu_run_test(
    run_dir: Path,
    df: pd.DataFrame | None = None,
) -> None:
    """
    Recompute derived test-set metrics and interval coverage in `run_dir` from saved
    `predictions/*.csv` files.
    """
    meta = json.loads((run_dir / "run.json").read_text())
    split_id = str(meta["split_id"])
    model_name = str(meta.get("model", "mufu"))
    seeds = [int(s) for s in meta.get("seeds", SEEDS)]
    if df is None:
        df = load_benchmark_split(split_id=split_id)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    # 1. Per-seed MC-Dropout test coverage (if interval columns are present in test_seed_*.csv)
    mc_rows = []
    for seed in seeds:
        pred_path = run_dir / "predictions" / f"test_seed_{seed}.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path, float_precision="round_trip")
        if all(f"{col}_lo90" in pred_df.columns for col in PIC50_COLUMNS):
            cov_df = coverage_from_predictions_df(pred_df, test_df)
            cov_df.insert(0, "seed", seed)
            mc_rows.append(cov_df)
    if mc_rows:
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(mc_rows, ignore_index=True).to_csv(
            logs_dir / "mc_dropout_coverage.csv", index=False, lineterminator="\n"
        )

    # 2. Ensemble test metrics & coverage (if test_ensemble.csv is present)
    ens_pred_path = run_dir / "predictions" / "test_ensemble.csv"
    ens_sidecar = run_dir / "checkpoints" / f"{model_name}_ensemble.json"
    if ens_pred_path.exists() and ens_sidecar.exists():
        ens_preds = load_split_predictions(ens_pred_path)
        ens_metrics = pd.DataFrame(
            evaluate_split_predictions(df, {"test": ens_preds}, n_boot=0)
        )
        ens_metrics["ensemble_sha256"] = file_sha256(ens_sidecar)
        ens_metrics.to_csv(
            run_dir / "ensemble_metrics.csv", index=False, lineterminator="\n"
        )

        ens_df = pd.read_csv(ens_pred_path, float_precision="round_trip")
        if all(f"{col}_lo90" in ens_df.columns for col in PIC50_COLUMNS):
            ens_cov = coverage_from_predictions_df(ens_df, test_df)
            ens_cov.to_csv(
                run_dir / "ensemble_coverage.csv", index=False, lineterminator="\n"
            )
