"""
SCRUM-79 3-seed ablations, EGFR diagnostics, and post-hoc calibration checks.

Writes only to `ai/outputs/scrum79/` and `ai/checkpoints/scrum79/`, never to
`results.csv` or the main checkpoints.

Usage (from `ai/`):
    python experiments/sweep_scrum79.py all
    python experiments/sweep_scrum79.py sweep 0 1 2 3 4 5 6
    python experiments/sweep_scrum79.py egfr
    python experiments/sweep_scrum79.py calibrate
    python experiments/sweep_scrum79.py summary
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from baseline.benchmark import (
    evaluate_split_predictions,
    load_benchmark_split,
    run_benchmark,
    seed_everything,
)
from baseline.results import load_results, summarize_seeds
from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.smiles_encoder import SmilesEncoder
from featurize.fingerprint import FINGERPRINT_BITS
from mufu.calibration import (
    CALIBRATION_MODES,
    evaluate_calibration,
    fit_calibration,
)
from mufu.contract import COLUMN_GROUPS
from mufu.data.dataset import MuFuDataset, split_datasets
from mufu.experiment import (
    ENCODER,
    benchmark_params,
    load_mufu_checkpoint,
    make_fit_predict,
)
from mufu.loss.task_loss import dataset_pos_weights
from mufu.train import TrainConfig, predict_logits
from pipeline.bioactivity_loader import BACE1_CSV_PATH, EGFR_CSV_PATH

AI_DIR = Path(__file__).resolve().parents[1]
SWEEP_DIR = AI_DIR / "outputs" / "scrum79"
SWEEP_RESULTS_PATH = SWEEP_DIR / "sweep_results.csv"
SWEEP_CKPT_DIR = AI_DIR / "checkpoints" / "scrum79"
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)
NOISE_FLOOR = 0.006

# keep-rule metrics, direction -1 means lower is better
MAIN_METRICS: tuple[tuple[str, str, str, int], ...] = (
    ("bace1", "pIC50_BACE1", "rmse", -1),
    ("egfr", "pIC50_EGFR", "rmse", -1),
    ("tox21", "macro_tox21", "roc_auc", +1),
)
ITEM_CANDIDATES: dict[str, list[tuple[str, dict]]] = {
    "2": [
        ("mufu_s79_2_pw_cap5", {"pos_weight": "cap5"}),
        ("mufu_s79_2_pw_cap10", {"pos_weight": "cap10"}),
        ("mufu_s79_2_pw_sqrt", {"pos_weight": "sqrt"}),
        ("mufu_s79_2_pw_none", {"pos_weight": "none"}),
    ],
    "3": [
        ("mufu_s79_3_stop_norm_loss", {"stopping": "norm_loss"}),
        (
            "mufu_s79_3_stop_metric",
            {"checkpoint_mode": "task_family", "stopping": "metric"},
        ),
    ],
    "4": [
        ("mufu_s79_4_tw_lr1e2", {"log_var_lr": 1e-2}),
        ("mufu_s79_4_tw_lr5e2", {"log_var_lr": 5e-2}),
        ("mufu_s79_4_tw_fixed_1_025_1", {"task_weights": (1.0, 0.25, 1.0)}),
        ("mufu_s79_4_tw_fixed_1_05_05", {"task_weights": (1.0, 0.5, 0.5)}),
    ],
    "5": [
        ("mufu_s79_5_fp_drop05", {"fp_dropout": 0.5}),
        ("mufu_s79_5_fp_drop06", {"fp_dropout": 0.6}),
        ("mufu_s79_5_fp_512_256", {"fp_hidden_dims": (512, 256)}),
        (
            "mufu_s79_5_fp_512_256_drop06",
            {"fp_hidden_dims": (512, 256), "fp_dropout": 0.6},
        ),
    ],
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# SCRUM-76 config the sweep ran from, with the GNN size from before #51
def base_config() -> TrainConfig:
    return TrainConfig(min_delta=1e-3, gnn_hidden_dims=(128, 128), device=_device())


def get_completed_seeds(
    model_name: str, results_path: Path = SWEEP_RESULTS_PATH
) -> set[int]:
    if not results_path.exists():
        return set()
    df = pd.read_csv(results_path)
    sub = df[df["model"] == model_name]
    return set(sub["seed"].dropna().astype(int).unique())


def run_variant(
    model_name: str,
    cfg: TrainConfig,
    df: pd.DataFrame,
    datasets: dict[str, MuFuDataset],
    smiles_encoder: SmilesEncoder,
    *,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    results_path: Path = SWEEP_RESULTS_PATH,
    checkpoint_dir: Path = SWEEP_CKPT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    done = set() if force else get_completed_seeds(model_name, results_path)
    remaining = tuple(s for s in seeds if s not in done)
    history_dir = SWEEP_DIR / model_name

    if not remaining:
        print(f"[skip] {model_name}: seeds {seeds} already recorded", flush=True)
        all_rows = load_results(results_path)
        return summarize_seeds(all_rows[all_rows["model"] == model_name])

    print(
        f"\n=== Running {model_name} on seeds {remaining} ===\n{cfg}",
        flush=True,
    )
    t0 = time.time()
    fit_predict = make_fit_predict(
        datasets,
        smiles_encoder,
        cfg,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        history_dir=history_dir,
        verbose=True,
    )
    summary = run_benchmark(
        fit_predict,
        model=model_name,
        encoder=ENCODER,
        params=benchmark_params(cfg, smiles_encoder),
        seeds=remaining,
        df=df,
        path=results_path,
    )
    elapsed = time.time() - t0
    print(f"=== Finished {model_name} in {elapsed:.1f}s ===", flush=True)
    return summary


def extract_metric_map(
    results_path: Path, model_name: str, split: str = "val"
) -> dict[tuple[str, str, str], tuple[float, float]]:
    df = load_results(results_path)
    sub = df[(df["model"] == model_name) & (df["split"] == split)]
    if sub.empty:
        raise ValueError(f"no {split} results found for {model_name} in {results_path}")
    summary = summarize_seeds(sub)
    return {
        (row["dataset"], row["endpoint"], row["metric"]): (
            float(row["value"]),
            float(row["std"]),
        )
        for _, row in summary.iterrows()
    }


def evaluate_keep_rule(
    results_path: Path,
    candidate_model: str,
    reference_model: str,
    *,
    threshold: float = NOISE_FLOOR,
) -> dict:
    cand = extract_metric_map(results_path, candidate_model, split="val")
    ref = extract_metric_map(results_path, reference_model, split="val")

    improvements: dict[str, float] = {}
    details: dict[str, dict[str, float]] = {}
    for dataset, endpoint, metric, direction in MAIN_METRICS:
        key = (dataset, endpoint, metric)
        cand_val, cand_std = cand[key]
        ref_val, ref_std = ref[key]
        imp = float(direction * (cand_val - ref_val))
        improvements[endpoint] = imp
        details[endpoint] = {
            "candidate_mean": cand_val,
            "candidate_std": cand_std,
            "reference_mean": ref_val,
            "reference_std": ref_std,
            "improvement": imp,
        }

    max_improvement = max(improvements.values())
    min_improvement = min(improvements.values())
    best_endpoint = max(improvements, key=improvements.get)
    worst_endpoint = min(improvements, key=improvements.get)
    keep = (max_improvement > threshold) and (min_improvement >= -threshold)

    return {
        "candidate": candidate_model,
        "reference": reference_model,
        "keep": bool(keep),
        "max_improvement": float(max_improvement),
        "best_endpoint": best_endpoint,
        "min_improvement": float(min_improvement),
        "worst_endpoint": worst_endpoint,
        "improvements": improvements,
        "details": details,
    }


def select_item_winner(
    results_path: Path,
    candidates: list[tuple[str, dict]],
    reference_model: str,
) -> tuple[tuple[str, dict] | None, list[dict]]:
    verdicts = []
    passing: list[tuple[float, str, dict, dict]] = []
    for model_name, overrides in candidates:
        verdict = evaluate_keep_rule(results_path, model_name, reference_model)
        verdict["overrides"] = overrides
        verdicts.append(verdict)
        if verdict["keep"]:
            passing.append((verdict["max_improvement"], model_name, overrides, verdict))

    if not passing:
        return None, verdicts

    passing.sort(key=lambda x: x[0], reverse=True)
    _, best_name, best_overrides, _ = passing[0]
    for v in verdicts:
        v["selected_winner"] = v["candidate"] == best_name
    return (best_name, best_overrides), verdicts


def run_ablation_stages(
    stages: set[str],
    df: pd.DataFrame,
    datasets: dict[str, MuFuDataset],
    smiles_encoder: SmilesEncoder,
    *,
    force: bool = False,
) -> dict:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    base = base_config()

    if "0" in stages:
        run_variant(
            "mufu_s79_0_baseline",
            base,
            df,
            datasets,
            smiles_encoder,
            force=force,
        )

    if "1" in stages:
        run_variant(
            "mufu_s79_1_task_family",
            replace(base, checkpoint_mode="task_family"),
            df,
            datasets,
            smiles_encoder,
            force=force,
        )

    item1_verdict = evaluate_keep_rule(
        SWEEP_RESULTS_PATH, "mufu_s79_1_task_family", "mufu_s79_0_baseline"
    )
    if item1_verdict["keep"]:
        anchor_name = "mufu_s79_1_task_family"
        anchor_overrides = {"checkpoint_mode": "task_family"}
    else:
        anchor_name = "mufu_s79_0_baseline"
        anchor_overrides = {}
    anchor_cfg = replace(base, **anchor_overrides)
    print(
        f"\n[Anchor Selection] Item 1 keep={item1_verdict['keep']} "
        f"(improvements={item1_verdict['improvements']}) -> Anchor: {anchor_name}",
        flush=True,
    )

    for stage_id, candidates in ITEM_CANDIDATES.items():
        if stage_id not in stages:
            continue
        for model_name, overrides in candidates:
            run_variant(
                model_name,
                replace(anchor_cfg, **overrides),
                df,
                datasets,
                smiles_encoder,
                force=force,
            )

    kept_overrides = dict(anchor_overrides)
    all_verdicts: dict[str, object] = {"item1": item1_verdict}
    for stage_id, candidates in ITEM_CANDIDATES.items():
        completed = all(
            get_completed_seeds(m, SWEEP_RESULTS_PATH) >= set(SWEEP_SEEDS)
            for m, _ in candidates
        )
        if not completed:
            continue
        winner, verdicts = select_item_winner(
            SWEEP_RESULTS_PATH, candidates, anchor_name
        )
        if stage_id == "3":
            all_verdicts["item3_metric_vs_task_family"] = evaluate_keep_rule(
                SWEEP_RESULTS_PATH, "mufu_s79_3_stop_metric", "mufu_s79_1_task_family"
            )
        all_verdicts[f"item{stage_id}"] = {
            "winner": winner[0] if winner else None,
            "winner_overrides": winner[1] if winner else {},
            "verdicts": verdicts,
        }
        if winner is not None:
            kept_overrides.update(winner[1])

    if "6" in stages:
        stacked = {
            "mufu_s79_6_stack_bio05": 0.5,
            "mufu_s79_6_stack_bio065": 0.65,
            "mufu_s79_6_stack_bio080": 0.80,
        }
        for label, bio_frac in stacked.items():
            run_variant(
                label,
                replace(base, **kept_overrides, bio_fraction=bio_frac),
                df,
                datasets,
                smiles_encoder,
                force=force,
            )
        for key, reference, models in (
            ("item6_vs_baseline", "mufu_s79_0_baseline", list(stacked)),
            ("item6_vs_anchor", anchor_name, list(stacked)),
            ("item6_bio_frac_vs_stack05", "mufu_s79_6_stack_bio05", list(stacked)[1:]),
        ):
            all_verdicts[key] = [
                evaluate_keep_rule(SWEEP_RESULTS_PATH, m, reference) for m in models
            ]

    verdicts_path = SWEEP_DIR / "ablation_verdicts.json"
    verdicts_path.write_text(json.dumps(all_verdicts, indent=2, default=str))
    print(f"\nSaved ablation verdicts to {verdicts_path}", flush=True)
    return all_verdicts


def _train_single_task_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    split_arr: np.ndarray,
    full_df: pd.DataFrame,
    col: str,
    *,
    hidden_dims: tuple[int, ...],
    dropout: float,
    weight_decay: float,
    epochs: int,
    seed: int,
) -> dict[str, float]:
    device = torch.device(_device())
    train_idx = np.flatnonzero(split_arr == "train")
    val_idx = np.flatnonzero(split_arr == "val")

    X_dev = X.float().to(device)
    y_dev = y.float().to(device)
    gen = seed_everything(seed)

    encoder = FingerprintEncoder(
        FINGERPRINT_BITS, hidden_dims=hidden_dims, dropout=dropout
    )
    head = nn.Linear(encoder.out_dim, 1)
    nn.init.constant_(head.bias, float(y[train_idx].mean().item()))
    net = nn.Sequential(encoder, head).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    loader = DataLoader(
        range(len(train_idx)), batch_size=64, shuffle=True, generator=gen
    )

    best_val_rmse = float("inf")
    best_state = None
    for _ in range(1, epochs + 1):
        net.train()
        for batch_i in loader:
            idx = train_idx[batch_i]
            optimizer.zero_grad()
            pred = net(X_dev[idx]).squeeze(-1)
            nn.functional.mse_loss(pred, y_dev[idx]).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_pred = net(X_dev[val_idx]).squeeze(-1).cpu().numpy()
        val_rmse = float(np.sqrt(np.mean((val_pred - y[val_idx].numpy()) ** 2)))
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {
                k: v.detach().cpu().clone() for k, v in net.state_dict().items()
            }

    net.load_state_dict(best_state)
    net.eval()
    preds_by_split = {}
    with torch.no_grad():
        for sp in ("val", "test"):
            sp_idx = np.flatnonzero(split_arr == sp)
            sp_mask = full_df.loc[full_df["split"] == sp, col].notna().to_numpy()
            scores = np.full(len(sp_mask), np.nan, dtype=float)
            scores[sp_mask] = net(X_dev[sp_idx]).squeeze(-1).cpu().numpy()
            preds_by_split[sp] = {col: scores}
    rows = evaluate_split_predictions(full_df, preds_by_split)
    return {f"{r['split']}_{r['metric']}": float(r["value"]) for r in rows}


def _nearest_train_tanimoto(
    fp_matrix: np.ndarray, split_arr: np.ndarray
) -> dict[str, dict[str, float]]:
    train_fp = fp_matrix[split_arr == "train"].astype(np.float32)
    train_bits = train_fp.sum(axis=1)
    out = {}
    for sp in ("val", "test"):
        eval_fp = fp_matrix[split_arr == sp].astype(np.float32)
        eval_bits = eval_fp.sum(axis=1)
        inter = eval_fp @ train_fp.T
        union = eval_bits[:, None] + train_bits[None, :] - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        max_sim = sim.max(axis=1)
        out[sp] = {
            "mean_max_tanimoto": float(max_sim.mean()),
            "median_max_tanimoto": float(np.median(max_sim)),
            "frac_below_0.4": float((max_sim < 0.4).mean()),
            "frac_above_0.7": float((max_sim >= 0.7).mean()),
        }
    return out


def run_egfr_investigation(
    df: pd.DataFrame, datasets: dict[str, MuFuDataset] | None = None
) -> dict:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {}

    split_stats = []
    for target in ("BACE1", "EGFR"):
        pic50_col = f"pIC50_{target}"
        active_col = f"active_{target}"
        for sp in ("train", "val", "test"):
            sub = df[df["split"] == sp]
            vals = sub[pic50_col].dropna()
            acts = sub[active_col].dropna()
            split_stats.append(
                {
                    "target": target,
                    "split": sp,
                    "n_labeled": len(vals),
                    "mean_pic50": float(vals.mean()),
                    "std_pic50": float(vals.std(ddof=1)),
                    "median_pic50": float(vals.median()),
                    "iqr_pic50": float(vals.quantile(0.75) - vals.quantile(0.25)),
                    "active_rate": float(acts.mean()),
                }
            )
    report["split_distributions"] = split_stats

    repeat_stats = {}
    for name, path in (("BACE1", BACE1_CSV_PATH), ("EGFR", EGFR_CSV_PATH)):
        if path.exists():
            raw_df = pd.read_csv(path)
            multi = raw_df[raw_df["n_meas"] > 1]
            repeat_stats[name] = {
                "n_molecules": len(raw_df),
                "n_repeat_molecules": len(multi),
                "repeat_fraction": float(len(multi) / len(raw_df)),
                "mean_spread_on_repeats": float(multi["pIC50_spread"].mean()),
                "median_spread_on_repeats": float(multi["pIC50_spread"].median()),
                "frac_repeats_spread_ge_0.5": float(
                    (multi["pIC50_spread"] >= 0.5).mean()
                ),
                "frac_repeats_spread_ge_1.0": float(
                    (multi["pIC50_spread"] >= 1.0).mean()
                ),
            }
    report["repeat_measurement_spread"] = repeat_stats

    ecfp_sweep_dir = AI_DIR / "outputs" / "encoders" / "ECFP4" / "sweep"
    sweep_csvs = sorted(ecfp_sweep_dir.glob("*.csv")) if ecfp_sweep_dir.exists() else []
    if sweep_csvs:
        frames = [pd.read_csv(p).assign(stage=p.stem) for p in sweep_csvs]
        all_sweep = pd.concat(frames, ignore_index=True)
        mlp_sweep = (
            all_sweep[all_sweep["MODEL"].isna()] if "MODEL" in all_sweep else all_sweep
        )
        grouped = (
            mlp_sweep.groupby(
                ["stage", "HIDDEN_DIMS", "DROPOUT", "FINAL_ACTIVATION", "WEIGHT_DECAY"],
                dropna=False,
            )["val_RMSE"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean")
        )
        report["ecfp4_sweep_top5"] = grouped.head(5).to_dict(orient="records")
    else:
        report["ecfp4_sweep_top5"] = (
            "outputs/encoders/ECFP4/sweep/*.csv not present in checkout; recomputed below on BACE1 and EGFR"
        )

    from featurize.fingerprint import mol_to_ecfp4
    from pipeline.preprocess import to_mol

    single_task_rows = []
    tanimoto_stats = {}
    for target in ("BACE1", "EGFR"):
        col = f"pIC50_{target}"
        labeled = df[col].notna().to_numpy()
        sub = df.loc[labeled].reset_index(drop=True)
        fps = np.stack(
            [mol_to_ecfp4(to_mol(s), FINGERPRINT_BITS) for s in sub["smiles"]]
        )
        split_arr = sub["split"].to_numpy()
        tanimoto_stats[target] = _nearest_train_tanimoto(fps, split_arr)

        X_t = torch.from_numpy(fps)
        y_t = torch.tensor(sub[col].to_numpy(dtype=float), dtype=torch.float32)

        for cfg_label, dims, drop, wd, ep in (
            ("mlp_default_1024_512_d0.2", (1024, 512), 0.2, 1e-5, 50),
            ("mlp_pick_512_256_d0.6", (512, 256), 0.6, 1e-3, 100),
        ):
            seed_metrics = [
                _train_single_task_mlp(
                    X_t,
                    y_t,
                    split_arr,
                    df,
                    col,
                    hidden_dims=dims,
                    dropout=drop,
                    weight_decay=wd,
                    epochs=ep,
                    seed=s,
                )
                for s in SWEEP_SEEDS
            ]
            m_df = pd.DataFrame(seed_metrics)
            single_task_rows.append(
                {
                    "target": target,
                    "config": cfg_label,
                    "val_rmse_mean": float(m_df["val_rmse"].mean()),
                    "val_rmse_std": float(m_df["val_rmse"].std(ddof=1)),
                    "val_r2_mean": float(m_df["val_r2"].mean()),
                    "test_rmse_mean": float(m_df["test_rmse"].mean()),
                    "test_rmse_std": float(m_df["test_rmse"].std(ddof=1)),
                    "test_r2_mean": float(m_df["test_r2"].mean()),
                }
            )
    report["tanimoto_similarity_to_train"] = tanimoto_stats
    report["single_task_ecfp4_mlp"] = single_task_rows

    out_json = SWEEP_DIR / "egfr_analysis.json"
    out_json.write_text(json.dumps(report, indent=2))
    pd.DataFrame(split_stats).to_csv(SWEEP_DIR / "egfr_split_stats.csv", index=False)
    pd.DataFrame(single_task_rows).to_csv(
        SWEEP_DIR / "egfr_single_task_mlp.csv", index=False
    )
    print(f"Saved EGFR analysis to {out_json}", flush=True)
    print("\n--- Split Distributions ---")
    print(pd.DataFrame(split_stats).round(4).to_string(index=False))
    print("\n--- Single-Task ECFP4 MLP (Seeds 0-2) ---")
    print(pd.DataFrame(single_task_rows).round(4).to_string(index=False))
    return report


def run_posthoc_calibration(
    model_names: list[str],
    datasets: dict[str, MuFuDataset],
    smiles_encoder: SmilesEncoder,
    *,
    checkpoint_dir: Path = SWEEP_CKPT_DIR,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
) -> pd.DataFrame:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    device = _device()
    all_rows: list[pd.DataFrame] = []

    for model_name in model_names:
        for seed in seeds:
            ckpt_path = checkpoint_dir / f"{model_name}_seed_{seed}.pt"
            if not ckpt_path.exists():
                print(f"[skip] missing checkpoint {ckpt_path}", flush=True)
                continue
            model, meta = load_mufu_checkpoint(ckpt_path, smiles_encoder, device=device)
            params = meta["checkpoint"].get("params", {})
            pw_mode = params.get("pos_weight", "raw")
            batch_size = int(params.get("batch_size", 64))
            pos_weights = dataset_pos_weights(datasets["train"], mode=pw_mode)

            val_logits = predict_logits(
                model,
                datasets["val"],
                batch_size=batch_size,
                family_states=meta["family_states"],
            )
            test_logits = predict_logits(
                model,
                datasets["test"],
                batch_size=batch_size,
                family_states=meta["family_states"],
            )

            for cal_mode in CALIBRATION_MODES:
                cal = fit_calibration(
                    cal_mode, val_logits, datasets["val"], pos_weights=pos_weights
                )
                for split_name, split_logits in (
                    ("val", val_logits),
                    ("test", test_logits),
                ):
                    adjusted = cal.apply(split_logits)
                    probs = {
                        col: torch.sigmoid(adjusted[key][:, j]).numpy()
                        for key, columns in COLUMN_GROUPS
                        if key in ("active_logits", "tox_logits")
                        for j, col in enumerate(columns)
                    }
                    cal_df = evaluate_calibration(probs, datasets[split_name])
                    cal_df.insert(0, "model", model_name)
                    cal_df.insert(1, "pos_weight", pw_mode)
                    cal_df.insert(2, "calibration", cal_mode)
                    cal_df.insert(3, "split", split_name)
                    cal_df.insert(4, "seed", int(seed))
                    all_rows.append(cal_df)

    if not all_rows:
        return pd.DataFrame()

    full_cal = pd.concat(all_rows, ignore_index=True)
    full_cal.to_csv(SWEEP_DIR / "calibration_per_seed.csv", index=False)

    metric_cols = ["prevalence", "mean_prob", "bias", "abs_bias", "brier", "bce", "ece"]
    agg = full_cal.groupby(
        ["model", "pos_weight", "calibration", "split", "group", "endpoint"],
        as_index=False,
    )[metric_cols].mean()
    agg.to_csv(SWEEP_DIR / "calibration_endpoints.csv", index=False)

    summaries = agg[agg["group"] == "summary"].sort_values(
        ["model", "endpoint", "split", "calibration"]
    )
    summaries.to_csv(SWEEP_DIR / "calibration_summary.csv", index=False)
    print("\n--- Post-Hoc Calibration Summary (Val & Test) ---")
    print(summaries.round(4).to_string(index=False))
    return summaries


def print_sweep_summary(results_path: Path = SWEEP_RESULTS_PATH) -> pd.DataFrame:
    if not results_path.exists():
        print(f"No sweep results found at {results_path}")
        return pd.DataFrame()

    df = load_results(results_path)
    rf_df = load_results()
    rf_df = rf_df[rf_df["model"].isin(["rf_bioactivity", "rf_toxicity"])]
    df = pd.concat([df, rf_df], ignore_index=True)

    summary = summarize_seeds(df)
    target_rows = [
        ("bace1", "pIC50_BACE1", "rmse"),
        ("bace1", "pIC50_BACE1", "r2"),
        ("bace1", "active_BACE1", "roc_auc"),
        ("egfr", "pIC50_EGFR", "rmse"),
        ("egfr", "pIC50_EGFR", "r2"),
        ("egfr", "active_EGFR", "roc_auc"),
        ("tox21", "macro_tox21", "roc_auc"),
        ("clintox", "macro_clintox", "roc_auc"),
        ("overall", "macro_overall", "roc_auc"),
    ]
    for sp in ("val", "test"):
        print(f"\n================ {sp.upper()} SPLIT ================")
        sub = summary[summary["split"] == sp]
        rows = []
        for model_name in sub["model"].unique():
            m_sub = sub[sub["model"] == model_name]
            entry: dict[str, object] = {
                "model": model_name,
                "n_seeds": int(m_sub["n_seeds"].max()),
            }
            for ds, ep, met in target_rows:
                match = m_sub[
                    (m_sub["dataset"] == ds)
                    & (m_sub["endpoint"] == ep)
                    & (m_sub["metric"] == met)
                ]
                if not match.empty:
                    v = float(match["value"].iloc[0])
                    s = float(match["std"].iloc[0])
                    entry[f"{ep}_{met}"] = f"{v:.3f}±{s:.3f}"
            rows.append(entry)
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_all = subparsers.add_parser(
        "all", help="run all sweep items, egfr, and calibration"
    )
    p_all.add_argument("--force", action="store_true")

    p_sweep = subparsers.add_parser("sweep", help="run specific ablation stages (0-6)")
    p_sweep.add_argument(
        "stages",
        nargs="*",
        default=["0", "1", "2", "3", "4", "5", "6"],
        help="stage numbers from 0 to 6",
    )
    p_sweep.add_argument("--force", action="store_true")

    subparsers.add_parser("egfr", help="run Item 7 EGFR data and single-task analysis")

    p_cal = subparsers.add_parser(
        "calibrate", help="run Item 8 post-hoc calibration on saved checkpoints"
    )
    p_cal.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="checkpoint model names to calibrate (default: anchor, pos_weight variants, and stacked)",
    )

    subparsers.add_parser(
        "summary", help="print formatted summary of sweep_results.csv"
    )

    args = parser.parse_args(argv)
    if args.command == "summary":
        print_sweep_summary()
        return

    print("Loading benchmark split...", flush=True)
    df = load_benchmark_split()
    if args.command == "egfr":
        run_egfr_investigation(df)
        return

    print("Featurizing splits...", flush=True)
    datasets = split_datasets(df)
    for sp in datasets:
        print(f"  {sp}: {len(datasets[sp])} molecules", flush=True)

    smiles_encoder = SmilesEncoder(freeze=True, max_cache_size=len(df))

    if args.command in ("all", "sweep"):
        stages = (
            {"0", "1", "2", "3", "4", "5", "6"}
            if args.command == "all"
            else set(args.stages)
        )
        run_ablation_stages(stages, df, datasets, smiles_encoder, force=args.force)
        print_sweep_summary()

    if args.command == "all":
        run_egfr_investigation(df, datasets)

    if args.command in ("all", "calibrate"):
        models = getattr(args, "models", None)
        if not models:
            models = [
                "mufu_s79_0_baseline",
                "mufu_s79_1_task_family",
                "mufu_s79_2_pw_cap5",
                "mufu_s79_2_pw_cap10",
                "mufu_s79_2_pw_sqrt",
                "mufu_s79_2_pw_none",
                "mufu_s79_6_stack_bio05",
                "mufu_s79_6_stack_bio065",
                "mufu_s79_6_stack_bio080",
            ]
        run_posthoc_calibration(models, datasets, smiles_encoder)


if __name__ == "__main__":
    main()
