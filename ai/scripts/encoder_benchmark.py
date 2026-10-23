"""
BACE-1 encoder experiment workflow built on the shared benchmark foundation.
"""

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline.benchmark import (
    evaluate_split_predictions,
    load_benchmark_split,
    params_json,
    split_sha256,
)
from baseline.results import file_sha256
from featurize.fingerprint import FINGERPRINT_BITS
from pipeline.cache import write_csv
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.preprocess import featurize_many
from pipeline.split import SPLITS

SWEEP_PROTOCOL = "encoder-validation-only-v1"
METRICS = {"RMSE": "rmse", "R2": "r2", "ROC-AUC": "roc_auc"}


def grid(base, **values):
    return [base | dict(zip(values, v)) for v in itertools.product(*values.values())]


@dataclass
class BACE1Benchmark:
    full: pd.DataFrame
    frame: pd.DataFrame
    data_sha256: str
    split_sha256: str
    data_path: Path
    indices: dict[str, np.ndarray]
    _features: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, data_path: Path = COMBINED_CSV_PATH):
        full = load_benchmark_split(data_path)
        data_hash = file_sha256(data_path)
        split_hash = split_sha256(full)
        frame = full.loc[full["pIC50_BACE1"].notna()].reset_index(drop=True)
        indices = {
            name: np.flatnonzero(frame["split"].to_numpy() == name) for name in SPLITS
        }
        if any(len(idx) == 0 for idx in indices.values()):
            raise ValueError("BACE-1 needs observed labels in train, val, and test")
        return cls(full, frame, data_hash, split_hash, data_path, indices)

    @property
    def y(self):
        return torch.tensor(self.frame["pIC50_BACE1"].to_numpy(), dtype=torch.float32)

    def features(self, n_bits=FINGERPRINT_BITS):
        if n_bits not in self._features:
            features, failed = featurize_many(self.frame["smiles"], n_bits)
            if failed or len(features) != len(self.frame):
                raise ValueError(
                    f"featurization would change the BACE-1 cohort; failed rows: {failed}"
                )
            if [graph.smiles for graph in features] != self.frame["smiles"].tolist():
                raise ValueError("featurization changed SMILES identity or row order")
            self._features[n_bits] = features
        return self._features[n_bits]

    def fingerprints(self, n_bits):
        return torch.cat([graph.fp for graph in self.features(n_bits)])

    def graphs(self):
        graphs = self.features()
        for graph, target in zip(graphs, self.y):
            graph.y = target.unsqueeze(0)
        return graphs

    def score(self, predict, *, evaluate_test=False):
        names = SPLITS if evaluate_test else SPLITS[:2]
        predictions = {}
        for name in names:
            preds = predict(self.indices[name])
            if isinstance(preds, torch.Tensor):
                preds = preds.detach().cpu().numpy()
            predictions[name] = {"pIC50_BACE1": preds}
        rows = evaluate_split_predictions(self.frame, predictions)
        by_key = {(r["split"], r["metric"]): r["value"] for r in rows}
        out = {
            f"{name}_{wide}": by_key[(name, metric)]
            for name in names
            for wide, metric in METRICS.items()
        }
        if not evaluate_test:
            out.update({f"test_{wide}": np.nan for wide in METRICS})
        return out


def validation_rmse(frame, preds, *, evaluate=evaluate_split_predictions) -> float:
    if hasattr(frame, "frame"):
        frame = frame.frame
    finite = (
        torch.isfinite(preds).all()
        if isinstance(preds, torch.Tensor)
        else np.isfinite(preds).all()
    )
    if not finite:
        return float("inf")
    rows = evaluate(frame, {"val": {"pIC50_BACE1": preds}})
    return next(r["value"] for r in rows if r["metric"] == "rmse")


def load_sweep_records(path: Path, data: BACE1Benchmark, experiment: str):
    if not path.exists():
        return pd.DataFrame()
    records = pd.read_csv(path)
    required = {
        "cfg",
        "seed",
        "val_RMSE",
        "data_sha256",
        "split_sha256",
        "sweep_protocol",
        "experiment",
    }
    if not required <= set(records.columns):
        raise ValueError(
            f"{path}: legacy/incompatible sweep records; preserve this file and "
            "choose a new --output-dir for shared experiments"
        )
    if records.empty:
        return records
    if not records["sweep_protocol"].eq(SWEEP_PROTOCOL).all():
        raise ValueError(f"{path}: incompatible sweep protocol")
    if not records["experiment"].eq(experiment).all():
        raise ValueError(f"{path}: incompatible encoder experiment")
    if not records["data_sha256"].eq(data.data_sha256).all():
        raise ValueError(f"{path}: incompatible data_sha256")
    if not records["split_sha256"].eq(data.split_sha256).all():
        raise ValueError(f"{path}: incompatible split_sha256")
    configs = records["cfg"].map(json.loads)
    if not configs.map(lambda config: isinstance(config, dict)).all():
        raise ValueError(f"{path}: configuration must be a JSON object")
    records["cfg"] = configs.map(params_json)
    records["val_RMSE"] = pd.to_numeric(records["val_RMSE"], errors="raise")
    if not np.isfinite(records["val_RMSE"]).all():
        raise ValueError(f"{path}: validation RMSE must be finite")
    seeds = pd.to_numeric(records["seed"], errors="raise")
    if not ((seeds >= 0) & (seeds < 2**32) & (seeds % 1 == 0)).all():
        raise ValueError(f"{path}: invalid training seeds")
    records["seed"] = seeds.astype(int)
    if (records.groupby(["cfg", "seed"])["val_RMSE"].nunique() > 1).any():
        raise ValueError(
            f"{path}: conflicting duplicate configuration/seed measurements"
        )
    return records


def select_configuration(records: pd.DataFrame, seeds):
    """Select minimum mean validation RMSE using the same complete seed set."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("selection seeds must be a nonempty unique list")
    if records.empty:
        raise ValueError("no compatible sweep records to select from")
    selected = records.loc[records["seed"].isin(seeds), ["cfg", "seed", "val_RMSE"]]
    if (selected.groupby(["cfg", "seed"])["val_RMSE"].nunique() > 1).any():
        raise ValueError("conflicting duplicate configuration/seed measurements")
    selected = selected.drop_duplicates(["cfg", "seed"])
    eligible = selected.groupby("cfg").filter(
        lambda group: set(group["seed"]) == set(seeds)
    )
    means = eligible.groupby("cfg")["val_RMSE"].mean().sort_index()
    if means.empty:
        raise ValueError("no configuration has the complete selection seed set")
    choice = means.idxmin()
    return json.loads(choice), {
        "method": "mean_validation_rmse",
        "seeds": list(seeds),
        "mean_val_rmse": float(means[choice]),
    }


def run_cli(argv, *, stages, output_dir, experiment, normalize_config, run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", help="exploratory stage numbers or names")
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--stage-config",
        type=Path,
        help="JSON configuration overrides for exploratory stages",
    )
    args = parser.parse_args(argv)
    by_number = {name.split("-")[0]: name for name in stages}
    names = [by_number.get(arg, arg) for arg in args.stages] or list(stages)
    if unknown := set(names) - set(stages):
        parser.error(f"unknown stages {sorted(unknown)}")
    if args.stage_config:
        overrides = json.loads(args.stage_config.read_text())
        if not isinstance(overrides, dict):
            parser.error("--stage-config must contain a JSON object")
        stages = {
            name: (
                [normalize_config(cfg | overrides) for cfg in stages[name][0]],
                stages[name][1],
            )
            for name in names
        }

    data = BACE1Benchmark.load()
    histories = {
        name: load_sweep_records(args.output_dir / f"{name}.csv", data, experiment)
        for name in names
    }
    print(
        params_json(
            {
                "data_sha256": data.data_sha256,
                "split_sha256": data.split_sha256,
                "mode": "exploratory",
                "stages": names,
            }
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        records = histories[name]
        done = (
            set(records[["cfg", "seed"]].itertuples(index=False, name=None))
            if len(records)
            else set()
        )
        configs, seeds = stages[name]
        runs = [(cfg, seed) for cfg in configs for seed in seeds]
        for position, (cfg, seed) in enumerate(runs, 1):
            key = (params_json(cfg), seed)
            if key in done:
                continue
            measurements = run(cfg, seed, data)
            row = pd.DataFrame(
                [
                    {
                        **measurements,
                        "data_sha256": data.data_sha256,
                        "split_sha256": data.split_sha256,
                        "sweep_protocol": SWEEP_PROTOCOL,
                        "experiment": experiment,
                    }
                ]
            )
            records = (
                pd.concat([records, row], ignore_index=True) if len(records) else row
            )
            write_csv(records, args.output_dir / f"{name}.csv")
            done.add(key)
            line = f"{name} [{position}/{len(runs)}] val {row['val_RMSE'][0]:.3f} {row['secs'][0]}s {row['cfg'][0]}"
            print(line, flush=True)
            with (args.output_dir / f"{name}.log").open("a") as log:
                print(line, file=log)
    return []
