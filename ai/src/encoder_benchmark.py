"""BACE-1 encoder experiment workflow built on the shared benchmark foundation.

Exploratory stage files retain their wide metrics and logs, but test measurements
are reserved for an explicitly selected canonical run. Legacy files are read-only
history unless their protocol metadata verifies against the current benchmark.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark as bm
from baseline.results import RESULTS_PATH
from pipeline.cache import write_csv
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.preprocess import featurize_many
from pipeline.split import SPLITS

SWEEP_PROTOCOL = "encoder-validation-only-v1"
METRICS = {"RMSE": "rmse", "R2": "r2", "ROC-AUC": "roc_auc"}


@dataclass
class BACE1Benchmark:
    full: pd.DataFrame
    frame: pd.DataFrame
    metadata: dict
    data_path: Path
    indices: dict[str, np.ndarray]
    _features: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, data_path: Path = COMBINED_CSV_PATH):
        full = bm.load_benchmark(data_path)
        metadata = bm.benchmark_metadata(full, data_path)
        frame = full.loc[full["pIC50_BACE1"].notna()].reset_index(drop=True)
        indices = {
            name: np.flatnonzero(frame["split"].to_numpy() == name) for name in SPLITS
        }
        if any(len(idx) == 0 for idx in indices.values()):
            raise ValueError("BACE-1 needs observed labels in train, val, and test")
        return cls(full, frame, metadata, data_path, indices)

    @property
    def y(self):
        return torch.tensor(self.frame["pIC50_BACE1"].to_numpy(), dtype=torch.float32)

    def features(self, n_bits=2048):
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
        out = {}
        names = SPLITS if evaluate_test else SPLITS[:2]
        for name in names:
            idx = self.indices[name]
            predictions = predict(idx)
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.detach().cpu().numpy()
            labels = self.frame.iloc[idx]
            scores = (
                predictions
                if isinstance(predictions, pd.Series)
                else pd.Series(predictions, index=labels.index)
            )
            regression = bm.evaluate_predictions(
                labels["pIC50_BACE1"], scores, "regression"
            )
            classification = bm.evaluate_predictions(
                labels["active_BACE1"], scores, "classification"
            )
            out.update(
                {
                    f"{name}_RMSE": regression["rmse"],
                    f"{name}_R2": regression["r2"],
                    f"{name}_ROC-AUC": classification["roc_auc"],
                }
            )
        if not evaluate_test:
            out.update({f"test_{name}": np.nan for name in METRICS})
        return out


def config_json(config):
    return json.dumps(config, sort_keys=True)


def load_sweep_records(path: Path, data: BACE1Benchmark, experiment: str):
    if not path.exists():
        return pd.DataFrame()
    records = pd.read_csv(path)
    required = {"cfg", "seed", "val_RMSE", "benchmark", "sweep_protocol", "experiment"}
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
    for stored in records["benchmark"].unique():
        bm.verify_benchmark_metadata(json.loads(stored), data.full, data.data_path)
    configs = records["cfg"].map(json.loads)
    if not configs.map(lambda config: isinstance(config, dict)).all():
        raise ValueError(f"{path}: configuration must be a JSON object")
    records["cfg"] = configs.map(config_json)
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


def canonical_rows(data, config, measurements, *, encoder, selection):
    params = config_json(
        {
            "benchmark": data.metadata,
            "seed": bm.TRAINING_SEED,
            "config": config,
            "best_epoch": measurements.get("best_epoch"),
            "selection": selection,
            "score_source": "predicted_pic50",
            "device": measurements.get("device"),
            "deterministic_algorithms": measurements.get("deterministic_algorithms"),
        }
    )
    return [
        {
            "encoder": encoder,
            "dataset": "bace1",
            "endpoint": "pIC50_BACE1",
            "split": name,
            "metric": metric,
            "value": measurements[f"{name}_{wide}"],
            "n_labeled": len(data.indices[name]),
            "n_positive": int(
                data.frame.iloc[data.indices[name]]["active_BACE1"].eq(1).sum()
            ),
            "params": params,
        }
        for name in ("val", "test")
        for wide, metric in METRICS.items()
    ]


def verify_export(data, path=RESULTS_PATH):
    """Check provenance and refuse an export that would discard existing RF rows."""
    bm.verify_benchmark_metadata(data.metadata, data.full, data.data_path)
    if path.exists():
        existing = pd.read_csv(path)
        rf = existing[existing["model"].isin(["rf_bioactivity", "rf_toxicity"])]
        if not rf["data_sha256"].eq(data.metadata["data_sha256"]).all():
            raise ValueError("export would discard RF results from another snapshot")


def run_cli(argv, *, stages, output_dir, experiment, encoder, normalize_config, run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*", help="exploratory stage numbers or names")
    parser.add_argument(
        "--benchmark", action="store_true", help="canonical seed-42 run"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config", type=Path, help="fixed configuration JSON")
    source.add_argument(
        "--select-from", type=Path, nargs="+", help="compatible stage CSVs"
    )
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--write-results", action="store_true", help="explicitly export results.csv"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="inspect without training or writes"
    )
    args = parser.parse_args(argv)
    if args.benchmark:
        if args.stages or not (args.config or args.select_from):
            parser.error(
                "--benchmark requires --config or --select-from, without stages"
            )
    elif args.config or args.select_from or args.write_results:
        parser.error("configuration selection and --write-results require --benchmark")
    by_number = {name.split("-")[0]: name for name in stages}
    names = [by_number.get(arg, arg) for arg in args.stages] or list(stages)
    if unknown := set(names) - set(stages):
        parser.error(f"unknown stages {sorted(unknown)}")

    data = BACE1Benchmark.load()
    if args.benchmark:
        if args.config:
            config = json.loads(args.config.read_text())
            selection = {"method": "fixed_config", "source": str(args.config)}
        else:
            records = pd.concat(
                [
                    load_sweep_records(path, data, experiment)
                    for path in args.select_from
                ],
                ignore_index=True,
            )
            config, selection = select_configuration(records, args.selection_seeds)
        config = normalize_config(config)
        verify_export(data)
        print(
            config_json(
                {"benchmark": data.metadata, "config": config, "selection": selection}
            )
        )
        if args.dry_run:
            return []
        measurements = run(config, bm.TRAINING_SEED, data, evaluate_test=True)
        rows = canonical_rows(
            data, config, measurements, encoder=encoder(config), selection=selection
        )
        print(pd.DataFrame(rows)[["split", "metric", "value"]].to_string(index=False))
        if args.write_results:
            verify_export(data)
            bm.record_results(rows, model=experiment, data_path=data.data_path)
        return rows

    histories = {
        name: load_sweep_records(args.output_dir / f"{name}.csv", data, experiment)
        for name in names
    }
    print(
        config_json(
            {"benchmark": data.metadata, "mode": "exploratory", "stages": names}
        )
    )
    if args.dry_run:
        return []
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
            key = (config_json(cfg), seed)
            if key in done:
                continue
            measurements = run(cfg, seed, data)
            row = pd.DataFrame(
                [
                    {
                        **measurements,
                        "benchmark": config_json(data.metadata),
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
