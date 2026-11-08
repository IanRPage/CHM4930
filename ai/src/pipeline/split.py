"""
Assign compounds to train, val, or test by scaffold or UMAP/HDBSCAN cluster.

Compounds that share a scaffold always land in the same split. Scaffolds ignore
stereochemistry, so stereoisomers of a ring system share one. Acyclic compounds have an
empty scaffold, so each one is its own group. You should split the combined table rather
than each source that way a compound stays in one split across every task. When the table
has `in_*` source flags, each source gets its own size budget, so every source comes out
close to `frac` rather than only the table as a whole.

How to use as CLI tool (from `ai/`, with `PYTHONPATH=src`):

    python -m pipeline.split data/baseline/combined.csv            # largest scaffold groups go to train
    python -m pipeline.split data/baseline/combined.csv --seed 0   # balanced split, shuffled with the seed
    python -m pipeline.split data/baseline/combined.csv --method umap --seed 42

Writes `smiles`, `group`, and `split` to `<csv stem>-splits.csv` next to the input,
then prints split sizes overall and for each `in_*` column. `group` contains
scaffold SMILES for scaffold splitting or numerical cluster IDs for UMAP splitting.
From Python, use `scaffold_split()` or `umap_split()`.
"""

import argparse
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import HDBSCAN
from umap import UMAP

from featurize.fingerprint import mol_to_ecfp4
from featurize.smiles import mol_to_smiles
from pipeline.cache import write_csv
from pipeline.preprocess import to_mol

SPLITS = ["train", "val", "test"]


def murcko_scaffold(smiles: str) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(to_mol(smiles))
    if not scaffold.GetNumAtoms():
        return ""
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def _validate_frac(frac: Sequence[float]) -> None:
    if len(frac) != 3:
        raise ValueError(f"frac needs 3 values (train, val, test), got {len(frac)}")
    if any(f < 0 for f in frac):
        raise ValueError(f"frac can't be negative, got {tuple(frac)}")
    if not math.isclose(sum(frac), 1.0):
        raise ValueError(f"frac must sum to 1, got {sum(frac)}")


def _source_flags(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("in_")]


def _scaffold_groups(
    smiles: Sequence[str], scaffolds: Sequence[str]
) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for pos, (smi, scaffold) in enumerate(zip(smiles, scaffolds)):
        key = scaffold or mol_to_smiles(to_mol(smi))
        groups.setdefault(key, []).append(pos)
    return [g for _, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def _assign_cluster_groups(
    df: pd.DataFrame,
    groups: list[list[int]],
    frac: Sequence[float],
    seed: int | None,
    *,
    reserve_holdouts: bool = True,
) -> list[str]:
    """Reserve holdout coverage before assigning the remaining whole groups.

    Set reserve_holdouts=False to restore the original train-first allocator.
    Reservations and repair moves respect per-source budgets. As in the original
    allocator, test receives leftover groups and may exceed its target size.
    This is a heuristic; the original empty-split and source checks still apply.
    """
    flags = _source_flags(df)
    member = (
        df[flags].fillna(False).to_numpy(bool)
        if flags
        else np.ones((len(df), 1), dtype=bool)
    )
    sizes = {name: f * member.sum(axis=0) for name, f in zip(SPLITS, frac)}

    if seed is not None:
        rng = random.Random(seed)
        half = min(frac[1], frac[2]) * len(df) / 2
        big = [g for g in groups if len(g) > half]
        small = [g for g in groups if len(g) <= half]
        rng.shuffle(big)
        rng.shuffle(small)
        groups = big + small

    counts = {name: np.zeros(member.shape[1]) for name in SPLITS}
    assigned = [""] * len(df)
    reserved: set[int] = set()
    group_counts_by_id = [member[group].sum(axis=0) for group in groups]
    group_names: dict[int, str] = {}
    row_counts = dict.fromkeys(SPLITS, 0)
    fractions = dict(zip(SPLITS, frac))

    def place(group_id: int, name: str) -> None:
        previous = group_names.get(group_id)
        contribution = group_counts_by_id[group_id]
        if previous is not None:
            counts[previous] -= contribution
            row_counts[previous] -= len(groups[group_id])
        counts[name] += contribution
        row_counts[name] += len(groups[group_id])
        group_names[group_id] = name
        for pos in groups[group_id]:
            assigned[pos] = name

    if reserve_holdouts:
        # The seed above determines tie order; stable sorting retains those ties.
        for name in ("val", "test"):
            if fractions[name] == 0:
                continue
            while True:
                missing = (sizes[name] >= 1) & (counts[name] == 0)
                needs_group = row_counts[name] == 0
                if not needs_group and not missing.any():
                    break
                remaining = [i for i in range(len(groups)) if i not in reserved]
                if not remaining:
                    break
                available_coverage = np.sum(
                    [group_counts_by_id[i] > 0 for i in remaining], axis=0
                )
                other_splits = [
                    other for other in SPLITS if other != name and fractions[other] > 0
                ]
                future_coverage = np.zeros(member.shape[1], dtype=int)
                for other in other_splits:
                    future_coverage += (sizes[other] >= 1) & (counts[other] == 0)
                future_groups = sum(row_counts[other] == 0 for other in other_splits)
                candidates = []
                for group_id in remaining:
                    contribution = group_counts_by_id[group_id]
                    gain = int(np.sum(missing & (contribution > 0))) + int(needs_group)
                    if not gain or np.any(counts[name] + contribution > sizes[name]):
                        continue
                    # Leave enough independent groups to cover the other splits.
                    if len(remaining) - 1 < future_groups or np.any(
                        available_coverage - (contribution > 0) < future_coverage
                    ):
                        continue
                    candidates.append(
                        (-gain / len(groups[group_id]), len(groups[group_id]), group_id)
                    )
                if not candidates:
                    break
                group_id = min(candidates)[2]
                reserved.add(group_id)
                place(group_id, name)

    order = list(range(len(groups)))
    if reserve_holdouts:
        order.sort(key=lambda i: -len(groups[i]))
    for group_id in order:
        if group_id in reserved:
            continue
        group_counts = group_counts_by_id[group_id]
        name = next(
            (
                s
                for s in SPLITS[:2]
                if (not reserve_holdouts or fractions[s] > 0)
                and np.all(counts[s] + group_counts <= sizes[s])
            ),
            "test",
        )
        if reserve_holdouts and fractions[name] == 0:
            raise ValueError(
                "a whole cluster group fits no positive-fraction split budget; "
                "change frac or clustering parameters"
            )
        place(group_id, name)

    if reserve_holdouts:
        totals = member.sum(axis=0)
        present = totals > 0

        def error(name: str, source_counts: np.ndarray, rows: int) -> float:
            overall = (rows / len(df) - fractions[name]) ** 2
            source = (
                np.mean(
                    (source_counts[present] / totals[present] - fractions[name]) ** 2
                )
                if flags and present.any()
                else 0.0
            )
            return float(overall + source)

        # One repair pass moves only complete training groups. Donor coverage
        # and nonemptiness are preserved; neither holdout budget is exceeded.
        for group_id in order:
            if group_names[group_id] != "train":
                continue
            contribution = group_counts_by_id[group_id]
            remaining_counts = counts["train"] - contribution
            remaining_rows = row_counts["train"] - len(groups[group_id])
            if remaining_rows == 0 or np.any(
                (sizes["train"] >= 1) & (remaining_counts == 0)
            ):
                continue
            best_name = None
            best_delta = -1e-12
            for name in ("val", "test"):
                if fractions[name] == 0 or np.any(
                    counts[name] + contribution > sizes[name]
                ):
                    continue
                delta = (
                    error("train", remaining_counts, remaining_rows)
                    + error(
                        name,
                        counts[name] + contribution,
                        row_counts[name] + len(groups[group_id]),
                    )
                    - error("train", counts["train"], row_counts["train"])
                    - error(name, counts[name], row_counts[name])
                )
                if delta < best_delta:
                    best_name, best_delta = name, delta
            if best_name is not None:
                place(group_id, best_name)

    empty = [name for name, f in zip(SPLITS, frac) if f > 0 and name not in assigned]
    if empty:
        raise ValueError(
            f"{', '.join(empty)} split(s) came out empty; a cluster group may be "
            "too large for the requested frac"
        )
    starved = [
        f"{flag} {name}"
        for name in SPLITS
        for flag, size, count in zip(flags, sizes[name], counts[name])
        if size >= 1 and count == 0
    ]
    if starved:
        raise ValueError(
            f"no compounds from {', '.join(starved)}; a cluster group may be too "
            "large for the requested frac"
        )
    return assigned


def scaffold_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int | None = None,
) -> pd.DataFrame:
    _validate_frac(frac)
    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    flags = _source_flags(df)
    member = (
        df[flags].fillna(False).to_numpy(bool)
        if flags
        else np.ones((len(df), 1), dtype=bool)
    )
    sizes = {name: f * member.sum(axis=0) for name, f in zip(SPLITS, frac)}

    groups = _scaffold_groups(df["smiles"], scaffolds)
    if seed is not None:
        rng = random.Random(seed)
        half = min(frac[1], frac[2]) * len(df) / 2
        big = [g for g in groups if len(g) > half]
        small = [g for g in groups if len(g) <= half]
        rng.shuffle(big)
        rng.shuffle(small)
        groups = big + small

    counts = {name: np.zeros(member.shape[1]) for name in SPLITS}
    assigned = [""] * len(scaffolds)
    for group in groups:
        group_counts = member[group].sum(axis=0)
        name = next(
            (s for s in SPLITS[:2] if np.all(counts[s] + group_counts <= sizes[s])),
            "test",
        )
        counts[name] += group_counts
        for pos in group:
            assigned[pos] = name

    empty = [name for name, f in zip(SPLITS, frac) if f > 0 and name not in assigned]
    if empty:
        raise ValueError(
            f"{', '.join(empty)} split(s) came out empty; a scaffold group may be "
            "too large for the requested frac"
        )
    starved = [
        f"{flag} {name}"
        for name in SPLITS
        for flag, size, count in zip(flags, sizes[name], counts[name])
        if size >= 1 and count == 0
    ]
    if starved:
        raise ValueError(
            f"no compounds from {', '.join(starved)}; a scaffold group may be too "
            "large for the requested frac"
        )
    return df.assign(group=scaffolds, split=assigned)


def umap_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int | None = 42,
    *,
    n_neighbors: int = 30,
    # num dimensions for comparing clusters, 2 used for visualization
    # this could be adjusted more/less
    n_components: int = 5,
    min_cluster_size: int = 10,
    min_samples: int = 10,
) -> pd.DataFrame:
    """Split whole clusters of ECFP4 fingerprints after unsupervised UMAP.

    Following https://umap-learn.readthedocs.io/en/latest/clustering.html,
    use a larger neighborhood and min_dist=0 before HDBSCAN.

    Canonical molecules are clustered once, so repeated/equivalent SMILES stay
    together, including noise. Each noise molecule is its own assignment group;
    the returned `group` column keeps HDBSCAN's -1 noise label. Split sizes
    are approximate because clusters are indivisible. Uses sklearn's HDBSCAN,
    whose min_samples includes the point itself (unlike the tutorial's package).
    """
    _validate_frac(frac)
    if n_neighbors < 2:
        raise ValueError("n_neighbors must be at least 2")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    molecules = {}
    canonical = []
    for smi in df["smiles"]:
        mol = to_mol(smi)
        key = mol_to_smiles(mol)
        canonical.append(key)
        molecules.setdefault(key, mol)
    if len(molecules) < max(3, min_cluster_size, min_samples):
        raise ValueError(
            "UMAP splitting needs at least max(3, min_cluster_size, min_samples) "
            "distinct molecules; lower clustering parameters or use scaffold splitting"
        )

    fingerprints = np.stack([mol_to_ecfp4(mol) for mol in molecules.values()])
    embedding = UMAP(
        n_neighbors=min(n_neighbors, len(molecules) - 1),
        min_dist=0.0,
        n_components=n_components,
        # Jaccard similarity used for comparing overlap between ECFP4 Molecular Fingerprints
        metric="jaccard",
        # Random initialization also works when n_components >= sample count.
        init="random",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(fingerprints)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="leaf",
    ).fit_predict(embedding)
    cluster_by_smiles = dict(zip(molecules, labels))
    clusters = [int(cluster_by_smiles[key]) for key in canonical]
    groups = {}
    for pos, (key, label) in enumerate(zip(canonical, clusters)):
        group_key = (label, key if label == -1 else "")
        groups.setdefault(group_key, []).append(pos)
    ordered = [
        g for _, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    logging.getLogger(__name__).info(
        "UMAP/HDBSCAN: %d clusters, %d noise molecules",
        len(set(labels) - {-1}),
        int(np.sum(labels == -1)),
    )
    if np.all(labels == -1):
        logging.getLogger(__name__).warning(
            "HDBSCAN found only noise; splitting individual molecules. "
            "Consider lowering min_cluster_size or min_samples."
        )
    assigned = _assign_cluster_groups(df, ordered, frac, seed)
    return df.assign(group=clusters, split=assigned)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = {"all": df["split"].value_counts()}
    for col in _source_flags(df):
        rows[col] = df.loc[df[col].astype(bool), "split"].value_counts()
    return pd.DataFrame(rows).T.reindex(columns=SPLITS).fillna(0).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="CSV with a `smiles` column")
    parser.add_argument(
        "--method",
        choices=("scaffold", "umap"),
        default="scaffold",
        help="split method (UMAP defaults to seed 42)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="shuffle scaffold groups with this seed (default: largest groups first)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=30, help="UMAP neighborhood size"
    )
    parser.add_argument(
        "--n-components", type=int, default=5, help="UMAP embedding dimensions"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=10, help="HDBSCAN minimum cluster size"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="HDBSCAN density threshold (includes the point itself)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.csv_path.is_file():
        parser.error(f"{args.csv_path} does not exist")
    df = pd.read_csv(args.csv_path)
    if "smiles" not in df.columns:
        parser.error(f"{args.csv_path} has no `smiles` column")

    try:
        if args.method == "umap":
            out = umap_split(
                df,
                seed=42 if args.seed is None else args.seed,
                n_neighbors=args.n_neighbors,
                n_components=args.n_components,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
            )
        else:
            out = scaffold_split(df, seed=args.seed)
    except ValueError as e:
        parser.error(str(e))
    except ImportError as e:
        parser.error(str(e))
    out_path = args.csv_path.with_name(f"{args.csv_path.stem}-splits.csv")
    write_csv(out[["smiles", "group", "split"]], out_path)
    print(f"\nsplit sizes: {len(out)} molecules")
    print(split_summary(out), end="\n\n")


if __name__ == "__main__":
    main()
