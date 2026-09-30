"""
Assign each compound to train, val, or test with a Bemis-Murcko scaffold split.

Compounds that share a scaffold always land in the same split. Acyclic compounds have an
empty scaffold, so each one is its own group. You should split the combined table rather
than each source that way a compound stays in one split across every task.

How to use as CLI tool (from `ai/`, with `PYTHONPATH=src`):

    python -m pipeline.split data/combined.csv            # largest scaffold groups go to train
    python -m pipeline.split data/combined.csv --seed 0   # balanced split, shuffled with the seed

Writes `smiles`, `scaffold`, and `split` to `<csv stem>-splits.csv` next to the input, then
prints split sizes overall for each `in_*` column. From Python, use `scaffold_split()`.
"""

import argparse
import logging
import math
import random
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold

from featurize.smiles import mol_to_smiles
from pipeline.cache import write_csv
from pipeline.preprocess import to_mol

log = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]
DEFAULT_FRAC = (0.8, 0.1, 0.1)


def murcko_scaffold(smiles: str) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(to_mol(smiles))
    return mol_to_smiles(scaffold) if scaffold.GetNumAtoms() else ""


def _validate_frac(frac: Sequence[float]) -> None:
    if len(frac) != 3:
        raise ValueError(f"frac needs 3 values (train, val, test), got {len(frac)}")
    if any(f < 0 for f in frac):
        raise ValueError(f"frac can't be negative, got {tuple(frac)}")
    if not math.isclose(sum(frac), 1.0):
        raise ValueError(f"frac must sum to 1, got {sum(frac)}")


def _scaffold_groups(scaffolds: Sequence[str]) -> list[list[int]]:
    by_scaffold: dict[str, list[int]] = {}
    groups = []
    for pos, scaffold in enumerate(scaffolds):
        if scaffold:
            by_scaffold.setdefault(scaffold, []).append(pos)
        else:
            groups.append([pos])
    return sorted(
        [*by_scaffold.values(), *groups],
        key=lambda g: (-len(g), scaffolds[g[0]], g[0]),
    )


def _order_groups(
    groups: list[list[int]], val_size: float, test_size: float, seed: int | None
) -> list[list[int]]:
    if seed is None:
        return groups
    rng = random.Random(seed)
    big = [g for g in groups if len(g) > val_size / 2 or len(g) > test_size / 2]
    small = [g for g in groups if len(g) <= val_size / 2 and len(g) <= test_size / 2]
    rng.shuffle(big)
    rng.shuffle(small)
    return big + small


def scaffold_split(
    df: pd.DataFrame,
    frac: Sequence[float] = DEFAULT_FRAC,
    seed: int | None = None,
) -> pd.DataFrame:
    _validate_frac(frac)
    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    n = len(scaffolds)
    train_size, val_size, test_size = (f * n for f in frac)

    counts = dict.fromkeys(SPLITS, 0)
    limits = {"train": train_size, "val": val_size}
    assigned = [""] * n
    for group in _order_groups(_scaffold_groups(scaffolds), val_size, test_size, seed):
        name = next(
            (s for s in SPLITS[:2] if counts[s] + len(group) <= limits[s]), "test"
        )
        counts[name] += len(group)
        for pos in group:
            assigned[pos] = name

    return df.assign(scaffold=scaffolds, split=assigned)


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = {"all": df["split"].value_counts()}
    for col in (c for c in df.columns if c.startswith("in_")):
        rows[col] = df.loc[df[col].astype(bool), "split"].value_counts()
    return pd.DataFrame(rows).T.reindex(columns=SPLITS).fillna(0).astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("csv_path", type=Path, help="CSV with a `smiles` column")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="shuffle scaffold groups with this seed (default: largest groups first)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.csv_path.is_file():
        parser.error(f"{args.csv_path} does not exist")
    df = pd.read_csv(args.csv_path)
    if "smiles" not in df.columns:
        parser.error(f"{args.csv_path} has no `smiles` column")

    out = scaffold_split(df, seed=args.seed)
    out_path = args.csv_path.with_name(f"{args.csv_path.stem}-splits.csv")
    write_csv(out[["smiles", "scaffold", "split"]], out_path)
    print(f"\nsplit sizes: {len(out)} molecules")
    print(split_summary(out), end="\n\n")


if __name__ == "__main__":
    main()
