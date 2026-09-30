"""
Assign each compound to train, val, or test with a Bemis-Murcko scaffold split.

Compounds that share a scaffold always land in the same split. Scaffolds ignore
stereochemistry, so stereoisomers of a ring system share one. Acyclic compounds have an
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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

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


def _scaffold_groups(
    smiles: Sequence[str], scaffolds: Sequence[str]
) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for pos, (smi, scaffold) in enumerate(zip(smiles, scaffolds)):
        key = scaffold or mol_to_smiles(to_mol(smi))
        groups.setdefault(key, []).append(pos)
    return [g for _, g in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))]


def scaffold_split(
    df: pd.DataFrame,
    frac: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int | None = None,
) -> pd.DataFrame:
    _validate_frac(frac)
    scaffolds = [murcko_scaffold(s) for s in df["smiles"]]
    sizes = dict(zip(SPLITS, (f * len(scaffolds) for f in frac)))

    groups = _scaffold_groups(df["smiles"], scaffolds)
    if seed is not None:
        rng = random.Random(seed)
        half = min(sizes["val"], sizes["test"]) / 2
        big = [g for g in groups if len(g) > half]
        small = [g for g in groups if len(g) <= half]
        rng.shuffle(big)
        rng.shuffle(small)
        groups = big + small

    counts = dict.fromkeys(SPLITS, 0)
    assigned = [""] * len(scaffolds)
    for group in groups:
        name = next(
            (s for s in SPLITS[:2] if counts[s] + len(group) <= sizes[s]), "test"
        )
        counts[name] += len(group)
        for pos in group:
            assigned[pos] = name

    empty = [name for name in SPLITS if sizes[name] > 0 and counts[name] == 0]
    if empty:
        raise ValueError(
            f"{', '.join(empty)} split(s) came out empty; a scaffold group may be "
            "too large for the requested frac"
        )
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
        help="shuffle scaffold groups with this seed (default: largest groups first)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.csv_path.is_file():
        parser.error(f"{args.csv_path} does not exist")
    df = pd.read_csv(args.csv_path)
    if "smiles" not in df.columns:
        parser.error(f"{args.csv_path} has no `smiles` column")

    try:
        out = scaffold_split(df, seed=args.seed)
    except ValueError as e:
        parser.error(str(e))
    out_path = args.csv_path.with_name(f"{args.csv_path.stem}-splits.csv")
    write_csv(out[["smiles", "scaffold", "split"]], out_path)
    print(f"\nsplit sizes: {len(out)} molecules")
    print(split_summary(out), end="\n\n")


if __name__ == "__main__":
    main()
