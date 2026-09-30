import sys

import pandas as pd
import pytest

from pipeline import split as sp

BENZENE = "c1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
BIPHENYL = "c1ccc(-c2ccccc2)cc1"

# scaffolds checked against RDKit's MurckoScaffold
KNOWN_SCAFFOLDS = [
    ("c1ccccc1", BENZENE),
    ("Cc1ccccc1", BENZENE),
    ("CCc1ccccc1", BENZENE),
    ("Oc1ccccc1", BENZENE),
    ("OC(=O)c1ccccc1", BENZENE),
    ("CC(=O)Oc1ccccc1C(=O)O", BENZENE),
    ("O=C([O-])c1ccccc1.[Na+]", BENZENE),
    ("C1CCCCC1", "C1CCCCC1"),
    ("CC1CCCCC1", "C1CCCCC1"),
    ("O=C1CCCCC1", "O=C1CCCCC1"),
    ("c1ccncc1", "c1ccncc1"),
    ("Cc1ccncc1", "c1ccncc1"),
    ("c1ccc2ccccc2c1", NAPHTHALENE),
    ("Cc1cccc2ccccc12", NAPHTHALENE),
    ("c1ccc(-c2ccccc2)cc1", BIPHENYL),
    ("CCc1ccc(-c2ccccc2)cc1", BIPHENYL),
    ("c1ccc(Cc2ccccc2)cc1", "c1ccc(Cc2ccccc2)cc1"),
    ("c1ccc2[nH]ccc2c1", "c1ccc2[nH]ccc2c1"),
    ("CCO", ""),
    ("CCN", ""),
    ("CCC", ""),
    ("CCCC", ""),
]
BENZENES = ["c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "CCc1ccccc1"]


def rings(count):
    return ["C1" + "C" * (n - 1) + "1" for n in range(3, 3 + count)]


def alkanes(count):
    return ["C" * n for n in range(1, 1 + count)]


def table(smiles):
    return pd.DataFrame({"smiles": smiles, "label": range(len(smiles))})


@pytest.mark.parametrize(("smiles", "expected"), KNOWN_SCAFFOLDS)
def test_murcko_scaffold(smiles, expected):
    assert sp.murcko_scaffold(smiles) == expected


def test_murcko_scaffold_rejects_unparseable_smiles():
    with pytest.raises(ValueError):
        sp.murcko_scaffold("not a smiles")


def test_compounds_sharing_a_scaffold_get_the_same_scaffold_string():
    out = sp.scaffold_split(table(BENZENES + rings(6)))
    assert out["scaffold"].iloc[:4].eq(BENZENE).all()
    assert out["scaffold"].nunique() == 7


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_every_compound_gets_one_split_and_no_scaffold_leaks(seed):
    df = table(BENZENES + rings(12) + ["CCO", "CCN", "Cc1ccncc1", "c1ccncc1"])
    out = sp.scaffold_split(df, seed=seed)
    assert out["split"].isin(sp.SPLITS).all()
    cyclic = out[out["scaffold"] != ""]
    assert (cyclic.groupby("scaffold")["split"].nunique() == 1).all()


def test_returns_a_copy_with_only_scaffold_and_split_added():
    df = table(BENZENES + rings(6)).assign(in_tox21=True)
    before = df.copy()
    out = sp.scaffold_split(df)
    pd.testing.assert_frame_equal(df, before)
    assert out.columns.tolist() == [*before.columns, "scaffold", "split"]
    pd.testing.assert_frame_equal(out[before.columns], before)


def test_keeps_the_input_index():
    df = table(rings(10)).set_axis(range(100, 110))
    assert sp.scaffold_split(df).index.tolist() == list(range(100, 110))


def test_default_split_is_80_10_10():
    out = sp.scaffold_split(table(rings(10)))
    assert out["split"].value_counts().to_dict() == {"train": 8, "val": 1, "test": 1}


def test_custom_frac():
    out = sp.scaffold_split(table(rings(10)), frac=(0.6, 0.2, 0.2))
    assert out["split"].value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}


def test_without_seed_the_largest_group_goes_to_train():
    out = sp.scaffold_split(table(BENZENES + rings(6)))
    assert (out.loc[out["scaffold"] == BENZENE, "split"] == "train").all()


def test_with_seed_groups_bigger_than_half_the_val_size_go_to_train_first():
    out = sp.scaffold_split(table(BENZENES + rings(16)), seed=3)
    assert (out.loc[out["scaffold"] == BENZENE, "split"] == "train").all()


@pytest.mark.parametrize("seed", [None, 0])
def test_acyclic_compounds_are_their_own_groups(seed):
    out = sp.scaffold_split(table(alkanes(10)), seed=seed)
    assert out["scaffold"].eq("").all()
    assert out["split"].value_counts().to_dict() == {"train": 8, "val": 1, "test": 1}


def test_repeated_acyclic_compound_stays_in_one_split():
    out = sp.scaffold_split(table(alkanes(10) + ["CCO"] * 5 + ["OCC"] * 5))
    assert out["split"].iloc[10:].nunique() == 1


def test_raises_when_a_split_comes_out_empty():
    with pytest.raises(ValueError, match="train, val split"):
        sp.scaffold_split(table(BENZENES))


def test_split_with_zero_frac_can_be_empty():
    out = sp.scaffold_split(table(rings(10)), frac=(0.9, 0.1, 0.0))
    assert "test" not in set(out["split"])


@pytest.mark.parametrize("seed", [None, 0])
def test_same_table_and_seed_give_the_same_split(seed):
    df = table(BENZENES + rings(16) + alkanes(6))
    first = sp.scaffold_split(df, seed=seed)
    pd.testing.assert_frame_equal(first, sp.scaffold_split(df, seed=seed))


def test_different_seeds_give_different_splits():
    df = table(rings(20))
    a = sp.scaffold_split(df, seed=0)["split"]
    b = sp.scaffold_split(df, seed=1)["split"]
    assert not a.equals(b)


@pytest.mark.parametrize(
    "frac",
    [(0.8, 0.1), (0.7, 0.1, 0.1), (0.9, 0.2, -0.1), (0.5, 0.5, 0.5, 0.0)],
)
def test_bad_frac_raises(frac):
    with pytest.raises(ValueError):
        sp.scaffold_split(table(rings(4)), frac=frac)


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["split", *map(str, args)])
    sp.main()


def test_cli_writes_only_smiles_scaffold_and_split(tmp_path, monkeypatch):
    csv = tmp_path / "combined.csv"
    table(BENZENES + rings(16)).assign(in_a=True).to_csv(csv, index=False)
    run_cli(monkeypatch, csv, "--seed", 0)
    out = pd.read_csv(tmp_path / "combined-splits.csv")
    assert out.columns.tolist() == ["smiles", "scaffold", "split"]
    assert len(out) == 20


def test_cli_prints_sizes_overall_and_per_source(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "combined.csv"
    table(rings(10)).assign(in_a=True, in_b=[True] * 5 + [False] * 5).to_csv(
        csv, index=False
    )
    run_cli(monkeypatch, csv)
    printed = capsys.readouterr().out
    assert "all" in printed
    assert "in_a" in printed
    assert "in_b" in printed


def test_cli_single_source_csv_has_no_per_source_rows(tmp_path, monkeypatch, capsys):
    csv = tmp_path / "bace1.csv"
    table(rings(10)).to_csv(csv, index=False)
    run_cli(monkeypatch, csv)
    assert "in_" not in capsys.readouterr().out


def test_cli_accepts_a_relative_path(tmp_path, monkeypatch):
    table(rings(10)).to_csv(tmp_path / "data.csv", index=False)
    monkeypatch.chdir(tmp_path)
    run_cli(monkeypatch, "data.csv")
    assert (tmp_path / "data-splits.csv").exists()


def test_cli_missing_file_exits(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, tmp_path / "nope.csv")


def test_cli_missing_smiles_column_exits(tmp_path, monkeypatch):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"mol": ["CCO"]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, csv)
    assert not (tmp_path / "bad-splits.csv").exists()


def test_split_summary_counts_each_source():
    df = table(rings(10)).assign(in_a=True, in_b=[True] * 2 + [False] * 8)
    summary = sp.split_summary(sp.scaffold_split(df))
    assert summary.index.tolist() == ["all", "in_a", "in_b"]
    assert summary.columns.tolist() == sp.SPLITS
    assert summary.loc["all"].sum() == 10
    assert summary.loc["in_a"].sum() == 10
    assert summary.loc["in_b"].sum() == 2
