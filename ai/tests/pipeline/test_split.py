import sys

import numpy as np
import pandas as pd
import pytest

from pipeline import split as sp

BENZENE = "c1ccccc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
BIPHENYL = "c1ccc(-c2ccccc2)cc1"
DECALIN = "C1CCC2CCCCC2C1"
STILBENE = "C(=Cc1ccccc1)c1ccccc1"

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
    ("C1CC[C@H]2CCCC[C@@H]2C1", DECALIN),
    ("C1CC[C@H]2CCCC[C@H]2C1", DECALIN),
    ("c1ccc(/C=C/c2ccccc2)cc1", STILBENE),
    ("c1ccc(/C=C\\c2ccccc2)cc1", STILBENE),
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
    assert out["group"].iloc[:4].eq(BENZENE).all()
    assert out["group"].nunique() == 7


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_every_compound_gets_one_split_and_no_scaffold_leaks(seed):
    df = table(BENZENES + rings(12) + ["CCO", "CCN", "Cc1ccncc1", "c1ccncc1"])
    out = sp.scaffold_split(df, seed=seed)
    assert out["split"].isin(sp.SPLITS).all()
    cyclic = out[out["group"] != ""]
    assert (cyclic.groupby("group")["split"].nunique() == 1).all()


def test_returns_a_copy_with_only_group_and_split_added():
    df = table(BENZENES + rings(6)).assign(in_tox21=True)
    before = df.copy()
    out = sp.scaffold_split(df)
    pd.testing.assert_frame_equal(df, before)
    assert out.columns.tolist() == [*before.columns, "group", "split"]
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
    assert (out.loc[out["group"] == BENZENE, "split"] == "train").all()


def test_with_seed_groups_bigger_than_half_the_val_size_go_to_train_first():
    out = sp.scaffold_split(table(BENZENES + rings(16)), seed=3)
    assert (out.loc[out["group"] == BENZENE, "split"] == "train").all()


@pytest.mark.parametrize("seed", [None, 0])
def test_acyclic_compounds_are_their_own_groups(seed):
    out = sp.scaffold_split(table(alkanes(10)), seed=seed)
    assert out["group"].eq("").all()
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


def test_cli_writes_only_smiles_group_and_split(tmp_path, monkeypatch):
    csv = tmp_path / "combined.csv"
    table(BENZENES + rings(16)).assign(in_a=True).to_csv(csv, index=False)
    run_cli(monkeypatch, csv, "--seed", 0)
    out = pd.read_csv(tmp_path / "combined-splits.csv")
    assert out.columns.tolist() == ["smiles", "group", "split"]
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


def test_each_source_gets_its_own_80_10_10():
    df = table(rings(20) + alkanes(10)).assign(
        in_a=[True] * 20 + [False] * 10, in_b=[False] * 20 + [True] * 10
    )
    summary = sp.split_summary(sp.scaffold_split(df))
    assert summary.loc["in_a"].tolist() == [16, 2, 2]
    assert summary.loc["in_b"].tolist() == [8, 1, 1]


def test_source_left_out_of_a_split_raises():
    df = table(BENZENES + rings(16)).assign(in_a=True, in_b=[True] * 4 + [False] * 16)
    with pytest.raises(ValueError, match="in_b train, in_b val"):
        sp.scaffold_split(df, frac=(0.5, 0.25, 0.25))


@pytest.fixture
def cluster_estimators(monkeypatch):
    """Control clustering so assignment invariants do not depend on UMAP versions."""
    calls = {}

    class UMAP:
        def __init__(self, **kwargs):
            calls["umap"] = kwargs

        def fit_transform(self, fingerprints):
            calls["fingerprints"] = fingerprints.copy()
            embedding = np.arange(len(fingerprints) * 2).reshape(-1, 2)
            calls["embedding"] = embedding
            return embedding

    class HDBSCAN:
        def __init__(self, **kwargs):
            calls["hdbscan"] = kwargs

        def fit_predict(self, embedding):
            assert embedding is calls["embedding"]
            return np.asarray(calls.get("labels", [-1] * len(embedding)))

    monkeypatch.setattr(sp, "UMAP", UMAP)
    monkeypatch.setattr(sp, "HDBSCAN", HDBSCAN)
    return calls


def test_umap_keeps_whole_clusters_and_input_data(cluster_estimators):
    cluster_estimators["labels"] = [0] * 6 + [1] * 2 + [2] * 2
    df = table(alkanes(10)).set_axis([4] * 10)
    before = df.copy(deep=True)
    out = sp.umap_split(
        df, frac=(0.6, 0.2, 0.2), seed=None, min_cluster_size=2, min_samples=2
    )
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(out[before.columns], before)
    assert out.columns.tolist() == [*before.columns, "group", "split"]
    assert out.groupby("group")["split"].nunique().eq(1).all()
    assert out["split"].value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}


def test_umap_uses_fingerprints_and_clustering_parameters(cluster_estimators):
    from featurize.fingerprint import mol_to_ecfp4
    from pipeline.preprocess import to_mol

    df = table(alkanes(10)).assign(target=float("nan"), in_a=True)
    sp.umap_split(
        df, seed=0, n_neighbors=50, n_components=3, min_cluster_size=4, min_samples=2
    )
    assert cluster_estimators["umap"] == {
        "n_neighbors": 9,
        "n_components": 3,
        "min_dist": 0.0,
        "metric": "jaccard",
        "init": "random",
        "random_state": 0,
        "n_jobs": 1,
    }
    assert cluster_estimators["hdbscan"] == {
        "min_cluster_size": 4,
        "min_samples": 2,
        "cluster_selection_method": "leaf",
    }
    np.testing.assert_array_equal(
        cluster_estimators["fingerprints"],
        np.stack([mol_to_ecfp4(to_mol(smi)) for smi in df["smiles"]]),
    )


@pytest.mark.parametrize("noise_label", [-1, 0])
def test_umap_equivalent_smiles_stay_together_even_in_noise(
    cluster_estimators, noise_label
):
    cluster_estimators["labels"] = [-1] * 10 + [noise_label]
    out = sp.umap_split(
        table(alkanes(10) + ["CCO", "OCC", "CCO"]), frac=(0.6, 0.2, 0.2)
    )
    assert len(cluster_estimators["fingerprints"]) == 11
    assert out["split"].iloc[-3:].nunique() == 1
    assert out["group"].iloc[-3:].eq(noise_label).all()
    assert out["split"].isin(sp.SPLITS).all()


def test_umap_noise_is_not_one_giant_cluster(cluster_estimators, caplog):
    out = sp.umap_split(table(alkanes(10)))
    assert out["group"].eq(-1).all()
    assert out["split"].value_counts().to_dict() == {"train": 8, "val": 1, "test": 1}
    assert "found only noise" in caplog.text


def test_umap_approximately_balances_each_source(cluster_estimators):
    cluster_estimators["labels"] = np.arange(30) // 2
    df = table(alkanes(30)).assign(
        in_a=[True] * 20 + [False] * 10,
        in_b=[False] * 20 + [True] * 10,
    )
    out = sp.umap_split(df, frac=(0.6, 0.2, 0.2), seed=None)
    summary = sp.split_summary(out)
    np.testing.assert_allclose(summary.loc["in_a"], [12, 4, 4], atol=2, rtol=0)
    np.testing.assert_allclose(summary.loc["in_b"], [6, 2, 2], atol=2, rtol=0)
    assert out.groupby("group")["split"].nunique().eq(1).all()


def test_umap_targets_include_molecules_without_source_membership(cluster_estimators):
    cluster_estimators["labels"] = [0] * 6 + [1] * 2 + [2] * 2
    df = table(alkanes(10)).assign(in_empty=False)
    out = sp.umap_split(
        df, frac=(0.6, 0.2, 0.2), seed=None, min_cluster_size=2, min_samples=2
    )
    assert out["split"].value_counts().to_dict() == {"train": 6, "val": 2, "test": 2}
    assert out.groupby("group")["split"].nunique().eq(1).all()


def test_umap_rejects_clusters_that_cannot_fit_validation_budget(cluster_estimators):
    cluster_estimators["labels"] = [0] * 4 + [1] * 3 + [2] * 3
    # The default validation budget is one molecule; every whole cluster is larger.
    with pytest.raises(ValueError, match=r"val split\(s\) came out empty"):
        sp.umap_split(table(alkanes(10)), min_cluster_size=2, min_samples=2)


def test_umap_uses_real_hdbscan_on_embedding(monkeypatch):
    rng = np.random.default_rng(0)
    # Each separated blob has exactly min_cluster_size points, so leaf selection
    # cannot divide it into multiple valid clusters as it could a larger blob.
    embedding = np.concatenate(
        [rng.normal(center, 0.01, size=(10, 2)) for center in (0, 100, 200)]
    )

    class UMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, fingerprints):
            assert len(fingerprints) == len(embedding)
            return embedding

    monkeypatch.setattr(sp, "UMAP", UMAP)
    out = sp.umap_split(
        table(alkanes(30)),
        frac=(1 / 3, 1 / 3, 1 / 3),
        seed=None,
        min_cluster_size=10,
        min_samples=2,
    )
    assert out["group"].nunique() == 3
    assert out["group"].ge(0).all()
    assert out.groupby("group")["split"].nunique().eq(1).all()
    assert out["split"].value_counts().to_dict() == {"train": 10, "val": 10, "test": 10}
    for start in (0, 10, 20):
        assert out["group"].iloc[start : start + 10].nunique() == 1


def test_umap_split_with_real_estimators():
    """Exercise molecular featurization, real UMAP, and real HDBSCAN together."""
    smiles = [smi for smi, _ in KNOWN_SCAFFOLDS] + [
        "CCCO",
        "CCCCO",
        "CCCN",
        "CCCCN",
        "CC(=O)N",
        "CCC(=O)N",
        "CC(=O)O",
        "CCC(=O)O",
        "c1ccoc1",
        "Cc1ccoc1",
        "c1ccsc1",
        "Cc1ccsc1",
        "CCO",
        "OCC",
    ]
    df = table(smiles).assign(in_a=True).set_axis(range(100, 100 + len(smiles)))
    before = df.copy(deep=True)
    # Real cluster sizes can vary across versions. Test allocation proportions
    # separately with controlled clusters, and isolate this integration check.
    kwargs = {
        "frac": (1.0, 0.0, 0.0),
        "seed": 42,
        "n_neighbors": 5,
        "min_cluster_size": 2,
        "min_samples": 2,
    }
    out = sp.umap_split(df, **kwargs)
    repeated = sp.umap_split(df, **kwargs)

    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(out[before.columns], before)
    pd.testing.assert_frame_equal(out, repeated)
    assert out.columns.tolist() == [*before.columns, "group", "split"]
    assert pd.api.types.is_integer_dtype(out["group"])
    assert out["group"].ge(-1).all()
    assert out.loc[out["group"] >= 0, "group"].nunique() >= 2
    assert out["split"].eq("train").all()
    # Repeated SMILES and equivalent representations share a real cluster label,
    # including when HDBSCAN marks that molecule as noise.
    ethanol = out[out["smiles"].isin(["CCO", "OCC"])]
    assert len(ethanol) == 3
    assert ethanol["group"].nunique() == 1
    assert ethanol["split"].nunique() == 1


def test_umap_raises_when_too_few_groups_can_populate_requested_splits(
    cluster_estimators,
):
    cluster_estimators["labels"] = [0] * 10
    with pytest.raises(ValueError, match="cluster group"):
        sp.umap_split(table(alkanes(10)))


def test_umap_single_requested_split_accepts_a_single_cluster(cluster_estimators):
    cluster_estimators["labels"] = [0] * 10
    out = sp.umap_split(table(alkanes(10)), frac=(1.0, 0.0, 0.0))
    assert out["group"].eq(0).all()
    assert out["split"].eq("train").all()


def test_umap_zero_fraction_can_be_empty(cluster_estimators):
    out = sp.umap_split(table(alkanes(10)), frac=(0.9, 0.1, 0.0))
    assert "test" not in set(out["split"])


def test_umap_seed_is_reproducible(cluster_estimators):
    df = table(alkanes(20))
    first = sp.umap_split(df, seed=0)
    pd.testing.assert_frame_equal(first, sp.umap_split(df, seed=0))
    assert not first["split"].equals(sp.umap_split(df, seed=1)["split"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"frac": (0.5, 0.1, 0.1)}, "frac"),
        ({"n_neighbors": 1}, "n_neighbors"),
        ({"n_components": 0}, "n_components"),
        ({"min_cluster_size": 1}, "min_cluster_size"),
        ({"min_samples": 0}, "min_samples"),
    ],
)
def test_umap_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        sp.umap_split(table(alkanes(10)), **kwargs)


@pytest.mark.parametrize("smiles", [[], ["CCO", "OCC"] * 10, alkanes(9)])
def test_umap_rejects_too_few_distinct_molecules(smiles):
    with pytest.raises(ValueError, match="distinct molecules"):
        sp.umap_split(table(smiles))


def test_splitters_return_the_same_column_names(cluster_estimators):
    df = table(alkanes(10))
    scaffold = sp.scaffold_split(df)
    umap = sp.umap_split(df)
    assert scaffold.columns.tolist() == umap.columns.tolist()
    assert scaffold.columns.tolist() == [*df.columns, "group", "split"]


def test_cli_umap_writes_group_column(tmp_path, monkeypatch, cluster_estimators):
    csv = tmp_path / "combined.csv"
    table(alkanes(10)).to_csv(csv, index=False)
    run_cli(
        monkeypatch,
        csv,
        "--method",
        "umap",
        "--seed",
        0,
        "--n-neighbors",
        4,
        "--n-components",
        2,
        "--min-cluster-size",
        3,
        "--min-samples",
        2,
    )
    out = pd.read_csv(tmp_path / "combined-splits.csv")
    assert out.columns.tolist() == ["smiles", "group", "split"]
    assert len(out) == 10
    assert out["split"].isin(sp.SPLITS).all()
    assert cluster_estimators["umap"]["random_state"] == 0
    assert cluster_estimators["umap"]["n_neighbors"] == 4
    assert cluster_estimators["umap"]["n_components"] == 2
    assert cluster_estimators["hdbscan"] == {
        "min_cluster_size": 3,
        "min_samples": 2,
        "cluster_selection_method": "leaf",
    }


def test_cli_umap_defaults_to_seed_42(tmp_path, monkeypatch, cluster_estimators):
    csv = tmp_path / "combined.csv"
    table(alkanes(10)).to_csv(csv, index=False)
    run_cli(monkeypatch, csv, "--method", "umap")
    assert cluster_estimators["umap"]["random_state"] == 42
