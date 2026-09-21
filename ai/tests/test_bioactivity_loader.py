import io
import json
import re
import sys
import urllib.request

import pandas as pd
import pytest

import bioactivity_loader as bl


def rec(cid="CHEMBL1", smiles="CCO", pchembl="7.0", **overrides):
    """One activity record as the ChEMBL API returns it (passes every filter)."""
    record = {
        "molecule_chembl_id": cid,
        "canonical_smiles": smiles,
        "pchembl_value": pchembl,
        "standard_relation": "=",
        "standard_units": "nM",
        "assay_type": "B",
        "potential_duplicate": 0,
        "data_validity_comment": None,
    }
    return record | overrides


def clean(*records):
    return bl.clean_activities(pd.DataFrame(list(records)))


PAGE_1 = {
    "activities": [
        rec("C1", "CCO", "7.0"),
        rec("C1", "CCO", "8.0"),  # repeat measurement of C1
        rec("C2", "c1ccccc1CC(=O)O", "5.0"),
        rec("C3", "CCN.Cl", "6.5"),  # ethylamine hydrochloride
        rec("C4", "CCN", "6.0", standard_relation=">"),  # censored, dropped
    ],
    "page_meta": {"next": "/chembl/api/data/activity.json?page=2", "total_count": 9},
}
PAGE_2 = {
    "activities": [
        rec("C5", "c1ccccc1CC(=O)O", "5.4"),  # same structure as C2
        rec("C6", "CCN", "6.0"),  # ethylamine: same parent as C3's salt
        rec("C7", "not_a_smiles", "6.0"),  # RDKit can't parse, dropped
        rec("C8", "CCCC", None),  # no pChEMBL, dropped
    ],
    "page_meta": {"next": None, "total_count": 9},
}


@pytest.fixture
def fake_chembl(monkeypatch):
    """Serve PAGE_1 then PAGE_2 in place of ChEMBL; returns the list of URLs hit."""
    urls = []

    def fake_urlopen(url, timeout=None):
        urls.append(url)
        page = PAGE_2 if "page=2" in url else PAGE_1
        return io.BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return urls


@pytest.fixture
def no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("this test should not touch the network")

    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CC(=O)[O-].[Na+]", "CC(=O)[O-]"),  # strips counterion, keeps the charge
        ("Cl.CCN", "CCN"),  # strips HCl salt
        ("[Na+].[Cl-].c1ccccc1CC(=O)O", "O=C(O)Cc1ccccc1"),  # keeps the big fragment
        ("C[C@H](N)C(=O)O", "C[C@H](N)C(=O)O"),  # stereo preserved
    ],
)
def test_standardize_smiles(raw, expected):
    assert bl.standardize_smiles(raw) == expected


def test_standardize_smiles_returns_none_when_unparseable():
    assert bl.standardize_smiles("not_a_smiles") is None


def test_clean_keeps_a_good_row():
    out = clean(rec())
    assert len(out) == 1
    assert out.loc[0, "pIC50"] == 7.0


@pytest.mark.parametrize(
    "bad",
    [
        {"standard_relation": ">"},
        {"standard_relation": "<"},
        {"standard_relation": None},
        {"standard_units": "uM"},
        {"assay_type": "F"},
        {"potential_duplicate": 1},
        {"data_validity_comment": "Outside typical range"},
        {"pchembl_value": None},
        {"pchembl_value": "not a number"},
        {"canonical_smiles": None},
        {"canonical_smiles": "not_a_smiles"},
    ],
    ids=lambda d: "-".join(f"{k}={v}" for k, v in d.items()),
)
def test_clean_drops_bad_rows(bad):
    out = clean(rec("GOOD", "CCO"), rec("BAD", "CCC", **bad))
    assert out["molecule_chembl_id"].tolist() == ["GOOD"]


def test_clean_converts_string_pchembl_to_numbers():
    # the ChEMBL API returns pchembl_value as a string
    out = clean(rec(pchembl="6.38"))
    assert out["pIC50"].dtype == float
    assert out.loc[0, "pIC50"] == pytest.approx(6.38)


def test_clean_aggregates_repeat_measurements():
    out = clean(
        rec("C1", "CCO", "6.0"),
        rec("C1", "CCO", "7.0"),
        rec("C1", "CCO", "9.0"),
    ).set_index("smiles")
    assert out.loc["CCO", "pIC50"] == 7.0  # median, not mean
    assert out.loc["CCO", "n_meas"] == 3
    assert out.loc["CCO", "pIC50_spread"] == pytest.approx(3.0)


def test_clean_single_measurement_has_zero_spread():
    out = clean(rec())
    assert out.loc[0, "n_meas"] == 1
    assert out.loc[0, "pIC50_spread"] == 0


def test_clean_merges_salt_and_parent_under_different_chembl_ids():
    out = clean(
        rec("FREE_BASE", "CCN", "6.0"),
        rec("HCL_SALT", "CCN.Cl", "6.5"),
    )
    assert len(out) == 1
    assert out.loc[0, "smiles"] == "CCN"
    assert out.loc[0, "n_meas"] == 2
    assert out.loc[0, "pIC50"] == pytest.approx(6.25)


def test_clean_keeps_enantiomers_separate():
    # BACE-1 activity is stereo-sensitive, so standardization must not merge these
    out = clean(
        rec("R", "C[C@H](N)C(=O)O", "8.0"),
        rec("S", "C[C@@H](N)C(=O)O", "5.0"),
    )
    assert len(out) == 2


def test_clean_output_is_one_row_per_smiles_with_expected_columns():
    out = clean(rec("C2", "CCO", "7.0"), rec("C1", "CCO", "8.0"), rec("C3", "CCC"))
    assert list(out.columns) == bl.CSV_COLUMNS
    assert out["smiles"].is_unique
    assert out["molecule_chembl_id"].is_monotonic_increasing


def test_fetch_follows_pagination(fake_chembl):
    raw = bl.fetch_activities()
    assert len(fake_chembl) == 2
    assert len(raw) == len(PAGE_1["activities"]) + len(PAGE_2["activities"])
    # second URL is built from the relative `next` path the API returns
    assert (
        fake_chembl[1] == "https://www.ebi.ac.uk/chembl/api/data/activity.json?page=2"
    )


def test_fetch_requests_bace1_ic50(fake_chembl):
    bl.fetch_activities()
    assert "target_chembl_id=CHEMBL4822" in fake_chembl[0]
    assert "standard_type=IC50" in fake_chembl[0]


def test_fetch_keeps_only_the_needed_fields(monkeypatch):
    page = {
        "activities": [rec() | {"some_other_field": "x"}],
        "page_meta": {"next": None, "total_count": 1},
    }
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=None: io.BytesIO(json.dumps(page).encode()),
    )
    assert list(bl.fetch_activities().columns) == bl.API_FIELDS


def serve(monkeypatch, *pages):
    """Serve the given payloads in order in place of ChEMBL."""
    it = iter(pages)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda url, timeout=None: io.BytesIO(json.dumps(next(it)).encode()),
    )


def test_fetch_rejects_a_truncated_download(monkeypatch):
    # `next` is None after 5 of the 9 rows ChEMBL reported (partial download)
    page = PAGE_1 | {"page_meta": {"next": None, "total_count": 9}}
    serve(monkeypatch, page)
    with pytest.raises(RuntimeError, match="reported 9"):
        bl.fetch_activities()


@pytest.mark.parametrize(
    "bad_page",
    [
        {"page_meta": {"next": None, "total_count": 1}},  # no activities
        {"activities": None, "page_meta": {"next": None, "total_count": 1}},
        {"activities": [rec()]},  # no page_meta
        {"activities": [rec()], "page_meta": {"next": None}},  # no total_count
    ],
)
def test_fetch_rejects_pages_with_the_wrong_shape(monkeypatch, bad_page):
    serve(monkeypatch, bad_page)
    with pytest.raises(TypeError, match="unexpected ChEMBL response"):
        bl.fetch_activities()


@pytest.mark.parametrize("bad_next", [5, "elsewhere"])
def test_fetch_rejects_a_bad_next_link(monkeypatch, bad_next):
    page = {"activities": [rec()], "page_meta": {"next": bad_next, "total_count": 1}}
    serve(monkeypatch, page)
    with pytest.raises(RuntimeError, match="bad next link"):
        bl.fetch_activities()


def test_fetch_rejects_records_missing_a_field(monkeypatch):
    record = rec()
    del record["pchembl_value"]
    serve(
        monkeypatch,
        {"activities": [record], "page_meta": {"next": None, "total_count": 1}},
    )
    with pytest.raises(RuntimeError, match="pchembl_value"):
        bl.fetch_activities()


def test_load_downloads_when_csv_is_missing(fake_chembl, tmp_path):
    csv = tmp_path / "nested" / "chembl-bace-1.csv"  # parent dir doesn't exist yet
    df = bl.load_bace1(csv_path=csv)

    assert csv.exists()
    assert len(fake_chembl) == 2
    # C1 x2 -> CCO; C2 + C5 -> phenylacetic acid; C3 salt + C6 -> ethylamine
    assert sorted(df["smiles"]) == sorted(["CCO", "O=C(O)Cc1ccccc1", "CCN"])
    by_smiles = df.set_index("smiles")
    assert by_smiles.loc["CCO", "pIC50"] == 7.5
    assert by_smiles.loc["O=C(O)Cc1ccccc1", "pIC50"] == pytest.approx(5.2)
    assert by_smiles.loc["CCN", "pIC50"] == pytest.approx(6.25)


def test_load_uses_the_cached_csv_without_network(fake_chembl, tmp_path):
    csv = tmp_path / "chembl-bace-1.csv"
    first = bl.load_bace1(csv_path=csv)
    calls_after_download = len(fake_chembl)

    second = bl.load_bace1(csv_path=csv)
    assert len(fake_chembl) == calls_after_download  # no new requests
    pd.testing.assert_frame_equal(first, second)


def test_load_never_downloads_if_csv_exists(no_network, tmp_path):
    csv = tmp_path / "chembl-bace-1.csv"
    pd.DataFrame(
        {
            "molecule_chembl_id": ["C1"],
            "smiles": ["CCO"],
            "pIC50": [7.0],
            "n_meas": [1],
            "pIC50_spread": [0.0],
        }
    ).to_csv(csv, index=False)
    assert len(bl.load_bace1(csv_path=csv)) == 1


def test_load_refresh_redownloads(fake_chembl, tmp_path):
    csv = tmp_path / "chembl-bace-1.csv"
    bl.load_bace1(csv_path=csv)
    bl.load_bace1(csv_path=csv, refresh=True)
    assert len(fake_chembl) == 4  # 2 pages, twice


def test_failed_download_leaves_no_csv(monkeypatch, tmp_path):
    def flaky(url, timeout=None):
        if "page=2" in url:
            raise TimeoutError("connection dropped")
        return io.BytesIO(json.dumps(PAGE_1).encode())

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    csv = tmp_path / "chembl-bace-1.csv"

    with pytest.raises(TimeoutError):
        bl.load_bace1(csv_path=csv)
    # a later run must retry the download, not trust a partial file
    assert not csv.exists()
    assert list(tmp_path.iterdir()) == []


def test_load_rejects_a_csv_with_missing_columns(no_network, tmp_path):
    csv = tmp_path / "chembl-bace-1.csv"
    pd.DataFrame({"smiles": ["CCO"], "pIC50": [7.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        bl.load_bace1(csv_path=csv)


@pytest.fixture
def cached_csv(no_network, tmp_path):
    csv = tmp_path / "chembl-bace-1.csv"
    pd.DataFrame(
        {
            "molecule_chembl_id": ["A", "B", "C"],
            "smiles": ["CCO", "CCC", "CCCC"],
            "pIC50": [5.0, 6.0, 7.0],
            "n_meas": [1, 1, 1],
            "pIC50_spread": [0.0, 0.0, 0.0],
        }
    ).to_csv(csv, index=False)
    return csv


def test_default_threshold_is_six_and_inclusive(cached_csv):
    df = bl.load_bace1(csv_path=cached_csv)
    assert df["active"].tolist() == [0, 1, 1]  # 6.0 counts as active


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [(5.0, [1, 1, 1]), (6.5, [0, 0, 1]), (8.0, [0, 0, 0])],
)
def test_threshold_is_an_argument(cached_csv, threshold, expected):
    assert (
        bl.load_bace1(threshold=threshold, csv_path=cached_csv)["active"].tolist()
        == expected
    )


def test_downloaded_csv_has_only_the_cleaned_columns(fake_chembl, tmp_path):
    # the label is derived at load time, so it must not be baked into the cache
    csv = tmp_path / "chembl-bace-1.csv"
    bl.load_bace1(csv_path=csv)
    assert list(pd.read_csv(csv).columns) == bl.CSV_COLUMNS


def test_preprocess_does_not_mutate_its_input():
    df = pd.DataFrame({"pIC50": [5.0, 7.0]})
    bl.preprocess(df, 6.0)
    assert list(df.columns) == ["pIC50"]


def test_main_forwards_flags_and_prints_a_summary(monkeypatch, capsys):
    seen = {}

    def fake_load(threshold, refresh):
        seen.update(threshold=threshold, refresh=refresh)
        return pd.DataFrame({"pIC50": [5.0, 7.0], "active": [0, 1]})

    monkeypatch.setattr(bl, "load_bace1", fake_load)
    monkeypatch.setattr(
        sys, "argv", ["bioactivity_loader.py", "--refresh", "--threshold", "6.5"]
    )
    bl.main()

    assert seen == {"threshold": 6.5, "refresh": True}
    out = capsys.readouterr().out
    assert "2 molecules" in out
    # pandas labels the median row "50%"; the two fake pIC50s (5.0, 7.0) give 6.0
    assert re.search(r"50%\s+6\.000", out)


def test_main_defaults(monkeypatch):
    seen = {}

    def fake_load(threshold, refresh):
        seen.update(threshold=threshold, refresh=refresh)
        return pd.DataFrame({"pIC50": [6.0], "active": [1]})

    monkeypatch.setattr(bl, "load_bace1", fake_load)
    monkeypatch.setattr(sys, "argv", ["bioactivity_loader.py"])
    bl.main()
    assert seen == {"threshold": bl.PIC50_ACTIVE_THRESHOLD, "refresh": False}


@pytest.mark.network
def test_real_chembl_download(tmp_path):
    df = bl.load_bace1(csv_path=tmp_path / "chembl-bace-1.csv")
    # loose bounds: ChEMBL releases shift the exact counts
    assert 5_000 < len(df) < 15_000
    assert df["smiles"].is_unique
    assert df["pIC50"].between(2, 12).all()
    assert df["active"].isin([0, 1]).all()
    assert (df["n_meas"] >= 1).all()
    assert "." not in "".join(df["smiles"])  # no multi-fragment structures left
