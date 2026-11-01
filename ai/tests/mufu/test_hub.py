from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baseline.results import file_sha256
from mufu import hub
from mufu.ensemble import EnsembleParams, save_ensemble
from mufu.mc_dropout import ConformalParams, save_conformal

SEEDS = range(3)
COMMIT = "abc1234"
DATA = "d" * 64
CONFORMAL = ConformalParams((0.9,), np.ones((2, 1)), np.zeros(2), 2, (0.1,))


def write_run(directory, overrides=({}, {}, {})):
    paths = []
    for seed, override in zip(SEEDS, overrides, strict=True):
        path = directory / f"m_seed_{seed}.pt"
        ckpt = {
            "model": {},
            "calibration": {},
            "git_commit": COMMIT,
            "data_sha256": DATA,
        }
        torch.save({**ckpt, "seed": seed, **override}, path)
        save_conformal(CONFORMAL, path)
        paths.append(path)
    members = tuple((p.name, file_sha256(p)) for p in paths)
    save_ensemble(EnsembleParams(CONFORMAL, members), directory, "m")


def test_load_version_lists_the_full_set(tmp_path):
    write_run(tmp_path)

    version = hub.load_version(tmp_path, "m", SEEDS)

    assert version.git_commit == COMMIT
    assert sorted(p.name for p in version.files) == sorted(
        [f"m_seed_{s}.pt" for s in SEEDS]
        + [f"m_seed_{s}_mc_dropout.json" for s in SEEDS]
        + ["m_ensemble.json"]
    )


def stale_ensemble(directory):
    path = directory / "m_seed_0.pt"
    torch.save({**torch.load(path), "seed": 9}, path)
    save_conformal(CONFORMAL, path)


@pytest.mark.parametrize(
    ("overrides", "corrupt", "match"),
    [
        (({}, {}, {"git_commit": "other"}), None, "different runs"),
        (({"git_commit": f"{COMMIT}-dirty"},) * 3, None, "uncommitted"),
        (({}, {"calibration": None}, {}), None, "m_seed_1.pt has no model"),
        (({}, {}, {}), stale_ensemble, "refit the ensemble"),
    ],
)
def test_load_version_rejects_unpublishable_sets(tmp_path, overrides, corrupt, match):
    write_run(tmp_path, overrides)
    if corrupt:
        corrupt(tmp_path)

    with pytest.raises(ValueError, match=match):
        hub.load_version(tmp_path, "m", SEEDS)


def results_rows(model, value, commit=COMMIT, split_id="default"):
    return [
        {
            "model": model,
            "dataset": dataset,
            "endpoint": endpoint,
            "split_id": split_id,
            "split": "test",
            "metric": metric,
            "seed": seed,
            "value": value + seed * 0.01,
            "encoder": "e",
            "n_labeled": 10,
            "n_positive": 5,
            "params": "{}",
            "data_sha256": DATA,
            "git_commit": commit,
            "date": "2026-10-29",
        }
        for dataset, endpoint, metric, _ in hub.CARD_METRICS
        for seed in SEEDS
    ]


def test_model_card_uses_this_runs_metrics():
    version = hub.Version("m", COMMIT, DATA, "e", ())
    results = pd.DataFrame(
        results_rows("m", 0.8)
        + results_rows("m", 0.1, commit="old")
        + results_rows("rf_bioactivity", 0.5, commit="rf")
        + results_rows("rf_bioactivity", 0.7, commit="rf", split_id="scaffold-0")
    )
    ensemble = pd.DataFrame(results_rows("m", 0.9)).assign(ensemble_sha256="e")

    card = hub.model_card(version, "v1.0.0", results, ensemble, "org/m")

    assert {r.metric_value for r in card.data.eval_results} == {0.9}
    assert len(card.data.eval_results) == len(hub.CARD_METRICS)
    index = card.data.to_dict()["model-index"][0]["results"]
    assert len(index) == len({endpoint for _, endpoint, _, _ in hub.CARD_METRICS})
    assert "| 0.900 | 0.810 ± 0.010 | 0.510 |" in card.text

    with pytest.raises(ValueError, match="rerun fit_ensemble"):
        hub.model_card(
            version, "v1.0.0", results, ensemble.assign(ensemble_sha256="x"), "org/m"
        )
    with pytest.raises(ValueError, match="no m rows"):
        hub.model_card(
            version,
            "v1.0.0",
            results[results["git_commit"] != COMMIT],
            ensemble,
            "org/m",
        )


class FakeApi:
    def __init__(self, files):
        self.files, self.tags, self.operations = files, [], None

    def create_repo(self, repo_id, exist_ok):
        pass

    def list_repo_refs(self, repo_id):
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def list_repo_files(self, repo_id):
        return self.files

    def create_commit(self, repo_id, operations, commit_message):
        self.operations = operations
        return SimpleNamespace(oid="sha")

    def create_tag(self, repo_id, tag, revision):
        self.tags.append(tag)


def test_upload_replaces_every_file_and_tags_once(tmp_path):
    write_run(tmp_path)
    version = hub.load_version(tmp_path, "m", SEEDS)
    api = FakeApi([".gitattributes", "README.md", "m_seed_3.pt"])

    with pytest.raises(ValueError, match="MAJOR"):
        hub.upload_version(api, "org/m", version, "1.0.0", "card")
    hub.upload_version(api, "org/m", version, "v1.0.0", "card")

    added = {
        op.path_in_repo: op for op in api.operations if hasattr(op, "path_or_fileobj")
    }
    deleted = {op.path_in_repo for op in api.operations} - set(added)
    assert set(added) == {p.name for p in version.files} | {"README.md", "version.json"}
    assert deleted == {"m_seed_3.pt"}
    assert api.tags == ["v1.0.0"]
    with pytest.raises(ValueError, match="already has"):
        hub.upload_version(api, "org/m", version, "v1.0.0", "card")

    # what a download of the tag would get
    (tmp_path / "version.json").write_bytes(added["version.json"].path_or_fileobj)
    assert hub.load_tagged_version(tmp_path, "v1.0.0", "m", SEEDS) == version
    with pytest.raises(ValueError, match="not v1.1.0"):
        hub.load_tagged_version(tmp_path, "v1.1.0", "m", SEEDS)
