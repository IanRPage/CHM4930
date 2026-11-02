import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from baseline.results import file_sha256, runs_path
from mufu import hub
from mufu.ensemble import EnsembleParams, save_ensemble
from mufu.mc_dropout import ConformalParams, save_conformal

SEEDS = range(3)
COMMIT = "a" * 40
TREE = "b" * 40
DATA = "d" * 64
CONFORMAL = ConformalParams((0.9,), np.ones((2, 1)), np.zeros(2), 2, (0.1,))


def write_run(directory, overrides=({}, {}, {}), *, legacy=False):
    paths = []
    for seed, override in zip(SEEDS, overrides, strict=True):
        path = directory / f"m_seed_{seed}.pt"
        if legacy:
            ckpt = {
                "model": {},
                "calibration": {},
                "git_commit": COMMIT[:7],
                "data_sha256": DATA,
            }
        else:
            ckpt = {
                "model": {},
                "calibration": {},
                "base_commit": COMMIT,
                "ai_tree": TREE,
                "patch_sha256": "",
                "data_sha256": DATA,
            }
        torch.save({**ckpt, "seed": seed, **override}, path)
        save_conformal(CONFORMAL, path)
        paths.append(path)
    members = tuple((p.name, file_sha256(p)) for p in paths)
    save_ensemble(EnsembleParams(CONFORMAL, members), directory, "m")


def test_load_version_supports_new_and_legacy_checkpoints(tmp_path):
    write_run(tmp_path)
    version = hub.load_version(tmp_path, "m", SEEDS)
    assert version.base_commit == COMMIT
    assert version.ai_tree == TREE
    assert sorted(p.name for p in version.files) == sorted(
        [f"m_seed_{s}.pt" for s in SEEDS]
        + [f"m_seed_{s}_mc_dropout.json" for s in SEEDS]
        + ["m_ensemble.json"]
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    write_run(legacy_dir, legacy=True)
    (legacy_dir / "version.json").write_text(
        json.dumps({"tag": "v1.0.0", "git_commit": COMMIT[:7]}) + "\n"
    )
    legacy_ver = hub.load_tagged_version(legacy_dir, "v1.0.0", "m", SEEDS)
    assert legacy_ver.base_commit == COMMIT[:7]


def stale_ensemble(directory):
    path = directory / "m_seed_0.pt"
    torch.save({**torch.load(path), "seed": 9}, path)
    save_conformal(CONFORMAL, path)


@pytest.mark.parametrize(
    ("overrides", "corrupt", "match"),
    [
        (({}, {}, {"base_commit": "c" * 40}), None, "different runs"),
        (({"base_commit": f"{COMMIT[:7]}-dirty"},) * 3, None, "uncommitted"),
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


def results_rows(model, value, run_id="m_run_1", commit=COMMIT, split_id="default"):
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
            "run_id": run_id,
            "n_seeds": len(SEEDS),
            "params": "{}",
            "data_sha256": DATA,
            "base_commit": commit,
            "ai_tree": TREE,
            "patch_sha256": "",
            "date": "2026-10-30",
        }
        for dataset, endpoint, metric, _ in hub.CARD_METRICS
        for seed in SEEDS
    ]


def test_model_card_selects_by_run_id_even_when_commits_match():
    version = hub.Version(
        model_name="m",
        base_commit=COMMIT,
        ai_tree=TREE,
        patch_sha256="",
        data_sha256=DATA,
        ensemble_sha256="e",
        files=(),
        run_id="m_run_winner",
    )
    # Two runs sharing the exact same base_commit and data_sha256
    results = pd.DataFrame(
        results_rows("m", 0.8, run_id="m_run_winner", commit=COMMIT)
        + results_rows("m", 0.1, run_id="m_run_other", commit=COMMIT)
        + results_rows("rf_bioactivity", 0.5, run_id="rf_bio_default", commit="rf")
        + results_rows(
            "rf_bioactivity",
            0.7,
            run_id="rf_bio_s0",
            commit="rf",
            split_id="scaffold-0",
        )
    )
    ensemble = pd.DataFrame(results_rows("m", 0.9, run_id="m_run_winner")).assign(
        ensemble_sha256="e"
    )

    card = hub.model_card(version, "v1.0.0", results, ensemble, "org/m")

    assert {r.metric_value for r in card.data.eval_results} == {0.9}
    assert len(card.data.eval_results) == len(hub.CARD_METRICS)
    assert "| 0.900 | 0.810 ± 0.010 | 0.510 |" in card.text

    with pytest.raises(ValueError, match="rerun fit_ensemble"):
        hub.model_card(
            version, "v1.0.0", results, ensemble.assign(ensemble_sha256="x"), "org/m"
        )
    with pytest.raises(ValueError, match="no m rows"):
        hub.model_card(
            version,
            "v1.0.0",
            results[results["run_id"] != "m_run_winner"],
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


def write_run_folder(run_dir, model="m", score=6.0):
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "model": model,
                "encoder": "e",
                "split_id": "default",
                "seeds": list(SEEDS),
                "completed_seeds": list(SEEDS),
                "status": "complete",
                "params": "{}",
                "data_sha256": DATA,
                "split_sha256": "s" * 64,
                "base_commit": COMMIT,
                "ai_tree": TREE,
                "patch_sha256": "",
                "date": "2026-10-30",
            }
        )
    )
    (run_dir / "predictions" / "val_seed_0.csv").write_text(
        f"smiles,pIC50_BACE1\nCCO,{score}\n"
    )


def test_upload_includes_run_json_predictions_and_tags_once(tmp_path):
    run_dir = tmp_path / "m_v1_default"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    write_run(ckpt_dir)
    write_run_folder(run_dir, model="m", score=6.0)
    write_run_folder(
        tmp_path / "rf_bioactivity_default", model="rf_bioactivity", score=5.5
    )

    version = hub.load_version(ckpt_dir, "m", SEEDS)
    api = FakeApi([".gitattributes", "README.md", "m_seed_3.pt"])

    with pytest.raises(ValueError, match="MAJOR"):
        hub.upload_version(api, "org/m", version, "1.0.0", "card")
    hub.upload_version(api, "org/m", version, "v1.0.0", "card")

    added = {
        op.path_in_repo: op for op in api.operations if hasattr(op, "path_or_fileobj")
    }
    deleted = {op.path_in_repo for op in api.operations} - set(added)
    assert set(added) == {p.name for p in version.files} | {
        "README.md",
        "version.json",
        "run.json",
        "predictions/val_seed_0.csv",
        "baselines/rf_bioactivity_default/run.json",
        "baselines/rf_bioactivity_default/predictions/val_seed_0.csv",
    }
    assert deleted == {"m_seed_3.pt"}
    assert api.tags == ["v1.0.0"]

    (ckpt_dir / "version.json").write_bytes(added["version.json"].path_or_fileobj)
    assert hub.load_tagged_version(ckpt_dir, "v1.0.0", "m", SEEDS) == version
    with pytest.raises(ValueError, match="not v1.1.0"):
        hub.load_tagged_version(ckpt_dir, "v1.1.0", "m", SEEDS)

    dl_snapshot = tmp_path / "dl_snapshot"
    dl_snapshot.mkdir()
    write_run(dl_snapshot)
    (dl_snapshot / "version.json").write_bytes(added["version.json"].path_or_fileobj)
    write_run_folder(dl_snapshot, model="m", score=6.0)
    (dl_snapshot / "run.json").write_text((run_dir / "run.json").read_text())
    write_run_folder(
        dl_snapshot / "baselines" / "rf_bioactivity_default",
        model="rf_bioactivity",
        score=5.5,
    )
    unpacked_runs = tmp_path / "unpacked_runs"
    hub.populate_run_from_download(
        dl_snapshot, "v1.0.0", runs_dir=unpacked_runs, model_name="m", seeds=SEEDS
    )
    assert (
        unpacked_runs / "rf_bioactivity_default" / "predictions" / "val_seed_0.csv"
    ).exists()


def test_populate_run_from_legacy_download_synthesizes_run_json(tmp_path):
    dl_dir = tmp_path / "dl_v1.0.0"
    dl_dir.mkdir()
    write_run(dl_dir, legacy=True)
    (dl_dir / "version.json").write_text(
        json.dumps({"tag": "v1.0.0", "git_commit": COMMIT[:7]}) + "\n"
    )
    res_path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "model": "m",
                "encoder": "e",
                "split_id": "default",
                "run_id": "m_v1.0.0_default",
                "n_seeds": len(SEEDS),
                "params": "{}",
                "data_sha256": DATA,
                "split_sha256": "s" * 64,
                "base_commit": COMMIT,
                "ai_tree": TREE,
                "patch_sha256": "",
                "date": "2026-10-30",
            }
        ]
    ).to_csv(runs_path(res_path), index=False)

    out_dir = hub.populate_run_from_download(
        dl_dir,
        "v1.0.0",
        runs_dir=tmp_path / "runs",
        results_path=res_path,
        model_name="m",
        seeds=SEEDS,
    )
    assert out_dir == tmp_path / "runs" / "m_v1.0.0_default"
    meta = json.loads((out_dir / "run.json").read_text())
    assert meta["base_commit"] == COMMIT
    assert meta["ai_tree"] == TREE
    assert (out_dir / "checkpoints" / "m_ensemble.json").exists()
