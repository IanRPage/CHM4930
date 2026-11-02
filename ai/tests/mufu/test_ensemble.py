import numpy as np
import pytest
import torch
from conftest import SmallSmilesEncoder

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu import ensemble
from mufu.calibration import CalibrationParams
from mufu.contract import N_BIO, N_TOX, PIC50_COLUMNS, TOX_COLUMNS
from mufu.experiment import make_fit_predict
from mufu.mc_dropout import empirical_coverage
from mufu.model import MuFu
from mufu.train import TrainConfig, predict_logits

CFG = TrainConfig(
    batch_size=4, epochs=1, patience=1, gnn_hidden_dims=(32, 16), device="cpu"
)
COVERAGES = (0.5, 0.9)


@pytest.fixture
def checkpoints(labeled_dataset, tmp_path):
    data_path = tmp_path / "combined.csv"
    data_path.write_text("smiles\nCCO\n")
    encoder = SmallSmilesEncoder().requires_grad_(False)
    datasets = {split: labeled_dataset for split in ("train", "val", "test")}
    fit_predict = make_fit_predict(
        datasets,
        encoder,
        CFG,
        run_dir=tmp_path,
        model_name="mufu_test",
        data_path=data_path,
    )
    for seed in range(3):
        torch.manual_seed(seed)
        fit_predict(None, seed)
    return tmp_path / "checkpoints", encoder


@pytest.fixture
def fitted(checkpoints, labeled_dataset):
    ckpt_dir, encoder = checkpoints
    paths = ensemble.member_paths(ckpt_dir, "mufu_test", range(3))
    members = ensemble.load_members(paths, encoder)
    params = ensemble.fit_ensemble_conformal(
        members, labeled_dataset, n_samples=4, coverages=COVERAGES, batch_size=4
    )
    ensemble.save_ensemble(params, ckpt_dir, "mufu_test")
    return ckpt_dir, paths, members, params


def test_combined_sigma_matches_hand_calculation_and_ddof_one():
    mu = np.array([1.0, 2.0, 4.0]).reshape(3, 1, 1)
    sigma = np.array([0.5, 1.0, 1.5]).reshape(3, 1, 1)

    # mean of variances (0.25 + 1 + 2.25) / 3, sample variance of mu 7 / 3
    expected = np.sqrt((0.25 + 1 + 2.25) / 3 + 7 / 3)
    assert ensemble.combined_sigma(mu, sigma)[0, 0] == pytest.approx(expected)

    same = np.full((3, 2, N_BIO), 5.0)
    rms = np.sqrt((sigma**2).mean())
    assert ensemble.combined_sigma(same, np.broadcast_to(sigma, same.shape))[
        0, 0
    ] == pytest.approx(rms)


def test_probabilities_are_averaged_after_each_members_calibration(
    model, labeled_dataset
):
    doubled = CalibrationParams(
        active_scale=torch.full((N_BIO,), 2.0),
        active_shift=torch.zeros(N_BIO),
        tox_scale=torch.full((N_TOX,), 2.0),
        tox_shift=torch.zeros(N_TOX),
    )
    members = [
        ensemble.Member("a.pt", model, None, CalibrationParams.identity()),
        ensemble.Member("b.pt", model, None, doubled),
    ]

    mean, sigma = ensemble.ensemble_predict(members, labeled_dataset, n_samples=3)

    z = predict_logits(model, labeled_dataset)["tox_logits"][:, 0].detach()
    expected = (torch.sigmoid(z) + torch.sigmoid(2 * z)) / 2
    np.testing.assert_allclose(mean[TOX_COLUMNS[0]], expected.numpy(), rtol=1e-5)
    assert not np.allclose(mean[TOX_COLUMNS[0]], torch.sigmoid(1.5 * z).numpy())
    assert sigma.shape == (len(labeled_dataset), N_BIO)


def test_members_on_different_compounds_raise():
    a = {"smiles": ["CCO", "CCN"], "x": [1.0, 2.0]}
    b = {"smiles": ["CCN", "CCO"], "x": [1.0, 2.0]}

    with pytest.raises(ValueError, match="different compounds"):
        ensemble.average_predictions([a, b])


def test_saved_ensemble_round_trips_and_predicts_intervals(fitted, labeled_dataset):
    ckpt_dir, paths, members, params = fitted

    loaded = ensemble.load_ensemble_params(paths, ckpt_dir, "mufu_test")
    preds = ensemble.predict_with_ensemble_intervals(
        members, labeled_dataset, loaded, batch_size=4
    )

    assert loaded.conformal.q.shape == (N_BIO, len(COVERAGES))
    assert loaded.conformal.n_samples == 4
    assert [name for name, _ in loaded.members] == [p.name for p in paths]
    assert len({sha for _, sha in loaded.members}) == 3
    np.testing.assert_allclose(loaded.conformal.q, params.conformal.q)
    assert preds["smiles"] == labeled_dataset.smiles
    assert TOX_COLUMNS[0] in preds
    coverage = empirical_coverage(preds, labeled_dataset, COVERAGES)
    assert set(coverage["target"]) == set(PIC50_COLUMNS)


def test_stale_sidecar_raises(fitted):
    ckpt_dir, paths, _, _ = fitted

    with pytest.raises(ValueError, match="different members"):
        ensemble.load_ensemble_params(paths[:-1], ckpt_dir, "mufu_test")

    paths[0].write_bytes(paths[0].read_bytes() + b"x")
    with pytest.raises(ValueError, match="different members"):
        ensemble.load_ensemble_params(paths, ckpt_dir, "mufu_test")


def test_dropout_rate_mismatch_raises(fitted, labeled_dataset):
    _, _, members, params = fitted
    other = ensemble.Member(
        members[0].path,
        MuFu(
            SmallSmilesEncoder(),
            FingerprintEncoder(hidden_dims=(16,), dropout=0.5),
            GNNEncoder(hidden_dims=(12,), dropout=0.5),
        ),
        None,
        None,
    )

    with pytest.raises(ValueError, match="different dropout rates"):
        ensemble.fit_ensemble_conformal([*members, other], labeled_dataset)
    with pytest.raises(ValueError, match="differ from those q"):
        ensemble.predict_with_ensemble_intervals([other] * 2, labeled_dataset, params)


def test_member_paths_names_missing_seeds(tmp_path):
    (tmp_path / "m_seed_0.pt").touch()

    with pytest.raises(FileNotFoundError, match="m_seed_1.pt, m_seed_2.pt"):
        ensemble.member_paths(tmp_path, "m", range(3))


def test_load_members_rejects_mixed_runs(checkpoints, tmp_path):
    ckpt_dir, encoder = checkpoints
    paths = ensemble.member_paths(ckpt_dir, "mufu_test", range(3))

    with pytest.raises(ValueError, match="share a seed"):
        ensemble.load_members([paths[0], paths[0]], encoder)

    other = torch.load(paths[1], weights_only=True)
    other["params"] = {**other["params"], "lr": 123.0}
    torch.save(other, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="different params"):
        ensemble.load_members([paths[0], tmp_path / "other.pt"], encoder)
