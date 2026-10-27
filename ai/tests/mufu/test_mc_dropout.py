import numpy as np
import pytest
import torch
from conftest import SmallSmilesEncoder

from encoders.fingerprint_encoder import FingerprintEncoder
from encoders.gnn_encoder import GNNEncoder
from mufu.contract import N_BIO, PIC50_COLUMNS
from mufu.data.dataset import collate
from mufu.mc_dropout import (
    StochasticMuFu,
    conformal_quantile,
    empirical_coverage,
    fit_conformal,
    fit_mc_dropout,
    load_conformal,
    mc_predict,
    predict_with_intervals,
    sidecar_path,
)
from mufu.model import MuFu
from mufu.model.fusion import LateFusion


class CountingSmilesEncoder(SmallSmilesEncoder):
    calls = 0

    def forward(self, smiles):
        type(self).calls += 1
        return super().forward(smiles)


@pytest.fixture
def mc_model():
    CountingSmilesEncoder.calls = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return MuFu(
            smiles_encoder=CountingSmilesEncoder(),
            fingerprint_encoder=FingerprintEncoder(hidden_dims=(16,), dropout=0.3),
            graph_encoder=GNNEncoder(hidden_dims=(12,), dropout=0.3),
        ).eval()


def residual_data(n, sigma=1.0):
    pred = np.zeros((n, N_BIO))
    y = np.tile(np.arange(1.0, n + 1)[:, None], (1, N_BIO))
    return pred, np.full((n, N_BIO), sigma), y, np.ones((n, N_BIO), dtype=bool)


def fit(pred, sigma, y, mask, coverages=(0.9, 0.95)):
    return fit_conformal(
        pred, sigma, y, mask, n_samples=5, dropout_rates=(0.1,), coverages=coverages
    )


def test_stochastic_passes_differ_and_copy_leaves_source_untouched(mc_model, dataset):
    batch = collate([dataset[i] for i in range(len(dataset))])
    stochastic = StochasticMuFu(mc_model)
    stochastic.train()
    embedding = mc_model.smiles_encoder(batch["smiles"])
    args = (embedding, batch["fingerprint"], batch["graph"])

    with torch.no_grad():
        first, second = stochastic(*args), stochastic(*args)

    assert not stochastic.training
    assert not torch.equal(first, second)
    assert stochastic.dropout_rates == (0.3, 0.3, 0.1)
    assert any(isinstance(m, torch.nn.Dropout) for m in mc_model.modules())


def test_modality_dropout_stays_off(mc_model, labeled_dataset, monkeypatch):
    masks = []
    fusion_forward = LateFusion.forward
    monkeypatch.setattr(
        LateFusion,
        "forward",
        lambda self, projected, mask=None: (
            masks.append(mask) or fusion_forward(self, projected, mask)
        ),
    )

    mc_predict(mc_model, labeled_dataset, n_samples=3)

    assert masks and all(mask is None for mask in masks)


def test_smiles_encoder_runs_once_per_batch(mc_model, labeled_dataset):
    mc_predict(mc_model, labeled_dataset, n_samples=6, batch_size=4)

    assert CountingSmilesEncoder.calls == 2


def test_mc_predict_shapes_and_sigma(mc_model, labeled_dataset):
    pred, sigma = mc_predict(mc_model, labeled_dataset, n_samples=8, batch_size=3)

    assert pred.shape == sigma.shape == (len(labeled_dataset), N_BIO)
    assert (sigma > 0).all()
    with pytest.raises(ValueError, match="at least 2"):
        mc_predict(mc_model, labeled_dataset, n_samples=1)


def test_conformal_quantile_picks_the_ceil_order_statistic():
    scores = np.arange(1.0, 20.0)

    assert conformal_quantile(scores, 0.9) == 18.0
    assert conformal_quantile(scores, 0.95) == 19.0
    assert conformal_quantile(scores[:9], 0.95) == float("inf")


def test_fit_conformal_scores_use_sigma_and_mask():
    pred, sigma, y, mask = residual_data(19, sigma=2.0)
    mask[0] = False
    y[0] = 1e6

    params = fit(pred, sigma, y, mask, coverages=(0.5,))

    # scores are 2..19 over 18 rows, rank ceil(19 * 0.5) = 10 -> 11 / 2
    assert params.q.shape == (N_BIO, 1)
    assert np.allclose(params.q, 5.5)


def test_fit_conformal_floors_sigma():
    pred, sigma, y, mask = residual_data(19)
    sigma[0] = 1e-12

    params = fit(pred, sigma, y, mask)

    assert (params.sigma_floor > 1e-6).all()
    assert np.isfinite(params.q).all() and (params.q < 100).all()
    assert (fit(pred, np.zeros_like(sigma), y, mask).sigma_floor == 1e-6).all()


def test_fit_conformal_needs_labels():
    pred, sigma, y, mask = residual_data(5)
    mask[:, 0] = False

    with pytest.raises(ValueError, match="no labeled rows"):
        fit(pred, sigma, y, mask)


def test_conformal_coverage_is_near_nominal_on_gaussian_residuals():
    rng = np.random.default_rng(0)

    def draw(n):
        sigma = rng.uniform(0.5, 2.0, (n, N_BIO))
        pred = rng.normal(size=(n, N_BIO))
        return pred, sigma, pred + sigma * rng.normal(size=(n, N_BIO))

    pred, sigma, y = draw(2000)
    params = fit(pred, sigma, y, np.ones_like(y, dtype=bool), coverages=(0.9, 0.95))
    pred, sigma, y = draw(20000)

    for cov, (lo, hi) in params.intervals(pred, sigma).items():
        inside = ((y >= lo) & (y <= hi)).mean(axis=0)
        assert np.allclose(inside, cov, atol=0.02)


def test_fit_save_load_and_predict_with_intervals(mc_model, labeled_dataset, tmp_path):
    ckpt = tmp_path / "mufu_seed_0.pt"
    torch.save(mc_model.state_dict(), ckpt)

    params = fit_mc_dropout(mc_model, labeled_dataset, ckpt, n_samples=6)
    loaded = load_conformal(ckpt)
    assert sidecar_path(ckpt) == tmp_path / "mufu_seed_0_mc_dropout.json"
    preds = predict_with_intervals(mc_model, labeled_dataset, loaded, batch_size=3)

    assert loaded.n_samples == 6 and loaded.coverages == params.coverages
    assert np.array_equal(loaded.q, params.q)
    assert np.array_equal(loaded.sigma_floor, params.sigma_floor)
    assert loaded.dropout_rates == (0.3, 0.3, 0.1)
    assert preds["smiles"] == labeled_dataset.smiles
    for col in PIC50_COLUMNS:
        assert preds[col].shape == (len(labeled_dataset),)
        assert (preds[f"{col}_lo90"] <= preds[f"{col}_lo95"]).all()
        assert (preds[f"{col}_lo95"] <= preds[col]).all()
        assert (preds[col] <= preds[f"{col}_hi95"]).all()
        assert (preds[f"{col}_hi95"] >= preds[f"{col}_hi90"]).all()
    coverage = empirical_coverage(preds, labeled_dataset, loaded.coverages)
    assert list(coverage["target"]) == [c for c in PIC50_COLUMNS for _ in range(2)]
    assert list(coverage["n"]) == [4, 4, 3, 3]


def test_stale_q_is_rejected(mc_model, labeled_dataset, tmp_path):
    ckpt = tmp_path / "mufu_seed_0.pt"
    torch.save(mc_model.state_dict(), ckpt)
    params = fit_mc_dropout(mc_model, labeled_dataset, ckpt, n_samples=4)

    other = MuFu(
        CountingSmilesEncoder(),
        FingerprintEncoder(hidden_dims=(16,), dropout=0.5),
        GNNEncoder(hidden_dims=(12,), dropout=0.3),
    ).eval()
    with pytest.raises(ValueError, match="dropout rates"):
        predict_with_intervals(other, labeled_dataset, params)
    ckpt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="different checkpoint"):
        load_conformal(ckpt)
