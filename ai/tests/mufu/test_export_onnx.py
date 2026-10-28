import copy
import json

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from mufu.calibration import CalibrationParams
from mufu.contract import N_BIO, N_TOX
from mufu.data.dataset import collate
from mufu.export_onnx import (
    export_checkpoint,
    load_mc_session,
    mc_predict_onnx,
)
from mufu.mc_dropout import (
    ConformalParams,
    StochasticMuFu,
    save_conformal,
)
from mufu.train import cpu_state, predict


def _feeds(model, dataset, rows):
    batch = collate([dataset[i] for i in rows])
    with torch.no_grad():
        embedding = model.smiles_encoder(batch["smiles"])
    g = batch["graph"]
    return {
        "smiles_embedding": embedding.numpy(),
        "fingerprint": batch["fingerprint"].numpy(),
        "x": g.x.numpy(),
        "edge_index": g.edge_index.numpy(),
        "edge_attr": g.edge_attr.numpy(),
        "batch": g.batch.numpy(),
    }


@pytest.fixture
def exported(model, tmp_path):
    model.eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        tox_state = {
            k: v + 0.05 * torch.randn_like(v) if v.is_floating_point() else v
            for k, v in cpu_state(model).items()
        }
        calibration = CalibrationParams(
            torch.rand(N_BIO) + 0.5,
            torch.randn(N_BIO),
            torch.rand(N_TOX) + 0.5,
            torch.randn(N_TOX),
        )
    ckpt_path = tmp_path / "mufu_seed_0.pt"
    torch.save({"model": cpu_state(model), "seed": 0}, ckpt_path)
    save_conformal(
        ConformalParams(
            (0.9,),
            np.ones((N_BIO, 1)),
            np.full(N_BIO, 0.01),
            50,
            StochasticMuFu(model).dropout_rates,
        ),
        ckpt_path,
    )
    extras = {
        "family_states": {"tox": tox_state},
        "calibration": calibration,
        "checkpoint": {"seed": 0},
    }
    paths = export_checkpoint(
        model, extras, ckpt_path, tmp_path / "onnx", export_smiles=False
    )
    return paths, extras


@pytest.mark.parametrize("rows", [range(8), range(5, 6), range(2, 5)])
def test_deterministic_graph_matches_predict(model, dataset, exported, rows):
    # rows 5 is methane: one atom, no edges
    paths, extras = exported
    session = ort.InferenceSession(paths.mufu, providers=["CPUExecutionProvider"])
    got = np.concatenate(session.run(None, _feeds(model, dataset, rows)), axis=1)

    ref = predict(
        model,
        dataset,
        family_states=extras["family_states"],
        calibration=extras["calibration"],
    )
    want = np.stack([v for k, v in ref.items() if k != "smiles"], axis=1)[list(rows)]
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_tox_output_uses_family_state(model, dataset, exported):
    paths, extras = exported
    session = ort.InferenceSession(paths.mufu, providers=["CPUExecutionProvider"])
    tox = session.run(["tox_prob"], _feeds(model, dataset, range(8)))[0]
    no_family = predict(model, dataset, calibration=extras["calibration"])
    bio_tox = np.stack([v for k, v in no_family.items() if k != "smiles"], axis=1)
    assert not np.allclose(tox, bio_tox[:, 2 * N_BIO :], atol=1e-5)


def test_mc_graph_samples_vary(model, dataset, exported):
    paths, _ = exported
    feeds = _feeds(model, dataset, range(8))
    sigma = mc_predict_onnx(load_mc_session(paths.mc), feeds, n_samples=20)
    assert sigma.shape == (8, N_BIO)
    assert (sigma > 0).all()


def test_mc_predict_onnx_rejects_collapsed_samples():
    class Constant:
        def run(self, _names, _feeds):
            return [np.ones((3, N_BIO), dtype=np.float32)]

    with pytest.raises(RuntimeError, match="identical"):
        mc_predict_onnx(Constant(), {}, n_samples=5)


def test_mc_graph_spread_matches_pytorch(model, dataset, exported):
    paths, _ = exported
    rows = range(8)
    feeds = _feeds(model, dataset, rows)
    sigma_onnx = mc_predict_onnx(load_mc_session(paths.mc), feeds, n_samples=1000)

    batch = collate([dataset[i] for i in rows])
    stochastic = StochasticMuFu(copy.deepcopy(model))
    emb = torch.from_numpy(feeds["smiles_embedding"])
    with torch.no_grad():
        samples = torch.stack(
            [stochastic(emb, batch["fingerprint"], batch["graph"]) for _ in range(1000)]
        )
    np.testing.assert_allclose(sigma_onnx, samples.std(dim=0).numpy(), rtol=0.15)


def test_manifest(exported):
    paths, _ = exported
    manifest = json.loads(paths.manifest.read_text())
    assert manifest["tox_from_family_state"] is True
    assert manifest["calibrated"] is True
    assert manifest["conformal"]["coverages"] == [0.9]
    assert manifest["mc_ort_disabled_optimizers"] == ["EliminateDropout"]
    assert manifest["files"]["mufu"] == paths.mufu.name
