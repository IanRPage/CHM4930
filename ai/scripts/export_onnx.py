"""
Export saved MuFu checkpoints to ONNX.

From ai/: python scripts/export_onnx.py

For each seed writes, under outputs/<model_name>/onnx/:
    <model_name>_seed_<seed>.onnx         pic50, active_prob, tox_prob
    <model_name>_seed_<seed>_mc.onnx      one MC-Dropout pIC50 sample per run
    <model_name>_seed_<seed>_onnx.json    columns, input shapes, conformal q, hashes
and once, since MolFormer is frozen and shared by every seed:
    molformer_smiles.onnx + molformer_smiles_tokenizer/

Run scripts/fit_mc_dropout.py first if the conformal q should go in the manifest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import onnxruntime as ort
import torch

from baseline.benchmark import SEEDS, load_benchmark_split
from encoders.smiles_encoder import SmilesEncoder
from mufu.data.dataset import MuFuDataset, collate
from mufu.experiment import load_mufu_checkpoint
from mufu.export_onnx import (
    export_checkpoint,
    export_smiles_encoder,
    load_mc_session,
    mc_predict_onnx,
)
from mufu.train import predict

AI_DIR = Path(__file__).resolve().parents[1]

model_name = "mufu"
seeds = SEEDS
export_molformer = True
verify_rows = 64  # test rows checked against PyTorch per seed, 0 to skip
atol = 1e-4


def onnx_feeds(model, dataset: MuFuDataset) -> dict[str, np.ndarray]:
    batch = collate([dataset[i] for i in range(len(dataset))])
    with torch.no_grad():
        embedding = model.smiles_encoder(batch["smiles"]).cpu()
    graph = batch["graph"]
    return {
        "smiles_embedding": embedding.numpy(),
        "fingerprint": batch["fingerprint"].numpy(),
        "x": graph.x.numpy(),
        "edge_index": graph.edge_index.numpy(),
        "edge_attr": graph.edge_attr.numpy(),
        "batch": graph.batch.numpy(),
    }


def verify(model, extras, paths, dataset: MuFuDataset) -> None:
    """
    ONNX Runtime vs train.predict on the same rows, and check MC samples vary.
    """
    feeds = onnx_feeds(model, dataset)
    session = ort.InferenceSession(paths.mufu, providers=["CPUExecutionProvider"])
    got = np.concatenate(session.run(None, feeds), axis=1)

    ref = predict(
        model,
        dataset,
        batch_size=len(dataset),
        family_states=extras["family_states"],
        calibration=extras["calibration"],
    )
    want = np.stack([v for k, v in ref.items() if k != "smiles"], axis=1)
    err = float(np.abs(got - want).max())
    if err > atol:
        raise AssertionError(f"{paths.mufu.name}: max abs error {err:.2e} > {atol}")

    sigma = mc_predict_onnx(load_mc_session(paths.mc), feeds, n_samples=50)
    print(f"  parity max abs err {err:.2e} | mean MC sigma {sigma.mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Set options at the top of this file, not as arguments.")

    smiles_encoder = SmilesEncoder(freeze=True)
    out_dir = AI_DIR / "outputs" / model_name / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    if export_molformer:
        path = export_smiles_encoder(smiles_encoder, out_dir / "molformer_smiles.onnx")
        print(f"wrote {path.relative_to(AI_DIR)}", flush=True)

    check = None
    if verify_rows:
        df = load_benchmark_split()
        check = MuFuDataset(df[df["split"] == "test"].head(verify_rows))

    for seed in seeds:
        ckpt_path = AI_DIR / "checkpoints" / f"{model_name}_seed_{seed}.pt"
        model, extras = load_mufu_checkpoint(ckpt_path, smiles_encoder, device="cpu")
        paths = export_checkpoint(
            model, extras, ckpt_path, out_dir, export_smiles=False
        )
        print(f"seed {seed}: wrote {paths.mufu.name}, {paths.mc.name}", flush=True)
        if check is not None:
            verify(model, extras, paths, check)
