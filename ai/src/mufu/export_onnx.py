"""
Export a trained MuFu checkpoint to ONNX.

Writes up to three graphs plus a JSON manifest next to each other:

    <stem>_smiles.onnx   MolFormer: input_ids, attention_mask -> smiles_embedding
    <stem>.onnx          deterministic MuFu: pic50, active_prob, tox_prob
    <stem>_mc.onnx       StochasticMuFu: one MC-Dropout pIC50 sample per call
    <stem>_onnx.json     column names, featurization constants, conformal q, hashes

MolFormer is exported on its own because it is frozen, shared across seeds and never
saved in MuFu checkpoints, so the two MuFu graphs take its pooled output as an input,
the same split `mc_predict` uses (MolFormer once per batch, T stochastic passes after).

ONNX has no PyG `Batch`, so the graph is passed flat: `x`, `edge_index`, `edge_attr` and
`batch` (graph id per node), exactly the tensors `Batch.from_data_list` produces.
"""

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import global_add_pool

from encoders.gnn_encoder import GNNEncoder
from featurize.fingerprint import FINGERPRINT_BITS
from featurize.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from mufu.calibration import CalibrationParams
from mufu.contract import ACTIVE_COLUMNS, PIC50_COLUMNS, TOX_COLUMNS
from mufu.mc_dropout import (
    StochasticMuFu,
    load_conformal,
    sidecar_path,
)
from mufu.model import MuFu
from mufu.train import load_non_smiles_state

OPSET = 18
MANIFEST_SUFFIX = "_onnx.json"
GRAPH_INPUTS = ("x", "edge_index", "edge_attr", "batch")
MUFU_INPUTS = ("smiles_embedding", "fingerprint", *GRAPH_INPUTS)
MUFU_OUTPUTS = ("pic50", "active_prob", "tox_prob")
MC_OUTPUTS = ("pic50_sample",)
TRUNK_PARTS = (
    "fingerprint_encoder",
    "graph_encoder",
    "projection",
    "fusion",
    "regression_head",
    "classification_head",
    "toxicity_head",
)


def encode_graph(
    encoder: GNNEncoder,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: torch.SymInt | int,
) -> torch.Tensor:
    """
    `GNNEncoder.forward` on flat tensors with an explicit graph count.

    The encoder's own `global_add_pool(x, batch)` infers the count as
    `batch.max() + 1`, which the exporter would freeze at the example batch size.
    Here it comes from the fingerprint batch instead. Keep in sync with GNNEncoder.
    """
    for conv in encoder.mp_layers:
        x = encoder.mp_dropout(conv(x, edge_index, edge_attr=edge_attr))
    return encoder.pool_norm(global_add_pool(x, batch, size=num_graphs))


def _copy_parts(src: nn.Module, dst: nn.Module, parts: tuple[str, ...]) -> None:
    for name in parts:
        setattr(dst, name, copy.deepcopy(getattr(src, name)))


class _Trunk(nn.Module):
    """
    Everything in MuFu after the SMILES encoder, eval mode, on flat graph tensors.
    """

    def __init__(self, model: MuFu) -> None:
        super().__init__()
        _copy_parts(model, self, TRUNK_PARTS)
        self.eval()

    def forward(self, smiles_embedding, fingerprint, x, edge_index, edge_attr, batch):
        n = fingerprint.shape[0]
        projected = self.projection(
            {
                "smiles": smiles_embedding,
                "fingerprint": self.fingerprint_encoder(fingerprint),
                "graph": encode_graph(
                    self.graph_encoder, x, edge_index, edge_attr, batch, n
                ),
            }
        )
        latent = self.fusion(projected, None)
        return (
            self.regression_head(latent),
            self.classification_head(latent),
            self.toxicity_head(latent),
        )


class MuFuForExport(nn.Module):
    """
    Deterministic inference graph matching `train.predict`.

    pIC50 and activity come from the bio checkpoint (`ckpt["model"]`); toxicity comes
    from `family_states["tox"]` when the run used task-family checkpointing. Post-hoc
    calibration is folded in, so the classification outputs are probabilities.
    """

    def __init__(
        self,
        model: MuFu,
        family_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        calibration: CalibrationParams | None = None,
    ) -> None:
        super().__init__()
        self.bio = _Trunk(model)
        if family_states and "tox" in family_states:
            tox_model = copy.deepcopy(
                model, memo={id(model.smiles_encoder): model.smiles_encoder}
            )
            load_non_smiles_state(tox_model, family_states["tox"])
            self.tox = _Trunk(tox_model)
        else:
            self.tox = self.bio
        cal = calibration or CalibrationParams.identity()
        for name, value in cal.state_dict().items():
            self.register_buffer(name, value.float())
        self.eval()

    def forward(self, smiles_embedding, fingerprint, x, edge_index, edge_attr, batch):
        inputs = (smiles_embedding, fingerprint, x, edge_index, edge_attr, batch)
        pic50, active_logits, tox_logits = self.bio(*inputs)
        if self.tox is not self.bio:
            tox_logits = self.tox(*inputs)[2]
        return (
            pic50,
            torch.sigmoid(active_logits * self.active_scale + self.active_shift),
            torch.sigmoid(tox_logits * self.tox_scale + self.tox_shift),
        )


class StochasticMuFuForExport(nn.Module):
    """
    `StochasticMuFu` on flat graph tensors: one dropout-on pIC50 sample per call.

    MCDropout exports as ONNX `Dropout` with training_mode=True, so ONNX Runtime draws
    a fresh mask on every run. Call it T times (or tile the batch T times) for sigma.
    """

    def __init__(self, model: MuFu) -> None:
        super().__init__()
        self.inner = StochasticMuFu(model)
        self.eval()  # MCDropout ignores this; it keeps BatchNorm/LayerNorm in eval

    @property
    def dropout_rates(self) -> tuple[float, ...]:
        return self.inner.dropout_rates

    def forward(self, smiles_embedding, fingerprint, x, edge_index, edge_attr, batch):
        m = self.inner
        n = fingerprint.shape[0]
        projected = m.projection(
            {
                "smiles": smiles_embedding,
                "fingerprint": m.fingerprint_encoder(fingerprint),
                "graph": encode_graph(
                    m.graph_encoder, x, edge_index, edge_attr, batch, n
                ),
            }
        )
        return m.regression_head(m.fusion(projected, None))


class MolFormerForExport(nn.Module):
    """
    Bare MolFormer forward: token ids and attention mask -> pooled embedding.
    Tokenization stays outside the graph (saved alongside with `save_pretrained`).
    """

    def __init__(self, smiles_encoder: nn.Module) -> None:
        super().__init__()
        self.model = smiles_encoder.model
        self.eval()

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output


def example_inputs(
    smiles: list[str], smiles_dim: int
) -> tuple[tuple[torch.Tensor, ...], list[str]]:
    """
    Real featurized molecules as tracing examples. Random embeddings are fine since
    only shapes matter; the graph needs real topology so edge counts look normal.
    """
    from rdkit import Chem
    from torch_geometric.data import Batch

    from featurize.fingerprint import mol_to_ecfp4
    from featurize.graph import mol_to_graph

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    graph = Batch.from_data_list([mol_to_graph(m) for m in mols])
    fingerprint = torch.tensor(np.stack([mol_to_ecfp4(m) for m in mols])).float()
    embedding = torch.randn(len(smiles), smiles_dim)
    return (
        embedding,
        fingerprint,
        graph.x,
        graph.edge_index,
        graph.edge_attr,
        graph.batch,
    ), smiles


# Avoid sizes 0 and 1 in tracing examples; torch.export specializes on them.
EXAMPLE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C/C=C/c1ccc(O)cc1",
    "O=C(O)C[C@H](N)C(=O)O",
]


def _dynamic_shapes() -> dict[str, dict[int, torch.export.Dim]]:
    b = torch.export.Dim("batch_size", min=1)
    n = torch.export.Dim("num_nodes", min=1)
    e = torch.export.Dim.DYNAMIC
    return {
        "smiles_embedding": {0: b},
        "fingerprint": {0: b},
        "x": {0: n},
        "edge_index": {1: e},
        "edge_attr": {0: e},
        "batch": {0: n},
    }


def _export(
    module: nn.Module,
    args: tuple[torch.Tensor, ...],
    path: Path,
    output_names: tuple[str, ...],
    metadata: Mapping[str, str],
) -> Path:
    import onnx

    program = torch.onnx.export(
        module,
        args,
        dynamo=True,
        input_names=list(MUFU_INPUTS),
        output_names=list(output_names),
        dynamic_shapes=_dynamic_shapes(),
        opset_version=OPSET,
        optimize=True,
        external_data=False,
    )
    program.save(path)
    model = onnx.load(path)
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    return path


def export_smiles_encoder(
    smiles_encoder: nn.Module, path: Path, *, opset: int = OPSET
) -> Path:
    """
    Export MolFormer and save its tokenizer to `<path stem>_tokenizer/`.

    MolFormer loads with trust_remote_code, so if this fails the fallback is to keep
    MolFormer in PyTorch and feed its embedding to the MuFu graphs.
    """
    wrapper = MolFormerForExport(smiles_encoder)
    tokens = smiles_encoder.tokenizer(
        EXAMPLE_SMILES,
        padding=True,
        truncation=True,
        max_length=smiles_encoder.max_length,
        return_tensors="pt",
    )
    args = (tokens["input_ids"], tokens["attention_mask"])
    torch.onnx.export(
        wrapper,
        args,
        path,
        dynamo=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["smiles_embedding"],
        dynamic_shapes={
            "input_ids": {
                0: torch.export.Dim("batch_size"),
                1: torch.export.Dim("seq"),
            },
            "attention_mask": {
                0: torch.export.Dim("batch_size"),
                1: torch.export.Dim("seq"),
            },
        },
        opset_version=opset,
    )
    smiles_encoder.tokenizer.save_pretrained(path.with_name(path.stem + "_tokenizer"))
    return path


@dataclass(frozen=True)
class ExportPaths:
    mufu: Path
    mc: Path
    manifest: Path
    smiles: Path | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_checkpoint(
    model: MuFu,
    extras: Mapping,
    ckpt_path: Path,
    out_dir: Path,
    *,
    smiles_encoder: nn.Module | None = None,
    export_smiles: bool = True,
) -> ExportPaths:
    """
    Export one checkpoint loaded by `experiment.load_mufu_checkpoint`.

    `extras` is that function's second return value. The conformal sidecar is copied
    into the manifest when present; q is tied to the checkpoint hash and dropout rates
    it was fitted with, so both are checked and recorded.
    """
    model = model.cpu().eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem
    smiles_encoder = smiles_encoder or model.smiles_encoder
    smiles_dim = smiles_encoder.out_dim
    args, _ = example_inputs(EXAMPLE_SMILES, smiles_dim)

    deterministic = MuFuForExport(
        model, extras.get("family_states"), extras.get("calibration")
    )
    stochastic = StochasticMuFuForExport(model)

    ckpt_sha = _sha256(ckpt_path)
    conformal = None
    if sidecar_path(ckpt_path).exists():
        conformal = load_conformal(ckpt_path)  # raises on checkpoint hash mismatch
        if conformal.dropout_rates != stochastic.dropout_rates:
            raise ValueError("dropout rates differ from those q was fitted with")

    meta = {"source_checkpoint": ckpt_path.name, "source_sha256": ckpt_sha}
    mufu_path = _export(
        deterministic, args, out_dir / f"{stem}.onnx", MUFU_OUTPUTS, meta
    )
    mc_path = _export(stochastic, args, out_dir / f"{stem}_mc.onnx", MC_OUTPUTS, meta)

    smiles_path = None
    if export_smiles:
        smiles_path = export_smiles_encoder(
            smiles_encoder, out_dir / f"{stem}_smiles.onnx"
        )

    ckpt = extras.get("checkpoint", {})
    manifest = {
        "source_checkpoint": ckpt_path.name,
        "source_sha256": ckpt_sha,
        "git_commit": ckpt.get("git_commit"),
        "data_sha256": ckpt.get("data_sha256"),
        "seed": ckpt.get("seed"),
        "opset": OPSET,
        "files": {
            "mufu": mufu_path.name,
            "mc": mc_path.name,
            "smiles": smiles_path and smiles_path.name,
            "tokenizer": smiles_path and smiles_path.stem + "_tokenizer",
        },
        "inputs": {
            "smiles_embedding": ["batch_size", smiles_dim],
            "fingerprint": ["batch_size", FINGERPRINT_BITS],
            "x": ["num_nodes", NODE_FEATURE_DIM],
            "edge_index": [2, "num_edges"],
            "edge_attr": ["num_edges", EDGE_FEATURE_DIM],
            "batch": ["num_nodes"],
        },
        "input_dtypes": {
            "smiles_embedding": "float32",
            "fingerprint": "float32",
            "x": "float32",
            "edge_index": "int64",
            "edge_attr": "float32",
            "batch": "int64",
        },
        "outputs": {
            "pic50": list(PIC50_COLUMNS),
            "active_prob": list(ACTIVE_COLUMNS),
            "tox_prob": list(TOX_COLUMNS),
            "pic50_sample": list(PIC50_COLUMNS),
        },
        "smiles_encoder": {
            "model_name": getattr(smiles_encoder, "model_name", None),
            "revision": getattr(smiles_encoder, "revision", None),
            "max_length": getattr(smiles_encoder, "max_length", None),
        },
        "tox_from_family_state": deterministic.tox is not deterministic.bio,
        "calibrated": extras.get("calibration") is not None,
        "mc_dropout_rates": list(stochastic.dropout_rates),
        "mc_ort_disabled_optimizers": list(ORT_MC_DISABLED_OPTIMIZERS),
        "conformal": conformal and json.loads(conformal.to_json()),
    }
    manifest_path = out_dir / f"{stem}{MANIFEST_SUFFIX}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ExportPaths(mufu_path, mc_path, manifest_path, smiles_path)


# ONNX Runtime's basic graph optimizations delete Dropout nodes even with
# training_mode=True, which silently turns every MC pass into the same prediction.
ORT_MC_DISABLED_OPTIMIZERS = ("EliminateDropout",)


def load_mc_session(path: Path, providers: list[str] | None = None):
    """
    ONNX Runtime session for `<stem>_mc.onnx` with dropout elimination turned off.
    """
    import onnxruntime as ort

    return ort.InferenceSession(
        str(path),
        providers=providers or ["CPUExecutionProvider"],
        disabled_optimizers=list(ORT_MC_DISABLED_OPTIMIZERS),
    )


def mc_predict_onnx(
    mc_session, feeds: Mapping[str, np.ndarray], n_samples: int
) -> np.ndarray:
    """
    Standard deviation of `n_samples` MC-Dropout passes through `<stem>_mc.onnx`,
    the ONNX Runtime counterpart of `mc_predict`'s sigma. Pair it with the
    deterministic graph's pic50 and the manifest's conformal q / sigma_floor.
    Use `load_mc_session` to open the graph.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    samples = np.stack([mc_session.run(None, dict(feeds))[0] for _ in range(n_samples)])
    sigma = samples.std(axis=0, ddof=1)  # torch.std default is unbiased
    if not sigma.any():
        raise RuntimeError(
            "MC-Dropout samples are identical; the runtime removed Dropout. "
            "Open the graph with load_mc_session."
        )
    return sigma


__all__ = [
    "MolFormerForExport",
    "MuFuForExport",
    "StochasticMuFuForExport",
    "encode_graph",
    "export_checkpoint",
    "export_smiles_encoder",
    "load_mc_session",
    "mc_predict_onnx",
]
