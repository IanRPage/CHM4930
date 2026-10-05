"""
Interface contract for the MuFu model components.

Pytorch Dataset, projection, modality dropout, fusion, heads, and loss components import
their names, shapes, and defaults from here so a mismatches fail.

Batch (Dataset output), where B is the batch size:

    smiles        list[str]       len B
    fingerprint   float tensor    (B, FINGERPRINT_BITS)
    graph         PyG Batch       B graphs
    y_pic50       float tensor    (B, N_BIO)
    y_active      float tensor    (B, N_BIO)
    y_tox         float tensor    (B, N_TOX)
    mask_bio      bool tensor     (B, N_BIO)
    mask_tox      bool tensor     (B, N_TOX)

Model outputs: `pic50` (B, N_BIO), `active_logits` (B, N_BIO), `tox_logits` (B, N_TOX).
"""

from dataclasses import dataclass
from typing import Protocol, TypedDict

import torch
from torch_geometric.data import Batch

from featurize.graph import EDGE_FEATURE_DIM, NODE_FEATURE_DIM
from pipeline.combined_loader import TARGETS, TOX_TASKS

FINGERPRINT_BITS = 2048
BIO_TARGETS = tuple(TARGETS)
PIC50_COLUMNS = tuple(f"pIC50_{t}" for t in BIO_TARGETS)
ACTIVE_COLUMNS = tuple(f"active_{t}" for t in BIO_TARGETS)
TOX_COLUMNS = tuple(TOX_TASKS)
N_BIO = len(BIO_TARGETS)
N_TOX = len(TOX_COLUMNS)
LABEL_FILL = 0.0

MODALITIES = ("smiles", "fingerprint", "graph")
D_MODEL = 256
FUSED_DIM = len(MODALITIES) * D_MODEL
MODALITY_DROPOUT_P = 0.3
DROPOUT_MASKS = (
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
)

TASKS = ("pic50", "bio_cls", "tox")
LOG_VAR_INIT = 0.0

OUTPUT_WIDTHS = {"pic50": N_BIO, "active_logits": N_BIO, "tox_logits": N_TOX}


class MuFuBatch(TypedDict):
    smiles: list[str]
    fingerprint: torch.Tensor
    graph: Batch
    y_pic50: torch.Tensor
    y_active: torch.Tensor
    y_tox: torch.Tensor
    mask_bio: torch.Tensor
    mask_tox: torch.Tensor


class MuFuOutputs(TypedDict):
    pic50: torch.Tensor
    active_logits: torch.Tensor
    tox_logits: torch.Tensor


class Encoder(Protocol):
    out_dim: int

    def __call__(self, x) -> torch.Tensor: ...


@dataclass(frozen=True)
class TaskSpec:
    output: str
    label: str
    mask: str
    loss_scale: float


TASK_SPECS = {
    "pic50": TaskSpec("pic50", "y_pic50", "mask_bio", 0.5),
    "bio_cls": TaskSpec("active_logits", "y_active", "mask_bio", 1.0),
    "tox": TaskSpec("tox_logits", "y_tox", "mask_tox", 1.0),
}

TaskLosses = dict[str, torch.Tensor | None]


def _check_keys(kind: str, got: set[str], expected: set[str]) -> None:
    if got != expected:
        raise ValueError(
            f"{kind}: missing keys {sorted(expected - got)}, "
            f"unexpected keys {sorted(got - expected)}"
        )


def _check_tensor(name: str, value, shape: tuple[int, ...], dtype: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name}: expected a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(value.shape)}")
    if dtype == "bool" and value.dtype != torch.bool:
        raise ValueError(f"{name}: expected dtype bool, got {value.dtype}")
    if dtype == "float" and not value.is_floating_point():
        raise ValueError(f"{name}: expected a float dtype, got {value.dtype}")


def validate_batch(batch: MuFuBatch) -> None:
    _check_keys("batch", set(batch), set(MuFuBatch.__annotations__))

    size = len(batch["smiles"])
    _check_tensor(
        "fingerprint", batch["fingerprint"], (size, FINGERPRINT_BITS), "float"
    )

    graph = batch["graph"]
    if graph.num_graphs != size:
        raise ValueError(f"graph: expected {size} graphs, got {graph.num_graphs}")
    if graph.x.shape[1] != NODE_FEATURE_DIM:
        raise ValueError(
            f"graph.x: expected width {NODE_FEATURE_DIM}, got {graph.x.shape[1]}"
        )
    if graph.edge_attr.shape[1] != EDGE_FEATURE_DIM:
        raise ValueError(
            f"graph.edge_attr: expected width {EDGE_FEATURE_DIM}, "
            f"got {graph.edge_attr.shape[1]}"
        )

    for key, width in (("y_pic50", N_BIO), ("y_active", N_BIO), ("y_tox", N_TOX)):
        _check_tensor(key, batch[key], (size, width), "float")
        if batch[key].isnan().any():
            raise ValueError(
                f"{key}: contains NaN, fill missing labels with {LABEL_FILL}"
            )
    _check_tensor("mask_bio", batch["mask_bio"], (size, N_BIO), "bool")
    _check_tensor("mask_tox", batch["mask_tox"], (size, N_TOX), "bool")


def validate_outputs(outputs: MuFuOutputs, batch_size: int) -> None:
    _check_keys("outputs", set(outputs), set(MuFuOutputs.__annotations__))
    for key, width in OUTPUT_WIDTHS.items():
        _check_tensor(key, outputs[key], (batch_size, width), "float")
