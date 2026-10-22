"""Inference-only MuFu adapter for the frozen shared benchmark.

SCRUM-60 supplies a constructed model and training provenance, or a checkpoint
containing ``model_state_dict`` and ``training``. No training, model construction,
downloads, or result writes happen implicitly. Unattested models are restricted
to validation diagnostics; canonical export requires both validation and test.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import benchmark as bm
from baseline.results import RESULTS_PATH
from encoder_benchmark import config_json, verify_export
from mufu import contract as c
from mufu.dataset import LABELS, MASKS, MuFuDataset, collate
from mufu.model import MuFu
from pipeline.combined_loader import COMBINED_CSV_PATH
from pipeline.toxicity_loader import DATASETS

MODEL_ID = "mufu_full"
ADAPTER_VERSION = 1
OUTPUT_COLUMNS = {
    "pic50": c.PIC50_COLUMNS,
    "active_logits": c.ACTIVE_COLUMNS,
    "tox_logits": c.TOX_COLUMNS,
}


@dataclass
class MuFuEvaluation:
    rows: list[dict]
    predictions: dict[str, dict[str, pd.DataFrame]]
    benchmark: dict
    checkpoint: dict
    training_attested: bool


def _read_checkpoint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or not {"model_state_dict", "training"} <= checkpoint.keys()
    ):
        raise ValueError(
            "checkpoint must contain model_state_dict and training provenance"
        )
    return checkpoint, {"source": str(path), "file_sha256": digest.hexdigest()}


def _training_provenance(training, full, data_path):
    if not isinstance(training, dict) or training.get("trained") is not True:
        raise ValueError("training provenance must explicitly attest trained=True")
    steps = training.get("training_steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("training_steps must be a positive integer")
    seed = training.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != bm.TRAINING_SEED:
        raise ValueError("canonical MuFu training_seed must be 42")
    if training.get("selection_split") != "val":
        raise ValueError("checkpoint selection_split must be val, never test")
    if not isinstance(training.get("benchmark"), dict):
        raise TypeError("training provenance must contain full benchmark metadata")
    bm.verify_benchmark_metadata(training["benchmark"], full, data_path)
    # Snapshot caller metadata and require portable primitive checkpoint metadata.
    return json.loads(json.dumps(training, allow_nan=False))


def _validate_state(state, expected):
    if not isinstance(state, Mapping) or not state or set(state) != set(expected):
        raise ValueError(
            "checkpoint state keys must match the constructed MuFu model exactly"
        )
    for name, value in state.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f"checkpoint state {name} must be a dense tensor")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"checkpoint state shape/dtype mismatch: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"checkpoint state contains nonfinite values: {name}")


def _state_sha256(model):
    state = model.state_dict()
    _validate_state(state, state)
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        description = config_json(
            [name, str(tensor.dtype), list(tensor.shape)]
        ).encode()
        digest.update(len(description).to_bytes(8, "big"))
        digest.update(description)
        # Byte views also support bfloat16 and scalar BatchNorm counters.
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()


def _check_batch(batch, expected):
    c.validate_batch(batch)
    if batch["smiles"] != expected["smiles"].tolist():
        raise ValueError("evaluation SMILES coverage/order mismatch")
    for key, columns in MASKS.items():
        mask = torch.tensor(
            expected[list(columns)].notna().to_numpy(), dtype=torch.bool
        )
        if not torch.equal(batch[key].cpu(), mask):
            raise ValueError(f"evaluation label mask mismatch: {key}")
    for key, columns in LABELS.items():
        labels = torch.tensor(
            expected[list(columns)].fillna(c.LABEL_FILL).to_numpy(), dtype=torch.float
        )
        if not torch.equal(batch[key].cpu(), labels):
            raise ValueError(f"evaluation labels are misaligned: {key}")


def _predict_split(model, cohort, batch_size, device):
    dataset = MuFuDataset(cohort)
    if len(dataset) != len(cohort) or dataset.smiles != cohort["smiles"].tolist():
        raise ValueError("featurization changed the evaluation cohort or SMILES order")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(bm.TRAINING_SEED),
    )
    collected = {key: [] for key in OUTPUT_COLUMNS}
    offset = 0
    for batch in loader:
        size = len(batch["smiles"])
        _check_batch(batch, cohort.iloc[offset : offset + size])
        inputs = {
            "smiles": batch["smiles"],
            "fingerprint": batch["fingerprint"].to(device),
            "graph": batch["graph"].to(device),
        }
        outputs = model(inputs)
        c.validate_outputs(outputs, size)
        for key, values in outputs.items():
            collected[key].append(values.detach().to(device="cpu", dtype=torch.float64))
        offset += size
    if offset != len(cohort):
        raise ValueError("incomplete evaluation cohort coverage")
    index = pd.Index(cohort["smiles"], name="smiles")
    return {
        key: pd.DataFrame(
            torch.cat(chunks).numpy(), index=index, columns=OUTPUT_COLUMNS[key]
        )
        for key, chunks in collected.items()
    }


def _metric_rows(dataset, endpoint, split, metrics, labels, params):
    observed = labels.dropna() if labels is not None else None
    positive = (
        (
            observed >= bm.ACTIVE_THRESHOLD
            if endpoint in c.PIC50_COLUMNS
            else observed == 1
        )
        if observed is not None
        else None
    )
    return [
        {
            "encoder": "+".join(c.MODALITIES),
            "dataset": dataset,
            "endpoint": endpoint,
            "split": split,
            "metric": metric,
            "value": value,
            "n_labeled": len(observed) if observed is not None else np.nan,
            "n_positive": int(positive.sum()) if positive is not None else np.nan,
            "params": config_json(params),
        }
        for metric, value in metrics.items()
    ]


def _score_split(cohort, predictions, split, params):
    labels = cohort.set_index("smiles")
    rows = []
    for target, pic50, active in zip(c.BIO_TARGETS, c.PIC50_COLUMNS, c.ACTIVE_COLUMNS):
        mask = labels[pic50].notna()
        scores = predictions["pic50"][pic50]
        regression = bm.evaluate_predictions(labels[pic50], scores, "regression", mask)
        regression.update(
            bm.evaluate_predictions(labels[active], scores, "classification", mask)
        )
        rows.extend(
            _metric_rows(
                target.lower(),
                pic50,
                split,
                regression,
                labels[pic50],
                params | {"score_source": "predicted_pic50"},
            )
        )
        classification = bm.evaluate_predictions(
            labels[active], predictions["active_logits"][active], "classification", mask
        )
        rows.extend(
            _metric_rows(
                target.lower(),
                active,
                split,
                classification,
                labels[active],
                params | {"score_source": "active_logits"},
            )
        )

    tox_labels = labels[list(c.TOX_COLUMNS)]
    tox_scores = predictions["tox_logits"]
    mask = tox_labels.notna()
    overall = bm.evaluate_predictions(tox_labels, tox_scores, "multitask", mask)
    tox_params = params | {"score_source": "tox_logits"}
    for task, value in zip(c.TOX_COLUMNS, overall["per_task_roc_auc"]):
        dataset = next(
            name for name, config in DATASETS.items() if task in config["tasks"]
        )
        rows.extend(
            _metric_rows(
                dataset, task, split, {"roc_auc": value}, labels[task], tox_params
            )
        )
    groups = {"overall": overall}
    for name, config in DATASETS.items():
        columns = config["tasks"]
        groups[name] = bm.evaluate_predictions(
            tox_labels[columns], tox_scores[columns], "multitask", mask[columns]
        )
    for name, metrics in groups.items():
        rows.extend(
            _metric_rows(
                name,
                f"macro_{name}",
                split,
                {"roc_auc": metrics["mean_roc_auc"]},
                None,
                tox_params,
            )
        )
    return rows


def evaluate_mufu(
    model: MuFu,
    *,
    checkpoint: Path | None = None,
    training: dict | None = None,
    splits=("val",),
    batch_size=64,
    data_path: Path = COMBINED_CSV_PATH,
    write_results=False,
    results_path: Path = RESULTS_PATH,
) -> MuFuEvaluation:
    """Evaluate a constructed MuFu without training or selecting model weights.

    Checkpoint loading is strict and in-place, using weights_only=True. For an
    already loaded model, pass its training provenance directly. Provenance is an
    explicit trainer attestation, not a statistical proof of learning. Model modes
    are restored; the adapter does not move the model or reseed training RNGs.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch module implementing the MuFu contract")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if (
        not isinstance(splits, (tuple, list))
        or not splits
        or len(set(splits)) != len(splits)
        or set(splits) - {"val", "test"}
    ):
        raise ValueError("splits must contain unique val/test partition names")
    splits = tuple(name for name in ("val", "test") if name in splits)
    if write_results and splits != ("val", "test"):
        raise ValueError("export requires complete val and test results in one call")
    if checkpoint is not None and training is not None:
        raise ValueError("supply checkpoint provenance or training, not both")
    full = bm.load_benchmark(data_path)
    metadata = bm.benchmark_metadata(full, data_path)
    context = SimpleNamespace(full=full, metadata=metadata, data_path=data_path)
    state = None
    identity = {"source": "in_memory", "file_sha256": None}
    if checkpoint is not None:
        packet, identity = _read_checkpoint(checkpoint)
        training = packet["training"]
        state = packet["model_state_dict"]
    provenance = (
        _training_provenance(training, full, data_path)
        if training is not None
        else None
    )
    if provenance is None and (
        checkpoint is not None or write_results or "test" in splits
    ):
        raise ValueError(
            "test evaluation/export requires attested trained, validation-selected weights"
        )
    if write_results:
        verify_export(context, results_path)
    if state is not None:
        _validate_state(state, model.state_dict())
        model.load_state_dict(state, strict=True)
    identity["state_sha256"] = _state_sha256(model)
    devices = {tensor.device for tensor in (*model.parameters(), *model.buffers())}
    if len(devices) > 1:
        raise ValueError("MuFu evaluation requires a model on one device")
    device = next(iter(devices), torch.device("cpu"))
    params = {
        "benchmark": metadata,
        "checkpoint": identity,
        "training": provenance,
        "training_attested": provenance is not None,
        "evaluation": {
            "adapter_version": ADAPTER_VERSION,
            "splits": list(splits),
            "batch_size": batch_size,
            "device": str(device),
            "shuffle": False,
            "drop_last": False,
            "gradients": False,
            "modality_dropout": False,
            "loader_seed": bm.TRAINING_SEED,
        },
    }
    rows, predictions = [], {}
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        if any(module.training for module in model.modules()):
            raise ValueError("every MuFu component must enter evaluation mode")
        with torch.inference_mode():
            for split in splits:
                cohort = full.loc[full["split"] == split]
                if cohort.empty:
                    raise ValueError(f"empty evaluation cohort: {split}")
                predictions[split] = _predict_split(model, cohort, batch_size, device)
                rows.extend(_score_split(cohort, predictions[split], split, params))
    finally:
        for module, mode in modes.items():
            module.training = mode
    if write_results:
        verify_export(context, results_path)
        bm.record_results(rows, model=MODEL_ID, path=results_path, data_path=data_path)
    return MuFuEvaluation(rows, predictions, metadata, identity, provenance is not None)
