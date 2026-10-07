import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from encoders.fingerprint_encoder import FingerprintEncoder
from featurize.graph import mol_to_graph
from mufu import contract as c
from pipeline.combined_loader import load_combined


def make_batch(size=3):
    graphs = [
        mol_to_graph(Chem.MolFromSmiles(s)) for s in ["CCO", "c1ccccc1", "CC(=O)O"]
    ]
    return {
        "smiles": ["CCO", "c1ccccc1", "CC(=O)O"][:size],
        "fingerprint": torch.zeros(size, c.FINGERPRINT_BITS),
        "graph": Batch.from_data_list(graphs[:size]),
        "y_pic50": torch.zeros(size, c.N_BIO),
        "y_active": torch.zeros(size, c.N_BIO),
        "y_tox": torch.zeros(size, c.N_TOX),
        "mask_bio": torch.ones(size, c.N_BIO, dtype=torch.bool),
        "mask_tox": torch.ones(size, c.N_TOX, dtype=torch.bool),
    }


def make_outputs(size=3):
    return {k: torch.zeros(size, w) for k, w in c.OUTPUT_WIDTHS.items()}


def test_task_counts():
    assert c.N_TOX == 14
    assert c.N_BIO == 2


def test_columns_exist_in_combined_snapshot():
    columns = set(load_combined().columns)
    assert {*c.PIC50_COLUMNS, *c.ACTIVE_COLUMNS, *c.TOX_COLUMNS} <= columns


def test_fingerprint_bits_match_encoder():
    assert FingerprintEncoder().in_bits == c.FINGERPRINT_BITS


def test_dropout_masks():
    assert len(set(c.DROPOUT_MASKS)) == 6
    for mask in c.DROPOUT_MASKS:
        assert len(mask) == len(c.MODALITIES)
        assert 0 < sum(mask) < len(c.MODALITIES)


def test_task_specs_are_consistent():
    assert set(c.TASK_SPECS) == set(c.TASKS)
    for spec in c.TASK_SPECS.values():
        assert spec.output in c.MuFuOutputs.__annotations__
        assert spec.label in c.MuFuBatch.__annotations__
        assert spec.mask in c.MuFuBatch.__annotations__
    assert c.TASK_SPECS["tox"].mask == "mask_tox"


def test_valid_batch_passes():
    c.validate_batch(make_batch())


def test_valid_outputs_pass():
    c.validate_outputs(make_outputs(), 3)


def _drop(key):
    def mutate(b):
        del b[key]

    return mutate


def _set(key, value):
    def mutate(b):
        b[key] = value

    return mutate


def _nan_label(b):
    b["y_tox"][0, 0] = float("nan")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_drop("mask_tox"), "missing keys"),
        (_set("extra", 1), "unexpected keys"),
        (_set("fingerprint", torch.zeros(3, 1024)), "fingerprint"),
        (_set("mask_bio", torch.ones(3, 2)), "expected dtype bool"),
        (_set("y_active", torch.zeros(3, 2, dtype=torch.long)), "float dtype"),
        (_nan_label, "NaN"),
        (_set("smiles", ["CCO"]), "fingerprint"),
    ],
)
def test_validate_batch_rejects(mutate, match):
    batch = make_batch()
    mutate(batch)
    with pytest.raises(ValueError, match=match):
        c.validate_batch(batch)


def test_validate_batch_rejects_non_tensor():
    batch = make_batch()
    batch["fingerprint"] = [[0.0]]
    with pytest.raises(TypeError, match="expected a tensor"):
        c.validate_batch(batch)


def test_validate_batch_rejects_graph_count_mismatch():
    batch = make_batch()
    batch["graph"] = make_batch(2)["graph"]
    with pytest.raises(ValueError, match="graphs"):
        c.validate_batch(batch)


def test_validate_batch_rejects_graph_feature_widths():
    batch = make_batch()
    batch["graph"].x = torch.zeros(batch["graph"].x.shape[0], 3)
    with pytest.raises(ValueError, match="graph.x"):
        c.validate_batch(batch)

    batch = make_batch()
    batch["graph"].edge_attr = torch.zeros(batch["graph"].edge_attr.shape[0], 3)
    with pytest.raises(ValueError, match="graph.edge_attr"):
        c.validate_batch(batch)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_drop("tox_logits"), "missing keys"),
        (_set("tox_logits", torch.zeros(3, 12)), "tox_logits"),
    ],
)
def test_validate_outputs_rejects(mutate, match):
    outputs = make_outputs()
    mutate(outputs)
    with pytest.raises(ValueError, match=match):
        c.validate_outputs(outputs, 3)
