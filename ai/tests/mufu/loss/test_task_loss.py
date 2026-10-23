import numpy as np
import pandas as pd
import torch

from mufu import contract as c
from mufu.data.dataset import MuFuDataset
from mufu.loss.task_loss import (
    TaskLoss,
    bce_loss,
    compute_pos_weight,
    compute_task_losses,
    masked_mean,
    pic50_loss,
)


def make_batch_and_outputs(size=3):
    batch = {
        "y_pic50": torch.zeros(size, c.N_BIO),
        "y_active": torch.zeros(size, c.N_BIO),
        "y_tox": torch.zeros(size, c.N_TOX),
        "mask_bio": torch.ones(size, c.N_BIO, dtype=torch.bool),
        "mask_tox": torch.ones(size, c.N_TOX, dtype=torch.bool),
    }
    outputs = {k: torch.zeros(size, w) for k, w in c.OUTPUT_WIDTHS.items()}
    return batch, outputs


def test_masked_mean():
    losses = torch.tensor([[10.0, 999.0], [2.0, 4.0], [500.0, 600.0]])
    mask = torch.tensor([[True, False], [True, True], [False, False]])

    assert torch.isclose(masked_mean(losses, mask), torch.tensor(6.5))
    assert masked_mean(losses, torch.zeros_like(mask)) is None


def test_batch_invariance_and_nan_masking():
    pred_a = torch.tensor([[6.5, 7.0], [5.0, 8.0]], requires_grad=True)
    target_a = torch.tensor([[6.0, float("nan")], [5.5, 7.5]])
    mask_a = torch.tensor([[True, False], [True, True]])
    loss_a = pic50_loss(pred_a, target_a, mask_a)
    loss_a.backward()

    pred_b = torch.tensor([[6.5, 7.0], [5.0, 8.0], [12.0, -4.0]], requires_grad=True)
    target_b = torch.tensor([[6.0, float("nan")], [5.5, 7.5], [float("nan"), 0.0]])
    mask_b = torch.tensor([[True, False], [True, True], [False, False]])
    loss_b = pic50_loss(pred_b, target_b, mask_b)
    loss_b.backward()

    assert torch.isclose(loss_a, loss_b)
    assert torch.allclose(pred_a.grad, pred_b.grad[:2])
    assert pred_b.grad[0, 1].item() == 0.0
    assert torch.all(pred_b.grad[2] == 0.0)


def test_pos_weight_upweights_inactives():
    labels = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [float("nan"), 0.0]]
    )
    mask = torch.tensor(
        [[True, True], [True, True], [True, False], [True, False], [False, False]]
    )
    pw = compute_pos_weight(labels, mask)
    assert torch.allclose(pw, torch.tensor([1.0 / 3.0, 1.0]))

    one = torch.tensor([[True]])
    inactive = bce_loss(
        torch.tensor([[2.0]]), torch.tensor([[0.0]]), one, pos_weight=pw[:1]
    )
    active = bce_loss(
        torch.tensor([[-2.0]]), torch.tensor([[1.0]]), one, pos_weight=pw[:1]
    )
    assert torch.isclose(inactive / active, torch.tensor(3.0))


def test_compute_task_losses_skips_missing_tasks():
    batch, outputs = make_batch_and_outputs(size=2)
    batch["mask_tox"].zero_()
    losses = compute_task_losses(outputs, batch)
    assert losses["pic50"] is not None and losses["bio_cls"] is not None
    assert losses["tox"] is None

    batch["mask_bio"].zero_()
    batch["mask_tox"].fill_(True)
    losses = compute_task_losses(outputs, batch)
    assert losses["pic50"] is None and losses["bio_cls"] is None
    assert losses["tox"] is not None


def test_task_loss_from_dataset():
    df = pd.DataFrame(
        np.nan,
        index=range(3),
        columns=[*c.PIC50_COLUMNS, *c.ACTIVE_COLUMNS, *c.TOX_COLUMNS],
    )
    df.insert(0, "smiles", ["CCO", "c1ccccc1", "CC(=O)O"])
    df.loc[0, [c.PIC50_COLUMNS[0], c.ACTIVE_COLUMNS[0]]] = [7.0, 1.0]
    df.loc[1, [c.PIC50_COLUMNS[0], c.ACTIVE_COLUMNS[0]]] = [5.0, 0.0]
    df.loc[2, [c.TOX_COLUMNS[0], c.TOX_COLUMNS[1]]] = [1.0, 0.0]

    criterion = TaskLoss.from_dataset(MuFuDataset(df))
    batch, outputs = make_batch_and_outputs(size=3)
    for k in outputs:
        outputs[k].requires_grad_(True)

    losses = criterion(outputs, batch)
    sum(v for v in losses.values() if v is not None).backward()
    assert all(outputs[k].grad is not None for k in outputs)
