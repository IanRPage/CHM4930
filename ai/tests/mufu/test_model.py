import pytest
import torch
from conftest import DEVICES

from mufu.contract import MODALITIES, validate_batch
from mufu.dataset import collate, to_device
from mufu.modality_dropout import sample_modality_mask
from mufu.task_loss import TaskLoss
from mufu.uncertainty_weighting import UncertaintyWeighting


def test_eval_keeps_all_modalities_without_sampling(model, dataset):
    batch = collate([dataset[i] for i in range(len(dataset))])
    model.eval()
    rng_before = torch.get_rng_state()
    with torch.no_grad():
        predictions = model(batch)
        repeated = model(batch)

    assert torch.equal(torch.get_rng_state(), rng_before)
    for name in predictions:
        assert torch.equal(predictions[name], repeated[name])


def test_training_masks_fingerprint_gradients_per_molecule(model, dataset, monkeypatch):
    masks = []

    # Keep this test's partial masks independent of the production default.
    def sample_partial_mask(batch_size, *, device):
        mask = sample_modality_mask(batch_size, p=1.0, device=device)
        masks.append(mask)
        return mask

    monkeypatch.setattr("mufu.model.sample_modality_mask", sample_partial_mask)
    batch = collate([dataset[i] for i in range(len(dataset))])
    batch["fingerprint"].requires_grad_()
    model.train()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        predictions = model(batch)
    predictions["tox_logits"].square().mean().backward()

    assert len(masks) == 1
    retained = masks[0][:, MODALITIES.index("fingerprint")]
    gradient_norms = batch["fingerprint"].grad.norm(dim=1)
    assert retained.any() and (~retained).any()
    assert torch.equal(
        gradient_norms[~retained], torch.zeros_like(gradient_norms[~retained])
    )
    assert (gradient_norms[retained] > 0).all()


def test_single_molecule_eval_matches_batch_prediction(model, dataset):
    batch = collate([dataset[i] for i in range(len(dataset))])
    single = collate([dataset[2]])
    model.eval()
    with torch.no_grad():
        batch_predictions = model(batch)
        single_predictions = model(single)

    for name in batch_predictions:
        torch.testing.assert_close(
            single_predictions[name], batch_predictions[name][2:3], rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("device", DEVICES)
def test_mixed_batch_loss_reaches_every_trainable_part(model, labeled_dataset, device):
    batch = to_device(
        collate([labeled_dataset[i] for i in range(len(labeled_dataset))]), device
    )
    validate_batch(batch)
    model = model.to(device)
    task_loss = TaskLoss.from_dataset(labeled_dataset).to(device)
    weighting = UncertaintyWeighting().to(device)
    model.train()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        losses = task_loss(model(batch), batch)
    total = weighting(losses)
    total.backward()

    for task, loss in losses.items():
        assert loss is not None and torch.isfinite(loss), task
    assert torch.isfinite(total)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name
    for task, log_var in weighting.log_vars.items():
        assert log_var.grad is not None and log_var.grad != 0, task


def test_tiny_batch_loss_decreases(model, labeled_dataset):
    batch = collate([labeled_dataset[i] for i in range(len(labeled_dataset))])
    task_loss = TaskLoss.from_dataset(labeled_dataset)
    weighting = UncertaintyWeighting()
    optimizer = torch.optim.Adam(
        [*model.parameters(), *weighting.parameters()], lr=1e-3
    )
    model.train()

    totals = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for _ in range(30):
            optimizer.zero_grad()
            total = weighting(task_loss(model(batch), batch))
            total.backward()
            optimizer.step()
            totals.append(total.item())

    assert totals[-1] < 0.1 * totals[0]
