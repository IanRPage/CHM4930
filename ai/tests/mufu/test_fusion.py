import pytest
import torch

from mufu.contract import D_MODEL, DROPOUT_MASKS, FUSED_DIM, MODALITIES
from mufu.fusion import LateFusion
from mufu.modality_dropout import sample_modality_mask


def make_projected(batch_size=4, requires_grad=False):
    return {
        m: torch.randn(batch_size, D_MODEL, requires_grad=requires_grad)
        for m in MODALITIES
    }


def test_output_shape_and_gradients():
    batch_size = 4
    fusion = LateFusion()
    projected = make_projected(batch_size, requires_grad=True)
    output = fusion(projected)

    assert output.shape == (batch_size, FUSED_DIM)
    output.sum().backward()
    assert all(projected[m].grad is not None for m in MODALITIES)
    assert all(p.grad is not None for p in fusion.parameters())


def test_no_mask_equals_all_present():
    fusion = LateFusion()
    projected = make_projected()
    mask = torch.ones(4, len(MODALITIES), dtype=torch.bool)

    assert torch.equal(fusion(projected), fusion(projected, mask))


@pytest.mark.parametrize("dropped", MODALITIES)
def test_masked_modality_is_ignored(dropped):
    fusion = LateFusion()
    projected = make_projected()
    mask = torch.ones(4, len(MODALITIES), dtype=torch.bool)
    mask[:, MODALITIES.index(dropped)] = False

    changed = {**projected, dropped: torch.randn(4, D_MODEL)}
    assert torch.equal(fusion(projected, mask), fusion(changed, mask))


def test_sampled_masks_ignore_dropped_content_and_block_its_gradients():
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(42)
        fusion = LateFusion()
        projected = make_projected(batch_size=32, requires_grad=True)
        mask = sample_modality_mask(
            32, p=0.3, generator=torch.Generator().manual_seed(17)
        )
        output = fusion(projected, mask)

    changed = {m: value.detach().clone() for m, value in projected.items()}
    for i, m in enumerate(MODALITIES):
        changed[m][~mask[:, i]] += 1000
    with torch.no_grad():
        assert torch.equal(output, fusion(changed, mask))

    output.square().mean().backward()
    for i, m in enumerate(MODALITIES):
        grad = projected[m].grad
        assert grad[~mask[:, i]].count_nonzero() == 0
        assert (grad[mask[:, i]].norm(dim=1) > 0).all()


def test_rejects_bad_keys():
    projected = make_projected()
    del projected["graph"]

    with pytest.raises(ValueError, match="projected: missing keys"):
        LateFusion()(projected)


def test_rejects_wrong_width():
    projected = {**make_projected(), "fingerprint": torch.randn(4, 128)}

    with pytest.raises(ValueError, match="fingerprint: expected shape"):
        LateFusion()(projected)


def test_rejects_batch_size_mismatch():
    projected = {**make_projected(), "graph": torch.randn(3, D_MODEL)}

    with pytest.raises(ValueError, match="graph: expected shape"):
        LateFusion()(projected)


def test_rejects_wrong_mask_shape():
    mask = torch.ones(4, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="mask: expected shape"):
        LateFusion()(make_projected(), mask)


def test_rejects_non_bool_mask():
    mask = torch.ones(4, len(MODALITIES))

    with pytest.raises(ValueError, match="mask: expected dtype bool"):
        LateFusion()(make_projected(), mask)


def test_rejects_a_fully_missing_sample_in_a_mixed_batch():
    mask = torch.ones(4, len(MODALITIES), dtype=torch.bool)
    mask[2] = False

    with pytest.raises(ValueError):
        LateFusion()(make_projected(), mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_mask_with_cuda_inputs():
    fusion = LateFusion().cuda()
    projected = {m: t.cuda() for m, t in make_projected().items()}
    mask = torch.tensor(DROPOUT_MASKS[:4])

    assert fusion(projected, mask).device == projected[MODALITIES[0]].device
