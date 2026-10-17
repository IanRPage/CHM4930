import pytest
import torch

from mufu.contract import D_MODEL, FUSED_DIM, MODALITIES
from mufu.fusion import LateFusion


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
