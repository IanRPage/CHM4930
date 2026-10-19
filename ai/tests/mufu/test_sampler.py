import random

import pytest

from mufu.sampler import AlternatingBatchSampler, MixedBatchSampler


class MaskRow:
    def __init__(self, values):
        self.values = values

    def any(self):
        return MaskRow([any(self.values)])

    def item(self):
        return self.values[0]


class MaskDataset:
    def __init__(self, bio_only=0, tox_only=0, both=0, unlabeled=0):
        # Labels in different columns exercise row-wise availability.
        self.masks = {
            "mask_bio": [MaskRow([False, True])] * bio_only
            + [MaskRow([False, False])] * tox_only
            + [MaskRow([True, False])] * both
            + [MaskRow([False, False])] * unlabeled,
            "mask_tox": [MaskRow([False, False])] * bio_only
            + [MaskRow([False, True])] * tox_only
            + [MaskRow([True, False])] * both
            + [MaskRow([False, False])] * unlabeled,
        }

    def __len__(self):
        return len(self.masks["mask_bio"])


@pytest.fixture(autouse=True)
def seeded_random():
    state = random.getstate()
    random.seed(17)
    try:
        yield
    finally:
        random.setstate(state)


def labeled_indices(dataset, mask):
    return {i for i, row in enumerate(dataset.masks[mask]) if row.any().item()}


def assert_unique_batches(batches, size, partial=False):
    assert batches
    for batch in batches:
        assert len(batch) == len(set(batch))
        assert len(batch) > 0
        if partial:
            assert len(batch) <= size
        else:
            assert len(batch) == size


@pytest.mark.parametrize(
    "bio,tox,size,fraction",
    [
        (100, 20, 10, 0.5),
        (20, 100, 10, 0.5),
        (13, 9, 7, 0.3),
    ],
)
def test_mixed_ratio_survives_pool_exhaustion(bio, tox, size, fraction):
    dataset = MaskDataset(bio, tox)
    batches = list(MixedBatchSampler(dataset, size, fraction))
    bio_indices = labeled_indices(dataset, "mask_bio")
    assert_unique_batches(batches, size)
    for batch in batches:
        assert len(set(batch) & bio_indices) == int(size * fraction)
    assert {i for batch in batches for i in batch} == set(range(len(dataset)))


@pytest.mark.parametrize("counts", [(7, 3, 5), (0, 7, 5), (7, 0, 5), (0, 0, 10)])
def test_mixed_dual_labels_satisfy_quotas_without_duplicates(counts):
    dataset = MaskDataset(*counts, unlabeled=3)
    batches = list(MixedBatchSampler(dataset, 6))
    bio = labeled_indices(dataset, "mask_bio")
    tox = labeled_indices(dataset, "mask_tox")
    assert_unique_batches(batches, 6)
    for batch in batches:
        assert len(set(batch) & bio) >= 3
        assert len(set(batch) & tox) >= 3
    assert {i for batch in batches for i in batch} == bio | tox


def test_mixed_repeated_epochs_preserve_coverage_and_pools():
    dataset = MaskDataset(11, 5, 4)
    sampler = MixedBatchSampler(dataset, 6)
    original = [
        pool.copy() for pool in (sampler.bio_only, sampler.tox_only, sampler.both)
    ]
    for _ in range(3):
        batches = list(sampler)
        assert_unique_batches(batches, 6)
        assert {i for batch in batches for i in batch} == set(range(len(dataset)))
    assert [sampler.bio_only, sampler.tox_only, sampler.both] == original


@pytest.mark.parametrize("sampler_class", [MixedBatchSampler, AlternatingBatchSampler])
def test_seed_reproduces_batches(sampler_class):
    sampler = sampler_class(MaskDataset(20, 10, 5), 8)
    random.seed(123)
    first = list(sampler)
    random.seed(123)
    assert first == list(sampler)


def test_mixed_infeasible_distinct_batch_raises():
    with pytest.raises(ValueError):
        list(MixedBatchSampler(MaskDataset(1, 1), 4))


@pytest.mark.parametrize("sampler_class", [MixedBatchSampler, AlternatingBatchSampler])
def test_no_labeled_examples_yields_no_batches(sampler_class):
    assert list(sampler_class(MaskDataset(unlabeled=3), 4)) == []


@pytest.mark.parametrize("bio,tox,expected_batches", [(9, 2, 6), (2, 9, 6), (6, 6, 4)])
def test_alternating_bio_then_tox_covers_both_pools(bio, tox, expected_batches):
    dataset = MaskDataset(bio, tox, unlabeled=2)
    batches = list(AlternatingBatchSampler(dataset, 4))
    assert_unique_batches(batches, 4, partial=True)
    assert len(batches) == expected_batches
    for offset, mask in enumerate(("mask_bio", "mask_tox")):
        eligible = labeled_indices(dataset, mask)
        task_batches = batches[offset::2]
        assert all(set(batch) <= eligible for batch in task_batches)
        assert {i for batch in task_batches for i in batch} == eligible


def test_alternating_partial_batches_are_not_padded():
    batches = list(AlternatingBatchSampler(MaskDataset(5, 3), 2))
    assert [len(batch) for batch in batches] == [2, 2, 2, 1, 1, 2]


def test_alternating_dual_labeled_compound_is_in_both_task_pools():
    batches = list(AlternatingBatchSampler(MaskDataset(1, 1, 1, 1), 10))
    assert len(batches) == 2
    assert set(batches[0]) == {0, 2}
    assert set(batches[1]) == {1, 2}
    assert_unique_batches(batches, 10, partial=True)


def test_alternating_all_dual_labeled_data():
    batches = list(AlternatingBatchSampler(MaskDataset(both=5), 3))
    assert_unique_batches(batches, 3, partial=True)
    assert len(batches) == 4
    for offset in (0, 1):
        assert {i for batch in batches[offset::2] for i in batch} == set(range(5))


def test_alternating_repeated_epochs_preserve_pools():
    sampler = AlternatingBatchSampler(MaskDataset(7, 3, 2), 4)
    original = [pool.copy() for pool in sampler.pools]
    for _ in range(3):
        batches = list(sampler)
        assert_unique_batches(batches, 4, partial=True)
        assert {i for batch in batches for i in batch} == set(range(12))
    assert sampler.pools == original
