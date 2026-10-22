import random

import pytest

from mufu.data.sampler import AlternatingBatchSampler, MixedBatchSampler


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
            assert 2 <= len(batch) <= size + 1
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
    sampler = MixedBatchSampler(dataset, size, fraction)
    batches = list(sampler)
    assert len(sampler) == len(batches)
    bio_indices = labeled_indices(dataset, "mask_bio")
    assert_unique_batches(batches, size)
    for batch in batches:
        assert len(set(batch) & bio_indices) == int(size * fraction)
    assert {i for batch in batches for i in batch} == set(range(len(dataset)))


@pytest.mark.parametrize("counts", [(7, 3, 5), (0, 7, 5), (7, 0, 5), (0, 0, 10)])
def test_mixed_dual_labels_satisfy_quotas_without_duplicates(counts):
    dataset = MaskDataset(*counts, unlabeled=3)
    sampler = MixedBatchSampler(dataset, 6)
    batches = list(sampler)
    assert len(sampler) == len(batches)
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
        assert len(sampler) == len(batches)
        assert_unique_batches(batches, 6)
        assert {i for batch in batches for i in batch} == set(range(len(dataset)))
    assert [sampler.bio_only, sampler.tox_only, sampler.both] == original


@pytest.mark.parametrize("sampler_class", [MixedBatchSampler, AlternatingBatchSampler])
def test_seed_reproduces_batches(sampler_class):
    sampler = sampler_class(MaskDataset(20, 10, 5), 8)
    random.seed(123)
    first = list(sampler)
    assert len(sampler) == len(first)
    random.seed(123)
    assert first == list(sampler)


@pytest.mark.parametrize(
    "counts,batch_size,match",
    [
        ((1, 4, 0), 4, r"bio.*1.*2"),
        ((4, 0, 0), 4, r"tox.*0.*2"),
        ((1, 5, 1), 6, r"bio.*2.*3"),
    ],
)
def test_mixed_rejects_pool_smaller_than_quota(counts, batch_size, match):
    with pytest.raises(ValueError, match=match):
        MixedBatchSampler(MaskDataset(*counts), batch_size)


@pytest.mark.parametrize("batch_size,fraction", [(2, 0.49), (10, 0.01), (64, 0.001)])
def test_mixed_rejects_fraction_that_rounds_bio_quota_to_zero(batch_size, fraction):
    with pytest.raises(ValueError, match="batch quotas"):
        MixedBatchSampler(MaskDataset(2, 2), batch_size, fraction)


@pytest.mark.parametrize("sampler_class", [MixedBatchSampler, AlternatingBatchSampler])
@pytest.mark.parametrize("unlabeled", [0, 3])
def test_no_labeled_examples_are_rejected(sampler_class, unlabeled):
    with pytest.raises(ValueError, match="labeled examples?"):
        sampler_class(MaskDataset(unlabeled=unlabeled), 4)


@pytest.mark.parametrize("bio,tox,expected_batches", [(9, 2, 4), (2, 9, 4), (6, 6, 4)])
def test_alternating_bio_then_tox_covers_both_pools(bio, tox, expected_batches):
    dataset = MaskDataset(bio, tox, unlabeled=2)
    sampler = AlternatingBatchSampler(dataset, 4)
    batches = list(sampler)
    assert len(sampler) == expected_batches
    assert_unique_batches(batches, 4, partial=True)
    assert len(batches) == expected_batches
    for offset, mask in enumerate(("mask_bio", "mask_tox")):
        eligible = labeled_indices(dataset, mask)
        task_batches = batches[offset::2]
        assert all(set(batch) <= eligible for batch in task_batches)
        assert {i for batch in task_batches for i in batch} == eligible


@pytest.mark.parametrize(
    "bio,tox,size,expected_sizes",
    [
        (5, 3, 2, [2, 3, 3, 3]),
        (9, 6, 4, [4, 4, 5, 2]),
        (7, 6, 3, [3, 3, 4, 3]),
    ],
)
def test_alternating_merges_singleton_tails(bio, tox, size, expected_sizes):
    dataset = MaskDataset(bio, tox)
    sampler = AlternatingBatchSampler(dataset, size)
    batches = list(sampler)
    assert len(sampler) == len(expected_sizes)
    assert [len(batch) for batch in batches] == expected_sizes
    assert_unique_batches(batches, size, partial=True)
    for offset, mask in enumerate(("mask_bio", "mask_tox")):
        task_batches = batches[offset::2]
        eligible = labeled_indices(dataset, mask)
        assert all(set(batch) <= eligible for batch in task_batches)
        assert {i for batch in task_batches for i in batch} == eligible
    assert [sampler.task_for_batch(step) for step in range(len(batches))] == [
        "bio",
        "tox",
        "bio",
        "tox",
    ]


@pytest.mark.parametrize("counts", [(1, 3, 0), (3, 1, 0), (0, 0, 1)])
def test_alternating_rejects_single_molecule_task_pool(counts):
    with pytest.raises(ValueError, match="at least two"):
        AlternatingBatchSampler(MaskDataset(*counts), 4)


@pytest.mark.parametrize("batch_size", [0, 1, -1])
def test_alternating_rejects_batch_size_below_two(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        AlternatingBatchSampler(MaskDataset(3, 3), batch_size)


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


@pytest.mark.parametrize("counts", [(9, 2, 2), (2, 9, 2), (0, 0, 5)])
def test_alternating_exposes_task_for_each_batch(counts):
    dataset = MaskDataset(*counts)
    sampler = AlternatingBatchSampler(dataset, 3)
    for _ in range(2):
        # Fetch the whole epoch first to simulate sampler advancement by workers.
        batches = list(sampler)
        for step, batch in enumerate(batches):
            task = sampler.task_for_batch(step)
            assert task == ("bio" if step % 2 == 0 else "tox")
            assert set(batch) <= labeled_indices(dataset, f"mask_{task}")
