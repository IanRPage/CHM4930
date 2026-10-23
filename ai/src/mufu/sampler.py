import random

from torch.utils.data import Sampler

from mufu.dataset import MuFuDataset


class MixedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: MuFuDataset,
        batch_size: int,
        bio_fraction: float = 0.5,  # ratio bio data/tox data
    ):

        n_bio = int(batch_size * bio_fraction)
        n_tox = batch_size - n_bio
        if n_bio < 1 or n_tox < 1:
            raise ValueError(
                "batch quotas must allocate at least one example to each task"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.bio_fraction = bio_fraction

        self.bio_only = []
        self.tox_only = []
        self.both = []

        for i in range(len(dataset)):
            has_bio = dataset.masks["mask_bio"][i].any().item()
            has_tox = dataset.masks["mask_tox"][i].any().item()

            if has_bio and has_tox:
                self.both.append(i)
            elif has_bio:
                self.bio_only.append(i)
            elif has_tox:
                self.tox_only.append(i)

        if not (self.bio_only or self.tox_only or self.both):
            raise ValueError(
                "Mixed batching requires at least one labeled example per task"
            )

        for task, pool_size, quota in (
            ("bio", len(self.bio_only) + len(self.both), n_bio),
            ("tox", len(self.tox_only) + len(self.both), n_tox),
        ):
            if pool_size < quota:
                raise ValueError(
                    f"{task} pool has {pool_size} labeled examples, fewer than quota {quota}"
                )

    def __iter__(self):
        bio_only = self.bio_only.copy()
        tox_only = self.tox_only.copy()
        both = self.both.copy()

        random.shuffle(bio_only)
        random.shuffle(tox_only)
        random.shuffle(both)

        n_bio = int(self.batch_size * self.bio_fraction)
        n_tox = self.batch_size - n_bio

        # Keep both separate for initial draws; count its labels toward either quota.
        bio_labels = set(self.bio_only + self.both)
        tox_labels = set(self.tox_only + self.both)
        all_indices = bio_labels | tox_labels

        bio_i = 0
        tox_i = 0
        both_i = 0

        while bio_i < len(bio_only) or tox_i < len(tox_only) or both_i < len(both):
            batch = []

            bio_take = bio_only[bio_i : bio_i + n_bio]
            tox_take = tox_only[tox_i : tox_i + n_tox]

            batch.extend(bio_take)
            batch.extend(tox_take)

            bio_i += len(bio_take)
            tox_i += len(tox_take)

            # Fill any missing spots with examples that have both labels
            remaining = self.batch_size - len(batch)

            if remaining > 0:
                both_take = both[both_i : both_i + remaining]
                batch.extend(both_take)
                both_i += len(both_take)

            # Reuse examples across batches when needed, never within a batch.
            for eligible, quota in ((bio_labels, n_bio), (tox_labels, n_tox)):
                selected = set(batch)
                missing = max(0, quota - len(selected & eligible))
                if missing > 0:
                    batch.extend(random.sample(list(eligible - selected), missing))

            # Both-label examples can satisfy both quotas with fewer slots.
            remaining = self.batch_size - len(batch)
            if remaining > 0:
                batch.extend(random.sample(list(all_indices - set(batch)), remaining))

            random.shuffle(batch)
            yield batch


class AlternatingBatchSampler(Sampler[list[int]]):
    task_names = ("bio", "tox")

    def __init__(self, dataset: MuFuDataset, batch_size: int):
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")

        self.batch_size = batch_size
        self.pools = []

        for task in self.task_names:
            pool = []
            for i, labels in enumerate(dataset.masks[f"mask_{task}"]):
                if labels.any().item():
                    pool.append(i)
            # the batch samplers are intended to be used with non-empty datasets
            if not pool:
                raise ValueError(
                    "Alternating batching requires labeled examples for both tasks"
                )

            if len(pool) < 2:
                raise ValueError(
                    "Alternating batching requires at least two labeled examples for each task"
                )

            self.pools.append(pool)

    def task_for_batch(self, batch_index: int) -> str:
        return self.task_names[batch_index % len(self.task_names)]

    def __iter__(self):
        pools = [pool.copy() for pool in self.pools]
        for pool in pools:
            random.shuffle(pool)
        positions = [0] * len(pools)

        rounds = 0
        for pool in pools:
            full_batches, remainder = divmod(len(pool), self.batch_size)
            # Merge a batch smaller than 2
            pool_rounds = full_batches + int(remainder >= 2)
            rounds = max(rounds, pool_rounds, 1)

        for each in range(rounds):
            for task, pool in enumerate(pools):
                if positions[task] == len(pool):
                    random.shuffle(pool)
                    positions[task] = 0
                start = positions[task]
                end = min(start + self.batch_size, len(pool))
                if len(pool) - end == 1:
                    end += 1
                batch = pool[start:end]
                positions[task] += len(batch)
                yield batch
