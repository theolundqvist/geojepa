import math

from torch.utils.data import DataLoader, Dataset

from src.data.components.tile_dataset import worker_init_fn
from src.data.components.tiles import collate_tiles


def nbr_features(tile):
    return tile.nbr_features


class _SimilarBatchLengthCollate:
    def __init__(self, size_fn, batch_size, sub_batch_collate):
        self.size_fn = size_fn
        self.batch_size = batch_size
        self.sub_batch_collate = sub_batch_collate

    def __call__(self, batch):
        batch.sort(key=self.size_fn)
        full_batches = len(batch) // self.batch_size

        sub_batches = [
            batch[i * self.batch_size : (i + 1) * self.batch_size]
            for i in range(full_batches)
        ]

        last_batch_len = len(batch) % self.batch_size
        if last_batch_len > 0:
            sub_batches.append(batch[-last_batch_len:])

        collated_batches = [
            self.sub_batch_collate(sub_batch) for sub_batch in sub_batches
        ]
        return collated_batches


class SimilarSizeBatchDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size,
        size_fn=nbr_features,
        group_size: int = 4,
        drop_last=False,
        collate_fn=collate_tiles,
        worker_init_fn=worker_init_fn,
        *args,
        **kwargs,
    ):
        self.group_size = group_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.size_fn = size_fn
        self.drop_last = drop_last
        self.sub_batch_collate = collate_fn  # store original collate_fn for sub-batches

        if group_size > 1:
            collate_fn = _SimilarBatchLengthCollate(
                size_fn, batch_size, self.sub_batch_collate
            )

        # Create the DataLoader
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size * group_size,
            drop_last=False,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            *args,
            **kwargs,
        )
        self.total_length = dataset.__len__()

    def __iter__(self):
        if self.group_size == 1:
            for x in self.loader.__iter__():
                yield x
        else:
            for batch_list in self.loader.__iter__():
                for tile_batch in batch_list:
                    yield tile_batch

    def __len__(self) -> int:
        if self.drop_last:
            # Number of full batches of size `batch_size`
            return self.total_length // self.batch_size
        else:
            return math.ceil(self.total_length / self.batch_size)
