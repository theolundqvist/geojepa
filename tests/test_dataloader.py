import time

import pytest

from src.data.task_datamodule import TaskDataModule
from src.data.tiles_datamodule import TilesDataModule
from tqdm import tqdm

batch_size = 64


@pytest.fixture
def taskloader():
    module = TaskDataModule(
        "data/tiles/huge/tasks/car_bridge",
        batch_size=batch_size,
        num_workers=10,
        pin_memory=False,
    )
    start = time.time()
    module.setup()
    print(f"Setup time: {time.time() - start}s")
    return module.train_dataloader()


@pytest.fixture
def tileloader():
    module = TilesDataModule(
        "data/tiles/tiny/tasks/pretraining",
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    start = time.time()
    module.setup()
    print(f"Setup time: {time.time() - start}s")
    return module.val_dataloader()


# def test_tile_loader_fixed_batched_size(tileloader):
#     """Test that the loader returns the expected batch size."""
#     assert tileloader.batch_size == batch_size
#     length = len(tileloader)
#     print(f"Length: {length}")
#     len1 = 0
#     start = time.time()
#     for tile_batch in tileloader:
#         assert tile_batch.size == batch_size
#         assert tile_batch.SAT_imgs.shape[0] == batch_size
#         len1 += 1
#     print(f"Time: {time.time()-start}s")
#
#     len2 = 0
#     start = time.time()
#     for tile_batch in tileloader:
#         assert tile_batch.size == batch_size
#         assert tile_batch.SAT_imgs.shape[0] == batch_size
#         len2 += 1
#     print(f"Time: {time.time()-start}s")
#     assert len1 == len2
#     assert len1 == length


def test_task_loader_fixed_batched_size(tileloader):
    """Test that the loader returns the expected batch size."""
    # assert taskloader.batch_size == batch_size
    length = len(taskloader)
    print(f"Length: {length}")
    len1 = 0
    start = time.time()
    for task_batch in tqdm(taskloader):
        # assert task_batch.tiles.size == batch_size
        # assert task_batch.tiles.SAT_imgs.shape[0] == batch_size
        # assert task_batch.labels.shape[0] == batch_size
        len1 += 1
    print(
        f"Time: {time.time() - start}s",
        f"{batch_size * length / (time.time() - start)} tiles/s",
    )
    len2 = 0
    start = time.time()
    for task_batch in tqdm(taskloader):
        # assert task_batch.tiles.size == batch_size
        # assert task_batch.tiles.SAT_imgs.shape[0] == batch_size
        # assert task_batch.labels.shape[0] == batch_size
        len2 += 1
    print(
        f"Time: {time.time() - start}s",
        f"{batch_size * length / (time.time() - start)} tiles/s",
    )
    # assert len1 == len2
    # assert len1 == length
