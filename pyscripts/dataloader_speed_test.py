import time
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.components.tile_dataset import TileDataset
from src.data.components.tiles import collate_tiles
from tqdm import tqdm

from src.utils.data_utils import SimilarSizeBatchDataLoader


# profile_runner.py


def test(dataset, num_workers, pin_memory, batch_size, group_size):
    dir = "data/tiles/huge/tasks/pretraining"
    split = "train"
    cnf = {
        "workers": num_workers,
        "pin_mem": pin_memory,
        "batch_size": batch_size,
        "group_size": group_size,
    }
    print(f"\n----------------------------------------\n\n{dataset.__name__}, {cnf}")
    start = time.time()
    dataset = dataset(dir, split, load_images=True)
    print(f"Setup time: {time.time() - start}s")
    length = len(dataset)
    loader = SimilarSizeBatchDataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_tiles,
        pin_memory=pin_memory,
        group_size=group_size,
        persistent_workers=num_workers > 0,
    )
    start = time.time()
    for batch in tqdm(loader):
        if "16_11268_26027" in batch.names():
            print("found the tile")
            exit(0)
        if "14_2817_6506" in batch.group_names():
            print(
                "\nFound 14_2817_6506 but not 16, count from group:",
                len(list(filter(lambda x: x == "14_2817_6506", batch.group_names()))),
            )
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}")
    # start = time.time()
    #
    # for _ in tqdm(loader):
    #     pass
    # print(f"Time: {time.time() - start}s")
    # print(f"tiles/s: {length / (time.time() - start)}")


def run():
    test(TileDataset, num_workers=10, pin_memory=False, batch_size=64, group_size=16)
    test(TileDataset, num_workers=10, pin_memory=True, batch_size=5, group_size=1)
    test(TileDataset, num_workers=10, pin_memory=True, batch_size=5, group_size=2)
    test(TileDataset, num_workers=10, pin_memory=True, batch_size=5, group_size=3)
    test(TileDataset, num_workers=10, pin_memory=True, batch_size=5, group_size=4)


if __name__ == "__main__":
    run()
