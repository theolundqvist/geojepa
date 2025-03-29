#!/usr/bin/env python3
"""
build_h5_all.py

Description:
    Builds HDF5 datasets for images and all task directories in parallel.
    Images are stored in a shared HDF5 file: dataset/images.h5
    Task datasets are stored in: dataset/tasks/{task}/{split}.h5

Usage:
    python build_h5_all.py --dataset dataset/ --task task_name --logdir logs/build_h5_all --max_workers 6 --tags data/tiles/embeddings.pkl --out dataset/
    python build_h5_all.py --dataset dataset/ --logdir logs/build_h5_all --max_workers 6 --tags data/tiles/embeddings.pkl --out dataset/
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List
import h5py
from torch.utils.data import DataLoader
import logging

from src.data.components.raw_tile_dataset import RawTileDataset
from src.data.components.tiles import Tile

script_path = os.path.dirname(os.path.realpath(__file__))


def setup_logger(name: str, log_file: str | Path, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name and log file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create a file handler for writing logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create a stream handler for writing logs to sys.stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_formatter = logging.Formatter("%(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger


def save_tile_to_hdf5(hdf5_file: h5py.File, tile: Tile, idx: int):
    """
    Saves a single Tile object to the HDF5 file at the specified index.

    Args:
        hdf5_file (h5py.File): The open HDF5 file.
        tile (Tile): The Tile object to save.
        idx (int): The index at which to save the tile.
    """
    group = hdf5_file.create_group(str(idx))

    # Save scalar attributes
    assert isinstance(tile, Tile), f"Expected Tile object, got {type(tile)}"
    group.attrs["nbr_features"] = tile.nbr_features

    compression = "gzip"
    # Save tensors
    group.create_dataset(
        "tile_coord", data=tile.tile_coord.numpy(), compression=compression
    )
    group.create_dataset(
        "group_coord", data=tile.group_coord.numpy(), compression=compression
    )
    group.create_dataset("nodes", data=tile.nodes.numpy(), compression=compression)
    group.create_dataset(
        "bbox_local_coords",
        data=tile.bbox_local_coords.numpy(),
        compression=compression,
    )
    group.create_dataset(
        "inter_edges", data=tile.inter_edges.numpy(), compression=compression
    )
    group.create_dataset(
        "intra_edges", data=tile.intra_edges.numpy(), compression=compression
    )
    group.create_dataset(
        "node_to_feature", data=tile.node_to_feature.numpy(), compression=compression
    )
    group.create_dataset(
        "min_boxes", data=tile.min_boxes.numpy(), compression=compression
    )
    # group.create_dataset('box_areas', data=tile.box_areas.numpy(), compression=compression)
    # group.create_dataset('box_widths', data=tile.box_widths.numpy(), compression=compression)
    # group.create_dataset('box_heights', data=tile.box_heights.numpy(), compression=compression)
    # group.create_dataset('box_rotations', data=tile.box_rotations.numpy(), compression=compression)

    # group.create_dataset('SAT_img', data=tile.SAT_img.numpy(), compression=compression) # written to dataset/images.h5 instead
    group.create_dataset("tags", data=tile.tags.numpy(), compression=compression)


def collate_fn(tiles_list: List[List[Tile]]) -> List[Tile]:
    return [tile for tiles in tiles_list for tile in tiles]


def process_task(dataset_dir: Path, task_dir: Path, log_dir: Path):
    """
    Processes a single task directory: builds HDF5 datasets for train/val/test splits.
    """
    task_name = Path(task_dir).stem
    logger = setup_logger(task_name, os.path.join(log_dir, f"{task_name}.log"))

    try:
        logger.info(f"Starting build_h5 for task: {task_name}")
        start_time_total = time.time()

        for split in ["train", "test", "val"]:
            # Initialize your TileDataset
            task_dataset = RawTileDataset(
                task_dir=task_dir,
                split=split,
                load_images=False,
                image_dir=str(dataset_dir / "images"),
                tag_embeddings_file=str(dataset_dir / "../embeddings.pkl"),
            )

            loader = DataLoader(
                task_dataset,
                batch_size=5,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
                num_workers=6,
                persistent_workers=True,
                pin_memory=False,
            )

            # Define output HDF5 path
            hdf5_path = task_dir / f"{split}.h5"

            logger.info(f"Writing {task_name}/{split}")

            # Open an HDF5 file for writing
            with h5py.File(hdf5_path, "w") as hdf5_file:
                i = 0
                used = set()
                start_time_split = time.time()

                for tiles in loader:
                    for tile in tiles:
                        assert i not in used, (
                            f"Duplicate index {i} found in split {split}"
                        )
                        if tile.name() == "16_11268_26027":
                            print("found the tile", tile.name(), i, tile.group_name())
                        if tile.group_name() == "14_2817_6506":
                            print(
                                "\nFound 14_2817_6506 but not 16, count from group: 1"
                            )
                        save_tile_to_hdf5(hdf5_file, tile, i)
                        used.add(i)
                        i += 1

                elapsed_split = time.time() - start_time_split
                logger.info(
                    f"Completed split '{split}': {i} tiles in {elapsed_split:.2f}s ({i / elapsed_split:.2f} tiles/s)"
                )

            logger.info(
                f"All tiles for split '{split}' have been successfully saved to '{dataset_dir.name}/{task_name}/{split}.h5'."
            )

        elapsed_total = time.time() - start_time_total
        logger.info(f"Completed build_h5 for task: {task_name} in {elapsed_total:.2f}s")

    except Exception as e:
        logger.error(f"Error processing task {task_name}: {e}", exc_info=True)


def process_images(dataset_dir: str, log_dir: str):
    """
    Processes images and saves them into a shared HDF5 file.

    Args:
        dataset_dir (str): Parent dataset directory containing images.
        log_dir (str): Path to the directory where logs are stored.
    """
    logger = setup_logger(
        "build_h5_datasets.process_images", os.path.join(log_dir, "images.log")
    )

    try:
        logger.info("Starting build_h5 for images.")
        start_time_total = time.time()

        # Initialize your ImageDataset
        # Assuming RawTileDataset can handle image data with split="images"
        for split in ["train", "val", "test"]:
            image_dataset = RawTileDataset(
                task_dir=os.path.join(
                    dataset_dir, "tasks", "pretraining"
                ),  # Adjust if different
                image_dir=os.path.join(dataset_dir, "images"),
                split=split,  # Custom split name
                load_images=True,
                tag_embeddings_file=str(Path(dataset_dir) / "../embeddings.pkl"),
            )

            loader = DataLoader(
                image_dataset,
                batch_size=10,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=12,
                persistent_workers=True,
                pin_memory=False,
            )

            # Define output HDF5 path
            hdf5_path = str(Path(dataset_dir) / "images.h5")

            logger.info(f"Writing {split} images")

            # Open an HDF5 file for writing
            count = 0
            start_time_split = time.time()
            with h5py.File(hdf5_path, "a") as hdf5_file:
                for tiles in loader:
                    for i, tile in enumerate(tiles):
                        if hdf5_file.get(tile.name()) is None:
                            assert isinstance(tile, Tile), (
                                f"Expected Tile object, got {type(tile)}"
                            )
                            group = hdf5_file.create_group(tile.name())

                            # Save scalar attributes
                            # Save tensors
                            group.create_dataset(
                                "SAT_img", data=tile.SAT_img.numpy(), compression="gzip"
                            )
                            count += 1

            elapsed_split = time.time() - start_time_split
            logger.info(
                f"Completed {split} images: {count} new images in {elapsed_split:.2f}s ({count / elapsed_split:.2f} tiles/s)"
            )

        elapsed_total = time.time() - start_time_total
        logger.info(f"Completed build_h5 for images in {elapsed_total:.2f}s")

    except Exception as e:
        logger.error(f"Error processing images: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build HDF5 datasets for all task directories and images in parallel."
    )
    # images dir => dataset/images,
    # task dir => dataset/tasks/{task}/{split}/{tile}.pbf,
    # tags_file => dataset/../embeddings.pkl
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Parent directory containing tasks directory and images directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=False,
        help="If specified, only build HDF5 for the specified task.",
    )
    parser.add_argument("--images", action="store_true")
    parser.add_argument(
        "--logdir",
        type=str,
        default=f"{script_path}/logs/build_h5_all",
        help="Directory path for logs.",
    )
    parser.add_argument(
        "-n",
        "--max_workers",
        type=int,
        default=10,
        help="Maximum number of parallel processes.",
    )
    args = parser.parse_args()

    log_dir = Path(args.logdir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    main_logger = setup_logger("build_h5_datasets.main", log_dir / "main.log")

    dataset_dir = Path(args.dataset).resolve()
    if not os.path.exists(dataset_dir):
        dataset_dir = Path(script_path) / "../data" / "tiles" / args.dataset
        if not os.path.exists(dataset_dir):
            main_logger.error(
                f"Dataset directory '{dataset_dir}' does not exist. Exiting."
            )
            return

    # Setup main logger
    main_logger.info(f"Starting build_h5_all with dataset_dir={dataset_dir}")

    # Discover all task directories
    task_dirs = []
    if args.task:
        # Process only the specified task
        d = dataset_dir / "tasks" / args.task
        task_dirs = [d]
        if not d.is_dir():
            main_logger.error(
                f"Specified task directory '{dataset_dir / 'tasks' / args.task}' does not exist. Exiting."
            )
            return
    # else:
    #     # Process all tasks
    #     task_dirs = [p for p in (dataset_dir / "tasks").glob("*") if p.is_dir()]
    #     if not task_dirs:
    #         main_logger.error(f"No task directories found in '{dataset_dir / 'tasks'}'.")
    #         return

    main_logger.info(f"Discovered {len(task_dirs)} task directories to process.")

    # Initialize ProcessPoolExecutor
    start = time.time()

    # Submit image processing
    if args.images:
        process_images(dataset_dir=str(dataset_dir), log_dir=str(log_dir))

    # Submit task processing
    for task_dir in task_dirs:
        process_task(dataset_dir=dataset_dir, task_dir=task_dir, log_dir=log_dir)

    elapsed = time.time() - start
    main_logger.info(f"All build_h5 tasks have been processed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
