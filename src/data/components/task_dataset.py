from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import torch
from torch import tensor

from src.data.components.tile_dataset import TileDataset
from src.data.components.tiles import TileBatch, Tile, collate_tiles


@dataclass
class TaskBatch:
    tiles: TileBatch
    labels: tensor
    device: torch.device

    def to(self, device: torch.device):
        self.labels = self.labels.to(device)
        self.tiles = self.tiles.to(device)
        self.device = device
        return self


import pandas as pd
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    def __init__(
        self,
        task_dir: str | Path,
        split="train",
        use_image_transforms=True,
        add_original_image=False,
        load_images=True,
        size="huge",
        cheat=False,
        cache=False,
    ):
        self.tile_dataset = TileDataset(
            task_dir=task_dir,
            split=split,
            use_image_transforms=use_image_transforms,
            add_original_image=add_original_image,
            load_images=load_images,
            cheat=cheat,
            size=size,
            cache=cache,
        )
        if size != "huge":
            task_dir = task_dir.replace("huge", size)
        if cheat:
            task_dir += "_cheat"
        self.label_file = Path(task_dir) / "labels.txt"

        # Load labels using pandas
        try:
            self.labels_df = pd.read_csv(
                self.label_file,
                sep=":",
                header=None,
                names=["tile_name", "label"],
                dtype={"tile_name": str, "label": float},
                engine="c",  # Faster parsing
                memory_map=True,  # Efficient memory usage
            )
        except Exception as e:
            raise ValueError(f"Error reading label file {self.label_file}: {e}")

        # Validate tile names
        invalid_tiles = self.labels_df[
            ~self.labels_df["tile_name"].str.contains(r"^\d+_\d+_\d+$")
        ]
        if not invalid_tiles.empty:
            invalid_tile_names = invalid_tiles["tile_name"].tolist()
            raise ValueError(f"Invalid tile names found: {invalid_tile_names}")

        # Set tile_name as index for faster lookup
        self.labels_df.set_index("tile_name", inplace=True)

    def __len__(self):
        return len(self.tile_dataset)

    def __init_worker__(self):
        self.setup()

    def setup(self):
        self.tile_dataset.__init_worker__()

    def __getitem__(self, idx):
        tile = self.tile_dataset[idx]
        tile_name = tile.name()
        try:
            label = self.labels_df.at[tile_name, "label"]
        except KeyError:
            raise KeyError(
                f"Label for tile '{tile_name}' not found in {self.label_file}."
            )
        return tile, label


def collate_tasks(batch: List[Tuple[Tile, float]]) -> TaskBatch:
    tiles, labels = zip(*batch)
    label_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    tile_batch = collate_tiles(tiles)
    return TaskBatch(tile_batch, label_tensor, device=torch.device("cpu"))


def size_fn(task: Tuple[Tile, float]) -> int:
    return task[0].nbr_features
