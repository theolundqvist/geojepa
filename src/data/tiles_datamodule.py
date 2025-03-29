from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from src.data.components.tile_dataset import TileDataset, worker_init_fn
from src.data.components.tiles import collate_tiles, TileBatch
from src.utils.data_utils import SimilarSizeBatchDataLoader


class TilesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dir: str,
        batch_size: int = 32,
        group_size: int = 8,
        num_workers: int = 10,
        pin_memory: bool = False,
        load_images: bool = True,
        use_image_transforms: bool = True,
        cheat=False,
        size="huge",
        cache=False,
        shuffle=True,
        drop_last=True,
    ):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        d = self.hparams.dir
        kwargs = {
            "load_images": self.hparams.load_images,
            "use_image_transforms": self.hparams.use_image_transforms,
            "cheat": self.hparams.cheat,
            "size": self.hparams.size,
            "cache": self.hparams.cache,
        }
        self.train_dataset = TileDataset(d, "train", **kwargs)
        self.val_dataset = TileDataset(d, "val", add_original_image=True, **kwargs)
        self.test_dataset = TileDataset(d, "test", add_original_image=True, **kwargs)

    def train_dataloader(self) -> SimilarSizeBatchDataLoader | DataLoader:
        return SimilarSizeBatchDataLoader(
            self.train_dataset,
            group_size=self.hparams.group_size,
            # return DataLoader(
            #     self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_tiles,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> SimilarSizeBatchDataLoader | DataLoader:
        # return DataLoader(
        # self.val_dataset,
        return SimilarSizeBatchDataLoader(
            self.val_dataset,
            group_size=self.hparams.group_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=collate_tiles,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> SimilarSizeBatchDataLoader | DataLoader:
        # return DataLoader(
        # self.test_dataset,
        return SimilarSizeBatchDataLoader(
            self.test_dataset,
            group_size=self.hparams.group_size,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            collate_fn=collate_tiles,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def transfer_batch_to_device(self, batch: TileBatch, device, dataloader_idx):
        return batch.to(device)
