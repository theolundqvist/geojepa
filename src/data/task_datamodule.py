from lightning.pytorch.trainer.states import TrainerFn

from src import log
from lightning import LightningDataModule

from src.data.components.task_dataset import (
    TaskDataset,
    size_fn,
    collate_tasks,
    TaskBatch,
)
from src.utils.data_utils import SimilarSizeBatchDataLoader


class TaskDataModule(LightningDataModule):
    def __init__(
        self,
        dir: str,
        batch_size: int = 32,
        group_size: int = 8,
        num_workers: int = 10,
        pin_memory: bool = False,
        multiprocessing_context="fork",
        persistent_workers=False,
        use_image_transforms=True,
        load_images=True,
        size="huge",
        cheat=False,
        cache=False,
        drop_last=True,
        shuffle=True,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters()
        log.info(f"DATAMODULE: num_workers {num_workers}")

    def setup(self, stage: TrainerFn = "all"):
        d = self.hparams.dir
        kwargs = {
            "load_images": self.hparams.load_images,
            "use_image_transforms": self.hparams.use_image_transforms,
            "cheat": self.hparams.cheat,
            "size": self.hparams.size,
            "cache": self.hparams.cache,
        }
        self.train_dataset = TaskDataset(d, "train", **kwargs)
        self.val_dataset = TaskDataset(d, "val", **kwargs)
        self.test_dataset = TaskDataset(d, "test", add_original_image=True, **kwargs)

    def train_dataloader(self):
        assert self.train_dataset is not None, "Training dataset not set up"
        log.info(f"DATAMODULE: num_workers {self.hparams.num_workers}")
        return SimilarSizeBatchDataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            group_size=self.hparams.group_size,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistent_workers,
            multiprocessing_context=self.hparams.multiprocessing_context,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_tasks,
            size_fn=size_fn,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "Validation dataset not set up"
        return SimilarSizeBatchDataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            group_size=self.hparams.group_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistent_workers,
            multiprocessing_context=self.hparams.multiprocessing_context,
            collate_fn=collate_tasks,
            size_fn=size_fn,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test dataset not set up"
        return SimilarSizeBatchDataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            group_size=self.hparams.group_size,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0
            and self.hparams.persistent_workers,
            multiprocessing_context=self.hparams.multiprocessing_context,
            collate_fn=collate_tasks,
            size_fn=size_fn,
            pin_memory=self.hparams.pin_memory,
        )

    def transfer_batch_to_device(self, batch: TaskBatch, device, dataloader_idx):
        return batch.to(device)


# class EmbeddingTaskDataModule(LightningDataModule):
