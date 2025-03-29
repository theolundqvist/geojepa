import torch
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src import log
from lightning import LightningDataModule
from src.data.components.task_dataset import (
    TaskDataset,
    collate_tasks,
    TaskBatch,
)
from src.data.components.tiles import TileBatch
from src.modules.embedding_lookup import EmbeddingLookup


class EmbTaskDataset(Dataset):
    def __init__(self, task, emb_name, split="train", cheat=True):
        backbone = EmbeddingLookup(
            f"data/embeddings/{emb_name}/pretraining_huge", cls_only=True
        )
        data = TaskDataset(
            f"data/tiles/huge/tasks/{task}",
            split,
            size="huge",
            cheat=cheat,
            load_images=False,
        )
        X_names = []
        X = []
        y = []

        loader = DataLoader(
            data,
            num_workers=16,
            batch_size=64,
            collate_fn=collate_tasks,
        )

        for batch in tqdm(loader, desc=f"Getting {split} data", total=len(loader)):
            tiles: TileBatch = batch.tiles
            labels = batch.labels
            embeddings = backbone(tiles).detach().numpy()
            for name, label, emb in zip(tiles.names(), labels, embeddings):
                X_names.append(name)
                X.append(emb.tolist())
                y.append(label.item())
        self.names = X_names
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EmbTaskDataModule(LightningDataModule):
    def __init__(
        self,
        dir: str,
        batch_size: int = 32,
        pin_memory: bool = False,
        size="huge",
        cheat=True,
        drop_last=True,
        shuffle=True,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.save_hyperparameters()

    def setup(self, stage: TrainerFn = "all"):
        d = self.hparams.dir
        kwargs = {
            "cheat": self.hparams.cheat,
            "size": self.hparams.size,
        }
        self.train_dataset = EmbTaskDataset(d, "train", **kwargs)
        self.val_dataset = EmbTaskDataset(d, "val", **kwargs)
        self.test_dataset = EmbTaskDataset(d, "test", **kwargs)

    def train_dataloader(self):
        assert self.train_dataset is not None, "Training dataset not set up"
        log.info(f"DATAMODULE: num_workers {self.hparams.num_workers}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_tasks,
        )

    def val_dataloader(self):
        assert self.val_dataset is not None, "Validation dataset not set up"
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_tasks,
        )

    def test_dataloader(self):
        assert self.test_dataset is not None, "Test dataset not set up"
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate_tasks,
        )

    def transfer_batch_to_device(self, batch: TaskBatch, device, dataloader_idx):
        return batch.to(device)


# class EmbeddingTaskDataModule(LightningDataModule):
