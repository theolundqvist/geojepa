import warnings

import lightning as pl
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.components.tiles import TileBatch
from src.data.task_datamodule import TaskBatch
from src import log

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class RegressionTransfer(pl.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        backbone: nn.Module,
        head: nn.Module,
        selector: nn.Module,
        loss: nn.Module = nn.MSELoss(),
        cls_token: bool = True,
        compile: bool = False,
        warmup_fraction: float = 0.1,
        lr_min: float = 1e-4,
        min_value: float = float("-inf"),
        max_value: float = float("inf"),
    ):
        # incase inf is passed as a string
        min_value = float(min_value)
        max_value = float(max_value)
        super().__init__()
        # save hyperparameters to self.hparams, ignore nn.Modules since they are already checkpointed
        self.save_hyperparameters(ignore=["backbone", "head", "selector", "loss"])

        backbone.eval()
        backbone.requires_grad_(False)
        self.backbone = backbone
        self.head = head
        self.selector = selector

        # Define loss function
        self.loss_fn = loss
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.logged_images = 0
        self.should_clamp = min_value > float("-inf") or max_value < float("inf")

    def setup(self, stage=None):
        if self.hparams.compile and stage == "fit":
            self.backbone = torch.compile(self.backbone, fullgraph=True)
            self.head = torch.compile(self.head, fullgraph=True)

    def forward(self, batch: TileBatch) -> Tensor:
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        x = self.selector(batch)
        if self.hparams.cls_token:
            with torch.no_grad():
                cls, features = self.backbone(x)
            out = self.head(cls)
        else:
            with torch.no_grad():
                features = self.backbone(x)
            out = self.head(features)
        if not self.training and self.should_clamp:
            out = out.clamp(min=self.hparams.min_value, max=self.hparams.max_value)
        return out

    def training_step(self, batch: TaskBatch, batch_idx: int) -> Tensor:
        assert type(batch) == TaskBatch, f"Expected TaskBatch, got {type(batch)}"
        pred = self.forward(batch.tiles)
        loss = self.loss_fn(pred, batch.labels)
        if (
            batch.labels.min() < self.hparams.min_value
            or batch.labels.max() > self.hparams.max_value
        ):
            log.warning(
                f"Labels out of range: {batch.labels.min()} - {batch.labels.max()}"
            )
        self.log("train/mae", self.MAE(pred, batch.labels), prog_bar=True)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mse", self.MSE(pred, batch.labels), prog_bar=True)
        self.log("trainer/step", self.trainer.global_step, prog_bar=False)
        sched = self.lr_schedulers()
        self.log("trainer/lr", sched.get_last_lr()[0], prog_bar=False)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)
        if not loss.isfinite():
            self.trainer.should_stop = True
        return loss

    def validation_step(self, batch: TaskBatch, batch_idx: int) -> None:
        assert type(batch) == TaskBatch, f"Expected TaskBatch, got {type(batch)}"
        pred = self.forward(batch.tiles)
        loss = self.loss_fn(pred, batch.labels)
        self.log("val/mae", self.MAE(pred, batch.labels), prog_bar=True)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mse", self.MSE(pred, batch.labels), prog_bar=True)
        if hasattr(self.head, "get_metrics") and callable(self.head.get_metrics):
            self.log_dict(self.head.get_metrics(), on_step=False, on_epoch=True)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)
        # log.info("current lr", self.lr_schedulers().get_last_lr()[0])

    def test_step(self, batch: TaskBatch, batch_idx: int) -> None:
        assert type(batch) == TaskBatch, f"Expected TaskBatch, got {type(batch)}"
        pred = self.forward(batch.tiles)
        loss = self.loss_fn(pred, batch.labels)
        self.log("test/mae", self.MAE(pred, batch.labels), on_epoch=True)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/mse", self.MSE(pred, batch.labels), on_epoch=True)
        self.log_tb_images(batch, pred, 16)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)

    def log_tb_images(self, batch, preds, max_count) -> None:
        if self.logged_images < max_count:
            self.logged_images += 1
            # Get tensorboard logger
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, TensorBoardLogger):
                    tb_logger = logger.experiment

            # Log the images (Give them different names)
            if tb_logger is not None:
                tb_logger.add_image(
                    f"Images/tile:[{batch.tiles.names()[0]}] truth:[{batch.labels[0].item()}] pred:[{preds[0].item():.2f}] group:[{batch.tiles.group_names()[0]}]",
                    batch.tiles.original_imgs[0],
                    0,
                )

    def configure_optimizers(self):
        assert self.trainer is not None
        log.info(
            f"epochs: {self.trainer.max_epochs} batches: {self.trainer.estimated_stepping_batches}"
        )

        opt = self.hparams.optimizer(
            params=filter(lambda p: p.requires_grad, self.parameters())
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_fraction)
        warmup = LinearLR(
            opt, start_factor=0.10, end_factor=1.0, total_iters=warmup_steps
        )
        annealing = CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps, eta_min=self.hparams.lr_min
        )
        sched = SequentialLR(
            opt, schedulers=[warmup, annealing], milestones=[warmup_steps]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }
