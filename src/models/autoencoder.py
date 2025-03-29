import warnings
from typing import Tuple, Callable

import lightning as pl
import torch
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.components.tiles import TileBatch
from src import log
from src.modules.tag_models import TagCountAE
from src.modules.tokenizer import Modality
from src.utils.logging import log_emb_image, visualize_representation_space

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class AutoEncoder(pl.LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module = TagCountAE,
        loss: nn.Module = nn.SmoothL1Loss(beta=1.0),
        compile: bool = False,
        warmup_fraction: float = 0.1,
        lr_min: float = 1e-4,
    ):
        super().__init__()
        # save hyperparameters to self.hparams, ignore nn.Modules since they are already checkpointed
        self.save_hyperparameters(ignore=["loss", "model"])
        self.model = model

        # Define loss function
        self.loss_fn = loss
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.logged_images = 0
        self.debug_emb_i = None
        self.debug_emb_bool = None

    def setup(self, stage=None):
        if self.hparams.compile and stage == "fit":
            log.info("Compiling model")
            if hasattr(self.model, "compile_supported"):
                self.model.compile_supported()
            else:
                self.model = torch.compile(self.model, fullgraph=True)

    def _model_step(
        self, batch: TileBatch, debug=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        self.debug_emb_bool = debug
        x, enc, dec = self.model(batch, decode=True)
        self.debug_emb(x, "ae/truth")
        self.debug_emb(enc, "ae/enc")
        self.debug_emb(dec, "ae/dec")
        self.debug_emb(x - dec, "ae/x_diff_dec")
        return x, enc, dec

    def forward(self, batch: TileBatch):
        enc = self.model(batch, decode=False)
        return enc

    def training_step(self, batch: TileBatch, batch_idx: int) -> Tensor:
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        x, enc, dec = self._model_step(batch)
        loss = self.loss_fn(dec, x)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/mae", self.MAE(dec, x), prog_bar=True)
        self.log("train/mse", self.MSE(dec, x), prog_bar=True)
        self.log("trainer/step", self.trainer.global_step, prog_bar=False)
        sched = self.lr_schedulers()
        self.log("trainer/lr", sched.get_last_lr()[0], prog_bar=False)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)
        if not loss.isfinite():
            self.trainer.should_stop = True
        return loss

    def validation_step(self, batch: TileBatch, batch_idx: int) -> None:
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        name = "16_10576_25408"
        index = (
            -1 if name not in batch.names() else batch.names().index("16_10576_25408")
        )
        x, enc, dec = self._model_step(batch, debug=(index != -1))
        loss = self.loss_fn(dec, x)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/mae", self.MAE(dec, x), prog_bar=True)
        self.log("val/mse", self.MSE(dec, x), prog_bar=True)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)

    def test_step(self, batch: TileBatch, batch_idx: int) -> None:
        assert type(batch) == TileBatch, f"Expected TileBatch, got {type(batch)}"
        x, enc, dec = self._model_step(batch)
        loss = self.loss_fn(dec, x)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/mae", self.MAE(dec, x), on_epoch=True)
        self.log("test/mse", self.MSE(dec, x), on_epoch=True)
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        self.log("gpu/mb", mem_used_MB, prog_bar=True)

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
            opt, start_factor=0.0001, end_factor=1.0, total_iters=warmup_steps
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

    def debug_emb(self, tokens: torch.Tensor | Callable, name):
        if not self.debug_emb_bool:
            return
        if type(tokens) is not torch.Tensor:
            tokens = tokens()

        # use first sample in batch
        if len(tokens.shape) == 3:
            tokens = tokens[0]
        mods = torch.ones(tokens.shape[:-1], device=self.device) * Modality.OSM
        log_emb_image(self.get_tb_logger(), tokens, name, self.trainer.global_step)
        log.info(f"mods: {mods.shape}, tokens: {tokens.shape}")
        visualize_representation_space(
            self.get_tb_logger(), mods, tokens, name, self.trainer.global_step
        )

    def get_tb_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        return None
