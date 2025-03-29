from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
import time
import warnings
from typing import Dict, List, Tuple, Callable

import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src import log
from src.data.components.tiles import TileBatch
from src.lightning_utils.logging_utils import prefix_keys
from src.modules.losses import (
    vectorised_masked_vicreg_loss,
    vectorised_masked_smooth_l1_loss,
)
from src.modules.masks import MaskingStrategy, apply_mask
from src.modules.mlp import MLP
from src.modules.tokenizer import Modality
from src.modules.vision_backbones import ViTB16
from src.utils.logging import visualize_representation_space, log_emb_image
from src.modules.linear_ema import LinearEMA

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class GEOJEPA(pl.LightningModule):
    """
    Generic JEPA encoder training,
    check ex config/models/geojepa.yaml for example.
    """

    def __init__(
        self,
        token_dim: int,
        encoder: nn.Module,
        predictor: nn.Module,
        masking_strategies: Dict[str, MaskingStrategy],
        masking_strategy_chances: Dict[str, float],
        tokenizer: nn.Module,
        compile: bool = False,
        momentum_init: float = 0.99,
        momentum_end: float = 1.0,
        lr_base: float = 1e-3,
        lr_end: float = 1e-6,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        wd_init: float = 0.05,
        wd_end: float = 0.05,
        warmup_frac: float = 0.1,
        vicreg_beta: float = 0.008,
        smooth_l1_beta: float = 1.0,
        use_augmentations: bool = False,
        log_images: bool = True,
        log_image_interval: int = 2,
    ):
        super().__init__()
        # save hyperparameters to self.hparams, ignore nn.Modules since they are already checkpointed
        self.save_hyperparameters(ignore=["encoder", "predictor", "tokenizer"])

        # ----------------------------------------
        # Hyperparameters
        # self.avg_sim_coeff = avg_sim_coeff
        # self.info_nce_tau = info_nce_tau
        self.warmup_frac = warmup_frac
        self.masking_strategies = masking_strategies
        self.masking_strategy_chances = masking_strategy_chances
        self.use_augmentations = use_augmentations
        self.wds = (wd_init, wd_end)
        self.lrs = (lr_base, lr_end)
        self.adam_betas = (adam_beta1, adam_beta2)
        # self.target_sims = (target_sim_init, target_sim_end)

        # ----------------------------------------
        # Live parameters
        # self.target_sim = target_sim_init
        self.wd = wd_init

        # ----------------------------------------
        # Compilation
        self.compile = compile

        # ----------------------------------------
        # Tokenizer, encoder, predictor
        self.tokenizer = tokenizer
        self.context_encoder = encoder
        self.target_encoder: nn.Module = LinearEMA(encoder, momentum_init, momentum_end)
        self.predictor = predictor

        # ----------------------------------------
        # Position and modality encoders
        self.position_encoder = MLP(8, token_dim * 4, token_dim, drop=0.1)
        self.modality_encoder = MLP(len(Modality), token_dim * 4, token_dim, drop=0.1)
        self.pos_mod_fusion = MLP(token_dim * 2, token_dim * 2, token_dim, drop=0.1)

        # ----------------------------------------
        # Timer variables
        self.timers = {}
        self.last_timer = None

        self.model_init_time = time.time()
        self.epoch_start_time = None

    def setup(self, stage=None):
        if stage == "fit":
            self.target_encoder.set_training_steps(
                self.trainer.estimated_stepping_batches
            )
            if self.hparams.compile:
                torch._dynamo.config.cache_size_limit = 64
                self.predictor = torch.compile(self.predictor, fullgraph=True)
                self.context_encoder = torch.compile(
                    self.context_encoder, fullgraph=True
                )
                self.target_encoder.ema_model = torch.compile(
                    self.target_encoder.ema_model, fullgraph=True
                )
                self.position_encoder = torch.compile(
                    self.position_encoder, fullgraph=True
                )
                self.modality_encoder = torch.compile(
                    self.modality_encoder, fullgraph=True
                )
                # self.calculate_loss = torch.compile(self.calculate_loss, fullgraph=True)
                if (
                    self.tokenizer.tokenize_images
                    and type(self.tokenizer.img_encoder) is ViTB16
                ):
                    print("Compiling ViTB16")
                    self.tokenizer.img_encoder = torch.compile(
                        self.tokenizer.img_encoder, fullgraph=True
                    )

    def forward(self, batch: TileBatch):
        tokens, positions, modalities = self.tokenizer(batch)
        pos_embs = self._pos_emb(positions)
        mod_embs = self._mod_emb(modalities)
        pos_mod_embs = self.pos_mod_fusion(torch.cat([pos_embs, mod_embs], dim=-1))

        padding_mask = self._padding_mask(modalities)

        encoded = self.target_encoder(tokens, pos_mod_embs, padding_mask=padding_mask)

        cls = torch.cat(
            [torch.max(encoded, dim=1).values, torch.mean(encoded, dim=1)], dim=1
        )
        return cls, encoded, modalities

    def _model_step(
        self, batch, debug=False, debug_i=0, return_embs=False
    ) -> (
        Tuple[Dict[str, Tensor], Dict[str, Tensor]]
        | Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]
    ):
        self.debug_emb_i = debug_i
        self.debug_emb_bool = debug
        self._startt("model_step")
        self._startt("tokenizer")
        tokens, positions, modalities = self.tokenizer(batch)
        padding_mask = self._padding_mask(modalities)
        self._endt()

        self.debug_emb(
            tokens, "1. after tokenizer", modalities, tile_names=batch.names()
        )

        self.debug_emb(positions, "2. actual positions", modalities)
        if self.use_augmentations:
            # not really tested if this is good or bad, can't really hurt?
            positions += (
                torch.randn(positions.shape, device=positions.device) * 5e-4
            )  # add [~-3,~3]*15cm noise to all positions
        self.debug_emb(positions, "3. actual positions with noise")
        self.debug_emb(
            lambda: modalities.unsqueeze(-1).expand(-1, -1, self.hparams.token_dim),
            "4. modalities",
        )

        self._startt("select_targets")
        mask_strat = self._select_masking_strategy()
        ctx_mask, tgt_masks = mask_strat(tokens, positions, modalities)
        self._endt()
        M, B, T = tgt_masks.shape
        self.debug_emb(
            lambda: torch.cat(
                [
                    ctx_mask.unsqueeze(-1).expand(-1, -1, self.hparams.token_dim),
                    tgt_masks.permute(1, 0, 2)
                    .unsqueeze(-1)
                    .expand(-1, -1, -1, self.hparams.token_dim)
                    .permute(0, 2, 1, 3)
                    .reshape(B, T, M * self.hparams.token_dim),
                ],
                dim=2,
            ),
            "5. masks (ctx + tgts)",
        )

        # -------------------
        #  Position and modality embeddings
        self._startt("pos_emb")
        pos_embs = self._pos_emb(positions)
        mod_embs = self._mod_emb(modalities)
        pos_mod_embs = self.pos_mod_fusion(torch.cat([pos_embs, mod_embs], dim=-1))
        self._endt()
        self.debug_emb(pos_embs, "6. pos embeddings", modalities)
        self.debug_emb(mod_embs, "6.5. mod embeddings", modalities)

        # --------------------
        # Create ctx
        self._startt("apply_ctx_masks")
        ctx_tokens = apply_mask(tokens, ctx_mask)
        ctx_pos = apply_mask(pos_mod_embs, ctx_mask)
        ctx_pad_mask = apply_mask(padding_mask, ctx_mask)
        self._endt()
        self.debug_emb(ctx_tokens, "7. ctx tokens", apply_mask(modalities, ctx_mask))

        # TEACHER
        self._startt("target_encoder")
        with torch.no_grad():
            global_embs = self.target_encoder(tokens, pos_mod_embs, padding_mask)
            # make sure to normalize the output
            global_embs = F.layer_norm(global_embs, global_embs.shape[-1:])
        self._endt()
        self.debug_emb(global_embs, "8. global embeddings", modalities)

        # STUDENT
        self._startt("context_encoder")
        ctx_embs = self.context_encoder(
            ctx_tokens, pos_embs=ctx_pos, padding_mask=ctx_pad_mask
        )
        self._endt()
        self.debug_emb(
            ctx_embs, "9. context embeddings", apply_mask(modalities, ctx_mask)
        )

        # ---------------------
        # Predicting latent targets
        self._startt("predictor")
        tgt_predictions = []
        tgt_truth = []
        tgt_pad_masks = []
        assert M > 0, "No targets found"
        for i in range(M):
            tgt_pos = apply_mask(pos_mod_embs, tgt_masks[i])
            tgt_pad_mask = apply_mask(padding_mask, tgt_masks[i])
            tgt_predictions.append(
                self.predictor(
                    ctx_embs,
                    context_pos=ctx_pos,
                    target_pos=tgt_pos,
                    context_padding_mask=ctx_pad_mask,
                    target_padding_mask=tgt_pad_mask,
                )
            )
            tgt_truth.append(apply_mask(global_embs, tgt_masks[i]))
            tgt_pad_masks.append(tgt_pad_mask)
        self._endt()
        self.debug_emb(
            tgt_predictions[0],
            "10. target prediction",
            apply_mask(modalities, tgt_masks[0]),
        )
        self.debug_emb(
            tgt_predictions[0] - tgt_truth[0],
            "11. prediction diff",
            apply_mask(modalities, tgt_masks[0]),
        )

        # LOSS AND REGULARIZATION
        self._startt("loss")
        jepa_loss = torch.tensor(0.0, device=self.device)
        var_loss_pred = torch.tensor(0.0, device=self.device)
        cov_loss_pred = torch.tensor(0.0, device=self.device)

        M = len(tgt_predictions)
        for i in range(M):
            tl1loss = vectorised_masked_smooth_l1_loss(
                tgt_predictions[i],
                tgt_truth[i],
                tgt_pad_masks[i],
                beta=self.hparams.smooth_l1_beta,
            )
            if tl1loss.isfinite().all():
                jepa_loss += tl1loss
            else:
                log.warning(f"Smooth L1 loss is not finite {i}/{M}")
            varl, covl = vectorised_masked_vicreg_loss(
                tgt_predictions[i], tgt_pad_masks[i]
            )
            if varl.isfinite().all():
                var_loss_pred += varl
            else:
                log.warning(f"Var loss is not finite {i}/{M}")
            if covl.isfinite().all():
                cov_loss_pred += covl
            else:
                log.warning(f"Cov loss is not finite {i}/{M}")

        var_loss_ctx, cov_loss_ctx = vectorised_masked_vicreg_loss(
            ctx_embs, ctx_pad_mask
        )
        if not var_loss_ctx.isfinite().all():
            log.warning("Var loss ctx is not finite")
            var_loss_ctx = torch.tensor(0.0, device=self.device)
        if not cov_loss_ctx.isfinite().all():
            log.warning("Cov loss ctx is not finite")
            cov_loss_ctx = torch.tensor(0.0, device=self.device)

        var_loss_pred *= self.hparams.vicreg_beta
        cov_loss_pred *= self.hparams.vicreg_beta
        var_loss_ctx *= self.hparams.vicreg_beta
        cov_loss_ctx *= self.hparams.vicreg_beta

        loss = jepa_loss + var_loss_pred + cov_loss_pred + var_loss_ctx + cov_loss_ctx

        loss_metrics = {
            "loss": loss,
            "loss_jepa": jepa_loss,
            "loss_var_pred": var_loss_pred,
            "loss_cov_pred": cov_loss_pred,
            "loss_var_ctx": var_loss_ctx,
            "loss_cov_ctx": cov_loss_ctx,
        }
        assert loss.isfinite().all(), "Loss is not finite"
        assert jepa_loss.item() > 1e-5, "Total un-escapable collapse."
        self._endt()

        # LOGGING AND METRICS
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        if len(tgt_predictions) == 0:
            log.warning(f"{type(mask_strat).__name__} did not return any targets")
        res = {
            "tgts": M,
            "tgt_tokens": sum([tgt.sum().item() for tgt in tgt_masks]),
            "global_var": global_embs.var(dim=0).sum(),
            "context_var": ctx_embs.var(dim=0).sum(),
            "gpu_mem_usage_MB": mem_used_MB,
            "global_step": self.trainer.global_step,
        }
        res.update(loss_metrics)

        self._endt("model_step")
        timers = {
            **prefix_keys("ms_", self.timers),
            **prefix_keys("ms_tokenizer_", self.tokenizer.get_metrics()),
        }
        self.timers = {}

        if return_embs:
            return (
                res,
                timers,
                {
                    "ctx": ctx_embs,
                    "ctx_mask": ctx_mask,
                    "tgt_masks": tgt_masks,
                    "tgt": tgt_predictions,
                    "global": global_embs,
                    "modalities": modalities,
                },
            )
        else:
            return res, timers

    def training_step(self, batch, batch_idx: int):
        metrics, timers = self._model_step(batch)
        self.log_dict(
            prefix_keys("train/", metrics), on_step=True, on_epoch=True, prog_bar=True
        )
        self.log_dict(
            prefix_keys("train-timers/", timers),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return metrics["loss"]

    def validation_step(self, batch, batch_idx: int) -> None:
        name = "16_10576_25408"
        index = (
            -1 if name not in batch.names() else batch.names().index("16_10576_25408")
        )
        metrics, timers, embs = self._model_step(
            batch, debug=(index != -1), debug_i=index, return_embs=True
        )
        self.log_dict(
            prefix_keys("val/", metrics), on_step=True, on_epoch=True, prog_bar=True
        )
        self.log_dict(
            prefix_keys("val-timers/", timers),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self._log_analysis(batch, embs)

    def test_step(self, batch, batch_idx: int) -> None:
        metrics, timers = self._model_step(batch)
        self.log_dict(prefix_keys("test/", metrics), on_epoch=True)

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.model_init_time
        finished_epochs = self.trainer.current_epoch + 1
        self.log("train/epoch_time", epoch_time, on_epoch=True)
        print(
            f"Finished epoch {finished_epochs}/{self.trainer.max_epochs} in {epoch_time / 60:.2f}m. Total time: {total_time / 3600:.2f}h, avg epoch time: {(total_time / finished_epochs) / 60:.2f}m, estimated time left: {(total_time / finished_epochs * (self.trainer.max_epochs - finished_epochs)) / 3600:.2f}h"
        )

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("trainer/lr", lr, on_epoch=True, prog_bar=True)
        # adjust every step
        self.adjust_momentum()
        self.adjust_wd()
        # self.adjust_target_sim()

    # def adjust_target_sim(self):
    #     init, end = self.target_sims
    #     self.target_sim = cosine_decay_schedule(init, end, self._step_progress())
    #     self.log("opt/target_sim", self.target_sim, on_epoch=True, prog_bar=True)

    def adjust_momentum(self):
        self.target_encoder.update()
        self.log(
            "opt/ema_decay(%)",
            self.target_encoder.get_current_decay() * 100,
            on_epoch=True,
            prog_bar=True,
        )

    def adjust_wd(self):
        init, end = self.wds
        self.wd = init + (end - init) * self._step_progress()
        self.log("opt/weight_decay", self.wd, on_epoch=True, prog_bar=True)
        for g in self.optimizers().param_groups:
            g["weight_decay"] = self.wd

    def _step_progress(self):
        return self.trainer.global_step / float(self.trainer.estimated_stepping_batches)

    def _epoch_progress(self):
        return self.trainer.current_epoch / float(self.trainer.max_epochs)

    def configure_optimizers(self):
        total_steps = self.trainer.estimated_stepping_batches

        base_lr, min_lr = self.lrs
        base_wd, max_wd = self.wds
        beta1, beta2 = self.adam_betas
        warmup_steps = int(total_steps * self.hparams.warmup_frac)

        params: List[Dict] = [
            {
                "params": list(filter(lambda p: p.requires_grad, self.parameters())),
                "lr": base_lr,
            }
        ]
        opt = torch.optim.AdamW(
            params, lr=base_lr, weight_decay=base_wd, betas=(beta1, beta2)
        )

        warmup = LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        annealing = CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps, eta_min=min_lr
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

    def _pos_emb(self, positions):
        pos_embs = self.position_encoder(positions)
        return pos_embs

    def _mod_emb(self, modalities):
        mod_embs = self.modality_encoder(
            F.one_hot(modalities.long(), num_classes=len(Modality)).float()
        )
        return mod_embs

    @staticmethod
    def _padding_mask(modalities: torch.Tensor) -> torch.Tensor:
        return (modalities == 0).bool()

    def _select_masking_strategy(self):
        chances = self.hparams.masking_strategy_chances
        strats = self.hparams.masking_strategies
        while True:
            for k, w in chances.items():
                if w > torch.rand(1).item():
                    return strats[k]

    def _startt(self, name):
        self.last_timer = name
        self.timers[name] = time.time()

    def _endt(self, name=None):
        if name is None:
            name = self.last_timer
        self.timers[name] = (time.time() - self.timers[name]) * 1000

    def _log_analysis(self, batch, embs):
        if not self.hparams.log_images:
            return
        tb = self.get_tb_logger()
        if tb is None:
            log.warning("No tensorboard logger found")
        mods = embs["modalities"]

        # choose good tiles from huge/val to visualize
        tile1 = "16_10576_25408"
        if tile1 in batch.names():
            dim = self.hparams.token_dim
            step = self.trainer.global_step
            if self.trainer.current_epoch % self.hparams.log_image_interval == 0:
                visualize_representation_space(
                    tb,
                    mods.reshape(-1),
                    embs["global"].reshape(-1, dim),
                    "all_tokens_batch",
                    step,
                )
                visualize_representation_space(
                    tb,
                    apply_mask(mods, embs["ctx_mask"]).reshape(-1),
                    embs["ctx"].reshape(-1, dim),
                    "all_ctx_batch",
                    step,
                )
            # visualize_representation_space(tb,
            #                                torch.cat((
            #                                    torch.ones_like(apply_mask(mods, embs['tgt_masks'][0]).reshape(-1)),
            #                                    torch.ones_like(apply_mask(mods, embs['tgt_masks'][0]).reshape(-1))*2,
            #                                ), dim=0),
            #                                torch.cat((
            #                                    apply_mask(embs['global'], embs['tgt_masks'][0]).reshape(-1, dim),
            #                                    apply_mask(embs['tgt'][0], embs['tgt_masks'][0]).reshape(-1, dim),
            #                                ), dim=0),
            #                                "target vs global", step)
            if self.trainer.current_epoch == 0:
                tb.add_images(
                    f"Images/tiles:{batch.names()}",
                    batch.original_imgs,
                    self.trainer.global_step,
                )

    def get_tb_logger(self):
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                return logger.experiment
        return None

    def debug_emb(
        self, tokens: torch.Tensor | Callable, name, mods=None, tile_names=None
    ):
        if not self.hparams.log_images:
            return
        if self.trainer.current_epoch % self.hparams.log_image_interval != 0:
            return
        if not self.debug_emb_bool:
            return
        if type(tokens) is not torch.Tensor:
            tokens = tokens()
        data = tokens[self.debug_emb_i]
        if mods is not None:
            data = data[mods[self.debug_emb_i] != 0, :]
        if tile_names is not None:
            tile_name = tile_names[self.debug_emb_i]
            # log.info(tile_name)
        log_emb_image(self.get_tb_logger(), data, name, self.trainer.global_step)

    #
    # if __name__ == "__main__":
    #     _ = JEPA()
