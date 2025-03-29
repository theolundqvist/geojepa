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
from src.modules.callbacks import UnfreezeParamGroupsCallback
from src.modules.losses import (
    vectorised_masked_vicreg_loss,
    vectorised_masked_smooth_l1_loss,
)
from src.modules.masks import MaskingStrategy, apply_mask
from src.modules.mlp import MLP
from src.modules.tokenizer import Modality
from src.modules.vision_backbones import ViTB16
from src.utils.logging import (
    visualize_representation_space,
    log_emb_image,
)
from src.utils.sort_utils import restore_tensor_order

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class JEPA(pl.LightningModule):
    """
    Generic JEPA encoder training,
    check ex config/models/geojepa.yaml for example.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        optimizer: torch.optim.Optimizer,
        ema_strategy: nn.Module,
        masking_strategies: Dict[str, MaskingStrategy],
        masking_strategy_chances: Dict[str, float],
        tokenizer: nn.Module,
        position_encoder: nn.Module = None,
        loss_beta: float = 2,
        token_dim: int = 1024,  # keep this for hydra use in encoder etc
        warmup_fraction: float = 0.06,
        lr_min: float = 1e-6,
        compile: bool = False,
        svm_validation: Dict[str, pl.LightningDataModule] = None,
        train_encoders_lr_modifier: Dict[str, float] = {},
        weight_decay_start: float = 0.04,
        weight_decay_end: float = 0.4,
        debug: bool = False,  # ignored
        vicreg_beta: float = 0.008,
        unfreeze_after: float = None,
        use_concat_pos_and_mod: bool = False,
        use_augmentations: bool = False,
    ):
        super().__init__()
        # save hyperparameters to self.hparams, ignore nn.Modules since they are already checkpointed
        self.unfreeze_param_groups = []
        self.save_hyperparameters(
            ignore=["encoder", "predictor", "position_encoder", "tokenizer"]
        )

        # self.loss_func = nn.SmoothL1Loss(beta=loss_beta)
        self.loss_func = nn.SmoothL1Loss(beta=2.0)
        self.collapse_loss = None

        # self.target_encoder = teacher_encoder  # created in setup
        self.tokenizer = tokenizer

        self.target_encoder: nn.Module = ema_strategy(encoder)
        self.context_encoder = encoder

        self.predictor = predictor
        self.layer_norm = nn.LayerNorm(token_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, token_dim))
        self.cls_modality = 4

        # ----------------------------------------
        # Position and modality encoders
        assert len(Modality) == 4
        if use_concat_pos_and_mod:
            self.position_encoder = MLP(8, 256, token_dim // 2, drop=0.1)
            self.modality_encoder = MLP(
                len(Modality) + 1, 256, token_dim // 2, drop=0.1
            )
        else:
            self.modality_encoder = MLP(len(Modality) + 1, 256, token_dim, drop=0.1)
            self.position_encoder = position_encoder

        # ----------------------------------------
        # Unfreeze tokenizer modules
        self.timers = {}
        self.last_timer = None
        if unfreeze_after is not None:
            self.unfreeze_callback = UnfreezeParamGroupsCallback(
                fraction=unfreeze_after
            )

    def setup(self, stage=None):
        if stage == "fit":
            self.target_encoder.set_training_steps(
                self.trainer.estimated_stepping_batches
            )
        # print("SETUP", stage)
        if self.hparams.compile and stage == "fit":
            torch._dynamo.config.cache_size_limit = 64
            # torch.compile(self._model_step)
            self.predictor = torch.compile(self.predictor, fullgraph=True)
            self.context_encoder = torch.compile(self.context_encoder, fullgraph=True)
            self.target_encoder.ema_model = torch.compile(
                self.target_encoder.ema_model, fullgraph=True
            )
            self.position_encoder = torch.compile(self.position_encoder, fullgraph=True)
            self.modality_encoder = torch.compile(self.modality_encoder, fullgraph=True)
            self.loss_func = torch.compile(self.loss_func, fullgraph=True)
            if (
                self.tokenizer.tokenize_images
                and type(self.tokenizer.img_encoder) is ViTB16
            ):
                print("Compiling ViTB16")
                self.tokenizer.img_encoder = torch.compile(
                    self.tokenizer.img_encoder, fullgraph=True
                )
            # if self.tokenizer.tokenize_geometry:
            #     print("Compiling geometry encoder")
            #     self.tokenizer.geometry_encoder = torch.compile(self.tokenizer.geometry_encoder, fullgraph=True)

    def forward(self, batch: TileBatch):
        tokens, positions, modalities, sort_indices = self.tokenizer(batch)
        # if tokens is None:
        #     return torch.zeros((batch.size, self.hparams.token_dim)), torch.zeros(
        #         (batch.size, 1, self.hparams.token_dim))
        # tokens, positions, modalities = self._add_cls_tokens(tokens, positions, modalities)
        pos_embs = self._pos_emb(positions)
        mod_embs = self._mod_emb(modalities)
        if self.hparams.use_concat_pos_and_mod:
            pos_mod_embs = torch.cat([pos_embs, mod_embs], dim=-1)
        else:
            pos_mod_embs = pos_embs + mod_embs
        # pos_mod_embs = self.layer_norm(pos_mod_embs)

        padding_mask = self._padding_mask(modalities)

        # print("target_encoder_weights", get_weight_statistics(self.target_encoder.ema_model))
        # print("context_encoder_weights", get_weight_statistics(self.context_encoder))
        encoded = self.target_encoder(tokens, pos_mod_embs, padding_mask=padding_mask)

        cls = torch.cat(
            [torch.max(encoded, dim=1).values, torch.mean(encoded, dim=1)], dim=1
        )
        feats = restore_tensor_order(encoded, sort_indices)
        return cls, feats, modalities, sort_indices

    def _add_cls_tokens(self, tokens, positions, modalities):
        B, T, C = tokens.shape
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        cls_pos = (
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], device=self.device)
            .repeat(B)
            .view(-1, 1, 8)
        )
        positions = torch.cat([cls_pos, positions], dim=1)

        cls_mod = (
            torch.ones(B, 1, dtype=torch.int, device=self.device) * self.cls_modality
        )
        modalities = torch.cat([cls_mod, modalities], dim=1)
        return tokens, positions, modalities

    def _pos_emb(self, positions):
        pos_embs = self.position_encoder(positions)
        return pos_embs

    def _mod_emb(self, modalities):
        mod_embs = self.modality_encoder(
            F.one_hot(modalities.long(), num_classes=self.cls_modality + 1).float()
        )
        return mod_embs

    def _padding_mask(self, modalities: torch.Tensor) -> torch.Tensor:
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

    def debug_emb(
        self, tokens: torch.Tensor | Callable, name, mods=None, tile_names=None
    ):
        if self.trainer.current_epoch % 3 != 0:
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
            log.info(tile_name)
        log_emb_image(self.get_tb_logger(), data, name, self.trainer.global_step)

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
        tokens, positions, modalities, indices = self.tokenizer(batch)
        self._endt()

        self.debug_emb(
            tokens, "1. after tokenizer", modalities, tile_names=batch.names()
        )

        self.debug_emb(positions, "2. actual positions", modalities)
        # positions += torch.randn(positions.shape,
        #                          device=positions.device) * 5e-4  # add [~-3,~3]*15cm noise to all positions
        # self.debug_emb(positions, "3. actual positions with noise")

        # self._startt("add_cls_tokens")
        # tokens, positions, modalities = self._add_cls_tokens(tokens, positions, modalities)
        # self._endt()
        # self.debug_emb(tokens, "3. tokens with cls token", modalities)
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

        self._startt("pos_emb")
        pos_embs = self._pos_emb(positions)
        mod_embs = self._mod_emb(modalities)
        if self.hparams.use_concat_pos_and_mod:
            pos_mod_embs = torch.cat([pos_embs, mod_embs], dim=-1)
        else:
            pos_mod_embs = pos_embs + mod_embs
        pos_mod_embs = self.layer_norm(pos_mod_embs)
        self._endt()
        self.debug_emb(pos_embs, "6. pos embeddings", modalities)
        self.debug_emb(mod_embs, "6.5. mod embeddings", modalities)

        padding_mask = self._padding_mask(modalities)

        self._startt("apply_ctx_masks")
        ctx_tokens = apply_mask(tokens, ctx_mask)
        ctx_pos = apply_mask(pos_mod_embs, ctx_mask)
        ctx_pad_mask = apply_mask(padding_mask, ctx_mask)
        self._endt()
        self.debug_emb(ctx_tokens, "7. ctx tokens", apply_mask(modalities, ctx_mask))

        # TEACHER
        self._startt("target_encoder")
        with torch.no_grad():
            # self.teacher_encoder.ema_model.save_eval() # what was this supposed to do?
            global_embs = self.target_encoder.ema_model(
                tokens, pos_mod_embs, padding_mask
            )
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

        # # Small chance of giving global class token to predictor to force cls token to be globally useful
        # if 0.1 > torch.rand(1).item():
        #     ctx_embs[:, 0, :] = global_embs[:, 0, :]

        # PREDICTOR
        tgt_predictions = []
        tgt_truth = []
        tgt_pad_masks = []
        self._startt("predictor")
        for i in range(M):
            # TODO TEST: prediction only gets the position of the target to facilitate modality-independent semantic learning
            # tgt_pos = apply_mask(pos_embs, tgt_masks[i])
            tgt_pos = apply_mask(pos_mod_embs, tgt_masks[i])
            tgt_pad_mask = apply_mask(padding_mask, tgt_masks[i])
            tgt_pad_masks.append(tgt_pad_mask)
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
            jepa_loss += vectorised_masked_smooth_l1_loss(
                tgt_predictions[i], tgt_truth[i], tgt_pad_masks[i], beta=1.0
            )
            varl, covl = vectorised_masked_vicreg_loss(
                tgt_predictions[i], tgt_pad_masks[i]
            )
            var_loss_pred += varl
            cov_loss_pred += covl

        var_loss_ctx, cov_loss_ctx = vectorised_masked_vicreg_loss(
            ctx_embs, ctx_pad_mask
        )

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
        self._endt()
        # make_dot(loss, params=dict(self.named_parameters()), show_attrs=True, show_saved=True).render("logs/renders/jepa_loss.gv", format="svg")

        # LOGGING AND METRICS
        free, total = torch.cuda.mem_get_info(self.device)
        mem_used_MB = (total - free) / 1024**2
        if len(tgt_predictions) == 0:
            log.warning(f"{type(mask_strat).__name__} did not return any targets")
        res = {
            "global_var": global_embs.var(dim=0).sum(),
            "context_var": ctx_embs.var(dim=0).sum(),
            "gpu_mem_usage_MB": mem_used_MB,
            "global_step": self.trainer.global_step,
        }
        res.update(loss_metrics)
        if len(tgt_predictions) > 0:
            res["pred_0_var"] = tgt_predictions[0].var(dim=0).sum()
        if len(tgt_predictions) > 1:
            res["pred_1_var"] = tgt_predictions[1].var(dim=0).sum()

        self._endt("model_step")
        timers = {
            **prefix_keys("ms_", self.timers),
            **prefix_keys("ms_tokenizer_", self.tokenizer.get_metrics()),
            "num_tokens": torch.tensor(tokens.size(0), device=self.device),
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

    # def on_train_start(self) -> None:
    #     lr = self.lr_schedulers().get_last_lr()[0]
    #     self.log("trainer/lr", lr, on_epoch=True, prog_bar=True)
    def on_train_epoch_end(self) -> None:
        if self.hparams.unfreeze_after is not None:
            self.unfreeze_callback.on_train_epoch_end(self.trainer, self)

    def training_step(self, batch, batch_idx: int):
        metrics, timers = self._model_step(
            batch, debug=(self.trainer.global_step == 0 and batch_idx == 0)
        )
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

    def _log_analysis(self, batch, embs):
        tb = self.get_tb_logger()
        if tb is None:
            log.warning("No tensorboard logger found")
        mods = embs["modalities"]

        # choose good tiles from huge/val to visualize
        tile1 = "16_10576_25408"
        if tile1 in batch.names():
            dim = self.hparams.token_dim
            step = self.trainer.global_step
            max_tiles = min(32, mods.size(0))
            if self.trainer.current_epoch % 3 == 0:
                visualize_representation_space(
                    tb,
                    mods[:max_tiles].reshape(-1),
                    embs["global"][:max_tiles].reshape(-1, dim),
                    "all_tokens_batch",
                    step,
                )
                visualize_representation_space(
                    tb,
                    apply_mask(mods, embs["ctx_mask"])[:max_tiles].reshape(-1),
                    embs["ctx"][:max_tiles].reshape(-1, dim),
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

    def test_step(self, batch, batch_idx: int) -> None:
        metrics, timers, embs = self._model_step(batch, return_embs=True)
        self.log_dict(prefix_keys("test/", metrics), on_epoch=True)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log("trainer/lr", lr, on_epoch=True, prog_bar=True)
        self.target_encoder.update()
        self.log(
            "opt/ema_decay(%)",
            self.target_encoder.get_current_decay() * 100,
            on_epoch=True,
            prog_bar=True,
        )
        initial_wd = self.hparams.weight_decay_start
        final_wd = self.hparams.weight_decay_end
        total_steps = self.trainer.estimated_stepping_batches
        step = self.trainer.global_step
        new_wd = initial_wd + (final_wd - initial_wd) * (step / float(total_steps))
        self.log("opt/weight_decay", new_wd, on_epoch=True, prog_bar=True)
        for g in self.optimizers().param_groups:
            g["weight_decay"] = new_wd

    def configure_optimizers(self):
        lr = self.hparams.optimizer(params=[nn.Parameter(torch.zeros(1))]).defaults[
            "lr"
        ]
        params: List[Dict] = [
            {
                "params": list(filter(lambda p: p.requires_grad, self.parameters())),
                "lr": lr,
            }
        ]
        for k, v in self.hparams.train_encoders_lr_modifier.items():
            if v == 0:
                continue
            encoder = getattr(self.tokenizer, f"{k}_encoder")
            encoder.requires_grad_(False)
            self.unfreeze_param_groups.append(
                {
                    "name": k,
                    "params": encoder.parameters(),
                    "lr_modifier": v,
                }
            )
        opt = self.hparams.optimizer(params=params)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_fraction)
        warmup = LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        annealing = CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps, eta_min=self.hparams.lr_min
        )
        sched = SequentialLR(
            opt, schedulers=[warmup, annealing], milestones=[warmup_steps]
        )
        # warmup_epochs = max(self.trainer.max_epochs * self.hparams.warmup_fraction, 1)
        # warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        # annealing = CosineAnnealingLR(opt, T_max=self.trainer.max_epochs - warmup_epochs, eta_min=self.hparams.lr_min)
        # sched = SequentialLR(opt, schedulers=[warmup, annealing], milestones=[warmup_epochs])

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


#
# if __name__ == "__main__":
#     _ = JEPA()
