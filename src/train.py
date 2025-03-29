import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="lightning.pytorch.trainer.connectors.data_connector",
)
warnings.filterwarnings("ignore", message=".*W1208.*")  # compiler non-static warning
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

torch.set_float32_matmul_precision("medium")

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import lightning_utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.lightning_utils.utils import get_choice, get_choices
from src.lightning_utils.logging_utils import prefix_keys
from src.lightning_utils.resolvers import sweep_name
from src.lightning_utils.instantiators import instantiate_model

from src.lightning_utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def unwrap_tensors(x: dict):
    for k, v in x.items():
        if type(v) is torch.Tensor:
            x[k] = v.item()
    return x


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate_model(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # print available GPU name
    if torch.cuda.is_available():
        log.info(f"GPU FOUND: {torch.cuda.get_device_name(torch.device('cuda'))}")

    start = time.time()
    if cfg.get("train"):
        log.info(get_choices())
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    training_time = time.time() - start

    model.setup("test")

    train_metrics = trainer.callback_metrics

    test_metrics = {}
    small_test_metrics = {}
    data_size = cfg.data.size
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        try:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        except Exception as e:
            log.warning(e)
        test_metrics = trainer.callback_metrics
        log.info(f"Best ckpt path: {ckpt_path}")
        if cfg.data.size != "small":
            log.info("Starting testing on SMALL!")
            log.info(f"Instantiating datamodule <{cfg.data._target_}>")
            data = cfg.data
            OmegaConf.set_struct(data, True)
            with open_dict(data):
                data.size = "small"
            try:
                datamodule: LightningDataModule = hydra.utils.instantiate(data)
                loggers = trainer.loggers
                trainer.loggers = []
                try:
                    trainer.test(
                        model=model, datamodule=datamodule, ckpt_path=ckpt_path
                    )
                except Exception as e:
                    log.warning(e)
                small_test_metrics = trainer.callback_metrics
                trainer.loggers = loggers
                for l in trainer.loggers:
                    l.log_metrics(
                        prefix_keys("small/", unwrap_tensors(small_test_metrics)),
                        step=trainer.global_step,
                    )
                    l.finalize("success")
            except:
                pass

    train_loss = train_metrics.get("train/loss", None)
    if train_loss is not None and train_loss.isfinite():
        lr = None
        lr_min = None
        wd = None
        try:
            lr = cfg.model.optimizer.lr
            lr_min = cfg.model.lr_min
            wd = cfg.model.optimizer.weight_decay
        except:
            try:
                lr, lr_min = cfg.model.lr[0]
                wd = cfg.model.wd[0]
            except:
                pass
        with open(Path(cfg.paths.output_dir) / "results.json", "w") as f:
            d = {
                "model": get_choice("model"),
                "dataset": get_choice("data"),
                "cheat": cfg.data.cheat,
                "training_time": training_time,
                data_size: unwrap_tensors(test_metrics),
                "small": unwrap_tensors(small_test_metrics),
                "date": datetime.today().strftime("%Y-%m-%d"),
                "time": datetime.today().strftime("%H:%M:%S"),
                "lr": lr,
                "lr_min": lr_min,
                "weight_decay": wd,
            }
            json.dump(d, f, indent=4)

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # train the model
    metric_dict, _ = train(cfg)

    print(metric_dict)
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    OmegaConf.register_new_resolver("sweep_name", sweep_name)
    main()
