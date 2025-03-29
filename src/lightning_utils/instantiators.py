from pathlib import Path
from typing import List, Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.lightning_utils.utils import get_choice
from src.lightning_utils.aim_logger import AimLogger

from src.lightning_utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    loggers: List[Logger] = []
    logger_cfg = cfg.get("logger")

    choices = HydraConfig.get().runtime.choices
    extra_tags = ["model", "data", "data.cheat", "data.size"]
    tags = cfg.get("tags", [])
    for tag_key in extra_tags:
        if tag_key in choices and choices[tag_key] is not None:
            tags.append(f"[{tag_key}: {get_choice(tag_key)}]")

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger = hydra.utils.instantiate(lg_conf)
            loggers.append(logger)
            if isinstance(logger, AimLogger):
                log.info(tags)
                logger.set_experiment_name(choices.get("experiment", "none"))
                logger.add_tags(tags)
            # if isinstance(logger, TensorBoardLogger):
            #     experiment = choices.get("experiment")
            #     if experiment is None:
            #         experiment = "default"
            #     logger._name = "-".join([experiment, *tags])

    return loggers


def instantiate_model(cfg: DictConfig, require_ckpt=False, overrides: dict = {}) -> Any:
    """
    Load and instantiate models from the configuration, handling sub-models recursively.
    If a sub-model has a 'ckpt' key, it replaces its config with the one from the checkpoint,
    and loads the weights from the checkpoint.

    Args:
        cfg (DictConfig): The model configuration.

    Returns:
        Any: The fully instantiated and loaded top-level model.
    """

    loaded_ckpt = False

    def load_overrides(parent_cfg: DictConfig, path: str) -> dict[str, Any]:
        cfg = parent_cfg[path]
        if isinstance(cfg, DictConfig) and "ckpt" in cfg:
            ckpt_path = Path(cfg["ckpt"])
            log.info(f"Loading model '{path}' from checkpoint: {ckpt_path}")

            # Load the Hydra configuration from the checkpoint directory
            hydra_config_path = ckpt_path / "hydra" / "config.yaml"
            if not hydra_config_path.exists():
                raise FileNotFoundError(
                    f"Hydra config not found at {hydra_config_path}"
                )

            loaded_cfg = OmegaConf.load(hydra_config_path)
            # **Override keys from parent_cfg[path] except 'ckpt'**
            cfg = OmegaConf.to_container(cfg)
            cfg.pop("ckpt")
            cfg = OmegaConf.merge(loaded_cfg.model, cfg)
            OmegaConf.resolve(cfg)
            parent_cfg.update({path: cfg})

            # overrides = load_overrides(model_cfg, current_path=model_cfg)
            model = hydra.utils.instantiate(cfg)

            # Load the checkpoint
            checkpoint_path = ckpt_path / "checkpoints" / "last.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Checkpoint file not found at {checkpoint_path}"
                )

            log.info(f"Loading state_dict from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                # if k.startswith("model."):
                #     k = k[len("model."):]
                if "._orig_mod" in k:
                    k = k.replace("._orig_mod", "")
                if "teacher_encoder" in k:
                    k = k.replace("teacher", "target")
                if "student_encoder" in k:
                    k = k.replace("student", "context")
                if k in new_state_dict:
                    raise ValueError(f"Duplicate key found in state_dict: {k}")
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict, strict=True, assign=True)
            log.info(f"Loaded weights into model: {cfg._target_}")
            loaded_ckpt = True
            return {path: model}
        else:
            overrides = {}
            for key, value in cfg.items():
                if isinstance(value, DictConfig):
                    ov = load_overrides(cfg, path=key)
                    overrides.update(ov)
            return overrides

    # Start instantiation from the top-level model
    args = load_overrides(cfg, path="model")
    if require_ckpt and not loaded_ckpt:
        raise LookupError("Required ckpt but did not load any")
    model = hydra.utils.instantiate(cfg.model, **{**args, **overrides})
    return model
