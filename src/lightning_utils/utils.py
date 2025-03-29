import glob
import json
import shutil
import warnings
from importlib.util import find_spec
from os import mkdir
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.lightning_utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich srcrary
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @lightning_utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        should_exit = False
        metric_dict, object_dict = {}, {}
        try:
            if cfg.get("record_memory_history", False):
                torch.cuda.memory._record_memory_history()
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                choices = HydraConfig.get().runtime.choices

                def clean(str):
                    if str is None:
                        log.warning(f"None value found in choices! {choices}")
                        return "none"
                    return str.replace("[", "").replace("]", "").split(" ")[-1]

                old_output_dir = cfg.paths.output_dir
                if cfg.get("ckpt"):
                    cfg.paths.output_dir += f"_ckpt_{clean(choices['data'])}"
                else:
                    cfg.paths.output_dir += (
                        f"_{clean(choices['model'])}_{clean(choices['data'])}"
                    )
                if cfg.get("data") and cfg.get("data").get("cheat"):
                    cfg.paths.output_dir += "_cheat"
            mkdir(cfg.paths.output_dir)
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("Error", ex)

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure

            # DISABLED RAISING EXCEPTION TO RUN ALL CONFIGS IN MULTIRUN ANYWAYS
            # raise ex
            if cfg.get("fast_fail", False):
                log.info("Fast fail is enabled! Skipping any other runs...")
                log.error(ex)
                should_exit = True
            else:
                log.info("Fast fail is disabled! Continuing with other configs...")

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")
            files = glob.glob(f"{old_output_dir}/*")
            for f in files:
                shutil.move(f, cfg.paths.output_dir)
            shutil.move(f"{old_output_dir}/.hydra", f"{cfg.paths.output_dir}/hydra")
            shutil.rmtree(old_output_dir)
            if cfg.get("record_memory_history", False):
                torch.cuda.memory._dump_snapshot(
                    cfg.paths.output_dir + "/memory_history.pkl"
                )

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        if should_exit:
            raise Exception("Fast fail is enabled! Skipping any other runs...")

        return metric_dict, object_dict

    return wrap


def get_choice(s: str) -> str:
    choices = HydraConfig.get().runtime.choices
    return str(choices[s]).replace("[", "").replace("]", "").split(" ")[-1]


def get_choices() -> str:
    return json.dumps(
        {k: v for k, v in HydraConfig.get().runtime.choices.items()}, indent=4
    )


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        log.info(metric_dict)
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    if isinstance(metric_value, torch.Tensor):
        metric_value = metric_value.item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    if not torch.tensor(metric_value).isfinite():
        return 1e9
        # raise Exception(
        #     "Got invalid metric value! "
        #     f"<metric_name={metric_name}, metric_value={metric_value}>")
    return metric_value
