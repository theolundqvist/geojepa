# Modified from https://github.com/Ayumu-J-S/Point-JEPA/blob/main/pointjepa/modules/EMA.py
# Who in turn modified from
# https://github.com/lucidrains/ema-pytorch/blob/638f1526b1f952b2597d52dd5af57f1a02669804/ema_pytorch/ema_pytorch.py

import copy
from typing import Set
import torch
from torch import nn

from src import log


class LinearEMA(nn.Module):
    """Implements exponential moving average shadowing for your model.

    Utilizes a linear decay rate schedule.

    Args:
        model (nn.Module): The model to shadow using EMA.
        model_influence_start (float):
        model_influence_end (float):
    """

    def __init__(
        self,
        model: nn.Module,
        momentum_start: float = 0.99,
        momentum_end: float = 1.0,
        # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        ema_model=None,  # will be copied from model if not provided
        model_influence_start=None,
        model_influence_end=None,
        update_after_step: int = 0,
        update_every: int = 1,
        param_or_buffer_names_no_ema: Set = set(),
        ignore_names: Set = set(),
    ):
        if model_influence_start is not None:
            log.warning(
                "model_influence_start is deprecated, please use tau_start instead"
            )
            momentum_start = 1 - model_influence_start
        if model_influence_end is not None:
            log.warning("model_influence_end is deprecated, please use tau_end instead")
            momentum_end = 1 - model_influence_end
        super().__init__()
        self.online_model = model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                log.info(
                    "Your model was not copyable. Please make sure you are not using any LazyLinear"
                )
                exit()

        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.tau_min = momentum_start
        self.tau_max = momentum_end
        self.tau_steps = -1

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.ignore_names = ignore_names

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def set_training_steps(self, steps):
        self.tau_steps = steps

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        assert self.tau_steps > 0, (
            "tau_steps must be set before calling get_current_decay, use set_training_steps"
        )
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)

        if epoch < self.tau_steps:
            value = (
                self.tau_min + (self.tau_max - self.tau_min) * epoch / self.tau_steps
            )
        else:
            value = self.tau_max

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.tau_min, max_value=self.tau_max)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if name in self.ignore_names:
                continue

            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if name in self.ignore_names:
                continue

            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass


def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value
