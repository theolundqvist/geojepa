import os

import torch
from torch import Tensor, nn
import warnings

from src import log

model_path = os.path.join(os.path.dirname(__file__), "../models/pretrained/scalemae")


class SatImgEncoder(nn.Module):
    def __init__(self, model_path: str, freeze=True):
        super().__init__()
        self.model_path = model_path
        self.freeze = freeze
        self.model = load_model(freeze, model_path, torch.device("cpu"))

    def to(self, device: torch.device):
        self.model = load_model(self.freeze, self.model_path, device)
        return self

    def forward(self, images: Tensor, input_res=10.0) -> (Tensor, Tensor):
        """
        @param images: Tensor of shape (B, C, H, W) -> (B, 3, 224, 224)
        @param input_res: meters per pixel
        @return: cls (1024), features (196x1024)
        """
        res = torch.tensor([input_res]).to(images)
        for p in self.parameters():
            if p.device != images.device:
                log.info(f"parameter has device: {p.device}")
        cls, features = self.model(images, res)
        return cls, features

    def train(self, *args, **kwargs):
        pass


def load_model(freeze: bool, path: str, device: torch.device):
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message=".*weights_only=False.*"
    )
    imported = torch.export.load(path)

    for node in imported.graph.nodes:
        if "device" in node.kwargs:
            kwargs = node.kwargs.copy()
            kwargs["device"] = device
            node.kwargs = kwargs

    for k, v in imported.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            imported._state_dict[k] = torch.nn.Parameter(v.to(device))
        else:
            imported._state_dict[k] = v.to(device)
    model = imported.module()
    model.to(device)
    if freeze:
        # model.eval() not yet supported for exported models, we exported it in eval mode so should be fine.
        for param in model.parameters():
            param.requires_grad = False
    return model
