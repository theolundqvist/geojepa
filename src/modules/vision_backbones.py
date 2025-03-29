from typing import Tuple

import torch
import torch.nn as nn
from einops import repeat
from torch import Tensor
from torchvision import models
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    EfficientNet_B0_Weights,
)

from src.data.components.tiles import TileBatch
from src import log


class ImageSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: TileBatch):
        assert type(x) == TileBatch
        return x.SAT_imgs


class BlindTileEmbedder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: TileBatch):
        return repeat(self.bias, "t -> b t", b=x.size)


def log_var(title, x: Tensor):
    log.info(f"[{title}] total batch variance: {x.std(dim=0).sum().item()}\n\n")


class ViTB16(nn.Module):
    """
    Vision Transformer (ViT-B_16) Backbone.
    This module initializes a pre-trained ViT-B_16 model, removes its classification head,
    and outputs a 768-dimensional feature vector representing the input image.
    """

    def __init__(self, *args, **kwargs):
        super(ViTB16, self).__init__()
        # Load the pre-trained ViT-B_16 model
        self.vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # self.preprocessing = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, 224, 224).
        Returns:
            torch.Tensor: Output feature tuple of shape ((B, 768), (B, 196, 768)).
        """
        # x = self.preprocessing(x)
        # code taken from ViT forward method
        assert type(x) == torch.Tensor, f"ViTB16: Expected torch.Tensor, got {type(x)}"
        assert len(x.shape) == 4, f"ViTB16: Expected 4D tensor, got {x.shape}"
        x = self.vit._process_input(x)
        B = x.shape[0]
        batch_class_token = self.vit.class_token.expand(B, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)

        # Classifier "token" as used by standard language architectures
        cls = x[:, 0]
        patches = x[:, 1:]
        return cls, patches


class EfficientNet(nn.Module):
    """
    EfficientNetB0 Backbone.
    This module initializes a pre-trained EfficientNetB0 model, removes its final classifier,
    and outputs a 1280-dimensional feature vector representing the input image.
    Args:
        pretrained (bool): If True, loads pretrained weights on ImageNet.
        freeze (bool): If True, freezes the backbone parameters to prevent training.
    """

    def __init__(self):
        super(EfficientNet, self).__init__()
        # Load the pre-trained EfficientNetB0 model
        efficientnet = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # Remove the classifier and add Flatten to obtain a 1D feature vector
        modules = list(efficientnet.children())[:-1]  # Exclude the 'classifier' layer
        self.backbone = nn.Sequential(*modules, nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Output feature tensor of shape (batch_size, 1280).
        """
        features = self.backbone(x)
        return features


class ResNet50(nn.Module):
    """
    ResNet50 Backbone.
    This module initializes a pre-trained ResNet50 model, removes its final fully connected layer,
    and outputs a 2048-dimensional feature vector representing the input image.
    """

    def __init__(self):
        super(ResNet50, self).__init__()
        # Load the pre-trained ResNet50 model
        resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully connected layer and add Flatten to obtain a 1D feature vector
        modules = list(resnet50.children())[:-1]  # Exclude the 'fc' layer
        self.backbone = nn.Sequential(*modules, nn.Flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).
        Returns:
            torch.Tensor: Output feature tensor of shape (batch_size, 2048).
        """
        features = self.backbone(x)
        return features
