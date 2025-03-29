import os

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch import Tensor, nn

model_path = os.path.join(os.path.dirname(__file__), "../models/pretrained/scalemae")


# Instantiate the model with pre-trained FMOW_RGB weights
class ScaleMAE_baseline(nn.Module, PyTorchModelHubMixin):
    def __init__(self, global_pool=False, cls_token_flag=True):
        super().__init__()
        from ..models.pretrained.scalemae.model import get_ScaleMAE_model

        self.model = get_ScaleMAE_model(
            global_pool=global_pool, cls_token=cls_token_flag
        )

    def forward(self, x, input_res):
        res = input_res.to(x.device)
        # start = time.time()
        x = self.model.forward(x, input_res=res)
        # log.info(f"Forward pass took: {time.time() - start:.2f}s")
        return x


class SatImgEncoder(nn.Module):
    def __init__(self, res=300 / 224):
        super().__init__()
        self.res = res
        self.model = SatImgEncoderVariableRes()

    def forward(self, images: Tensor) -> (Tensor, Tensor):
        return self.model.forward(images, torch.tensor([self.res]))


class SatImgEncoderVariableRes(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "MVRL/scalemae-vitlarge-800"
        hf_hub_download(
            model_name, "model.py", local_dir=model_path, local_files_only=False
        )
        self.model = ScaleMAE_baseline.from_pretrained(
            model_name, local_files_only=False
        )

    def forward(self, images: Tensor, input_res: Tensor) -> (Tensor, Tensor):
        """
        @param images: Tensor of shape (B, C, H, W) -> (B, 3, 224, 224)
        @param input_res: meters per pixel tensor([10.0])
        @return: cls (1024), features (196x1024)
        """
        y = self.model.forward(images, input_res)
        return y[:, 0], y[:, 1:]
