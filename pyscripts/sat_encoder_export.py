import os
import time

from rootutils import rootutils
import torch

from src.modules.sat_img_encoder import SatImgEncoderVariableRes

root = rootutils.setup_root(
    os.path.abspath(""), indicator=".project-root", pythonpath=True
)

from torch.export import export

images = torch.randn(10, 3, 224, 224)
res = torch.tensor([10.0])
print(res)

dynamic_shapes = {"images": {0: torch.export.Dim("batch")}, "input_res": {0: 1}}

encoder = SatImgEncoderVariableRes(freeze=True)

start = time.time()
encoder(images, res)
print(f"Inference took: {time.time() - start:.2f}s")

start = time.time()
exported = export(encoder, args=(images, res), dynamic_shapes=dynamic_shapes)
print(f"Export took: {time.time() - start:.2f}s")

start = time.time()
exported.module()(images, res)
print(f"Exported inference took: {time.time() - start:.2f}s")

torch.export.save(exported, "src/models/pretrained/scalemae/exported.pt2")

start = time.time()
import warnings

warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*weights_only=False.*"
)
imported = torch.export.load("src/models/pretrained/scalemae/exported.pt2")
imported = imported.module()
print(f"Import took: {time.time() - start:.2f}s")

start = time.time()
imported(images, res)
print(f"Exported inference took: {time.time() - start:.2f}s")

from src.modules.sat_img_encoder_fast import SatImgEncoder

res = 300 / 224

encoder = SatImgEncoder("src/models/pretrained/scalemae/exported.pt2", freeze=True)

start = time.time()
encoder.forward(images, res)
print(f"Exported inference took: {time.time() - start:.2f}s")

start = time.time()
encoder.forward(torch.cat((images, images), dim=0), res)
print(f"Exported inference took: {time.time() - start:.2f}s")
