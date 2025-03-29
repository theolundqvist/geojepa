import pickle
import time
from os import makedirs


from src.data.components.tile_dataset import TileDataset
from src.data.components.tiles import collate_tiles
from src.modules.sat_img_encoder import SatImgEncoder
from src.modules.vision_backbones import ResNet50, ViTB16, EfficientNet

start_import = time.time()
import torch

print("Time taken to import torch: ", time.time() - start_import)

device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")

path = "data/tiles/tiny"
dataset = TileDataset(path, "test")
tiles = collate_tiles([dataset[i] for i in range(len(dataset))])
tiles = tiles.to(device)

print("RUNNING SCALEMAE")
encoder = SatImgEncoder(freeze=True).to(device)

start = time.time()
cls, features = encoder(tiles.SAT_imgs)
print("Inference: ", time.time() - start, " s")
print("cls", cls.shape)
print("features", features.shape)
makedirs("pyscripts/sat/scalemae", exist_ok=True)
pickle.dump(
    {"tiles": tiles.to("cpu"), "cls": cls.to("cpu"), "features": features.cpu()},
    open("pyscripts/sat/scalemae/embeddings.pkl", "wb"),
)

print("RUNNING RESNET50")
encoder = ResNet50(freeze=True).to(device)
tiles.to(device)
start = time.time()
features = encoder(tiles.SAT_imgs)
print("Inference: ", time.time() - start, " s")
print("features", features.shape)
makedirs("pyscripts/sat/resnet", exist_ok=True)
pickle.dump(
    {"tiles": tiles.to("cpu"), "features": features.cpu()},
    open("pyscripts/sat/resnet/embeddings.pkl", "wb"),
)

print("RUNNING EfficientNet")
encoder = EfficientNet(freeze=True).to(device)
tiles.to(device)
start = time.time()
features = encoder(tiles.SAT_imgs)
print("Inference: ", time.time() - start, " s")
print("features", features.shape)
makedirs("pyscripts/sat/efficientnet", exist_ok=True)
pickle.dump(
    {"tiles": tiles.to("cpu"), "features": features.cpu()},
    open("pyscripts/sat/efficientnet/embeddings.pkl", "wb"),
)

print("RUNNING ViTB16")
encoder = ViTB16(freeze=True).to(device)
tiles.to(device)
start = time.time()
features = encoder(tiles.SAT_imgs)
print("Inference: ", time.time() - start, " s")
print("features", features.shape)
makedirs("pyscripts/sat/vitb16", exist_ok=True)
pickle.dump(
    {"tiles": tiles.to("cpu"), "features": features.cpu()},
    open("pyscripts/sat/vitb16/embeddings.pkl", "wb"),
)
