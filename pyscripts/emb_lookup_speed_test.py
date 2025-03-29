import time
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components.tiles import TileBatch
from tqdm import tqdm
from src.data.tiles_datamodule import TilesDataModule
from src.modules.embedding_lookup import EmbeddingLookup
from src.modules.vision_backbones import ViTB16


# profile_runner.py


def test():
    batch_size = 64
    group_size = 4
    img_data = TilesDataModule(
        "data/tiles/medium/tasks/pretraining",
        batch_size=batch_size,
        group_size=group_size,
        drop_last=False,
    )
    img_data.setup()
    img_loader = img_data.train_dataloader()
    length = len(img_loader.dataset)
    start = time.time()
    print("Pass with images")
    for _ in tqdm(img_loader):
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}\n")

    no_img_data = TilesDataModule(
        "data/tiles/medium/tasks/pretraining",
        load_images=False,
        batch_size=batch_size,
        group_size=group_size,
        drop_last=False,
    )
    no_img_data.setup()
    no_img_loader = no_img_data.train_dataloader()
    print("drop_last", no_img_loader.loader.drop_last)
    start = time.time()
    print("Pass without images")
    for _ in tqdm(no_img_loader):
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}\n")

    model = (
        EmbeddingLookup("data/embeddings/vitb16/pretraining_medium")
        .eval()
        .requires_grad_(False)
    )
    start = time.time()
    print("Pass with emb lookup")
    for batch in tqdm(img_loader):
        batch: TileBatch = batch
        model(batch)
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}\n")

    model = (
        EmbeddingLookup("data/embeddings/vitb16/pretraining_medium", cls_only=True)
        .eval()
        .requires_grad_(False)
    )
    start = time.time()
    print("Pass with emb lookup (CLS ONLY)")
    for batch in tqdm(img_loader):
        batch: TileBatch = batch
        model(batch)
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}\n")

    device = torch.device("cuda")
    model = ViTB16().cuda().eval().requires_grad_(False)
    start = time.time()
    print("Pass with vitb16")
    for batch in tqdm(img_loader):
        batch: TileBatch = batch
        model(batch.SAT_imgs.to(device))
        pass
    print(f"Time: {time.time() - start}s")
    print(f"tiles/s: {length / (time.time() - start)}")


if __name__ == "__main__":
    test()
