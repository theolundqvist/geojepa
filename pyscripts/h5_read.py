
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.modules.embedding_lookup import EmbeddingLookup


class FakeTileBatch:
    def __init__(self, name):
        self.name = name

    def names(self):
        return [self.name]


def read_cls_h5(cls_dir_path):
    model = EmbeddingLookup(cls_dir_path, cls_only=True)
    for k, v in model.id_map.items():
        yield k, model(FakeTileBatch(k)).numpy()


def read_cls_h5_tile(cls_dir_path, name):
    model = EmbeddingLookup(cls_dir_path, cls_only=True)
    return model(FakeTileBatch(name)).numpy()


def collate_fn(tiles_list):
    return [tile for tiles in tiles_list for tile in tiles]


def read_feats_h5(feat_dir_path):
    model = EmbeddingLookup(feat_dir_path, cls_only=False)
    for k, v in model.id_map.items():
        cls, features = model(FakeTileBatch(k))
        yield k, features.numpy()


def read_emb_h5(dir_path, shuffle=True):
    """
    Yields name, embedded tokens [T, C],  first token is CLS
    """
    model = EmbeddingLookup(dir_path, cls_only=False)
    map = list(model.id_map.items())
    indices = range(len(map))
    if shuffle:
        indices = torch.randint(0, len(map), (len(map),)).tolist()
    for i in indices:
        tile_name, v = map[i]
        cls, features = model(FakeTileBatch(tile_name))
        emb = torch.cat([cls.unsqueeze(0), features], dim=1)
        yield tile_name, emb[0]  # [T, C]


def read_all_h5(dir_path, shuffle=True):
    model = EmbeddingLookup(dir_path, cls_only=False)
    map = list(model.id_map.items())
    indices = range(len(map))
    if shuffle:
        indices = torch.randint(0, len(map), (len(map),)).tolist()
    for i in indices:
        k, v = map[i]
        cls, features = model(FakeTileBatch(k))
        emb = torch.cat([cls.unsqueeze(0), features], dim=1)
        mask = emb.abs().sum(dim=-1)[0]
        yield emb[:, mask > 0, :]


# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from src.data.components.raw_tile_dataset import RawTileDataset
# from src.data.components.tile_dataset import TileDataset
# from itertools import islice
#
# import h5py
# import numpy as np
# if __name__ == "__main__":
#     path = "data/tiles/huge/images.h5"
#     file = h5py.File(path)
#     for i, (k, v) in enumerate(islice(file.items(), 10)):
#         print(k, v)
#
#     image_dataset = RawTileDataset(
#         task_dir="data/tiles/huge/tasks/pretraining",  # Adjust if different
#         image_dir="data/tiles/huge/images",
#         split="test",  # Custom split name
#         load_images=True,
#         tag_embeddings_file="data/tiles/embeddings.pkl",
#     )
#
#     loader = DataLoader(
#         image_dataset,
#         batch_size=20,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=16,
#         persistent_workers=False,
#         pin_memory=False,
#     )
#     for tiles in tqdm(loader):
#         for i, tile in enumerate(tiles):
#             if tile.name() == "16_11406_26099":
#                 print("found the tile", tile.name(), i, tile.group_name())
#             elif tile.group_name() == "14_2851_6524":
#                 print("\nFound 14_2851_6524 but not 16, count from group: 1")
#     print(file["16_11406_26099"])
#
#
#     # ds = TileDataset("data/tiles/huge/tasks/pretraining", split="train", load_images=False)
#     # ds.__init_worker__()
#     # print(ds[str(34065)].name())
#     # print(ds[str(34066)].name())
#     # print(ds[str(34066)].group_name())
#     # p = "data/tiles/huge/images.h5"
#     # f = h5py.File(p, "r")
#     # print("length of image file", len(f))
#     # train_len = TileDataset("data/tiles/huge/tasks/pretraining", split="train", load_images=False).__len__()
#     # val_len = TileDataset("data/tiles/huge/tasks/pretraining", split="val", load_images=False).__len__()
#     # test_len = TileDataset("data/tiles/huge/tasks/pretraining", split="test", load_images=False).__len__()
#     # print("length of tile dataset", train_len+val_len+test_len)
#     # raw = RawTileDataset("data/tiles/huge/tasks/building_count", "data/tiles/huge/images", "train", load_images=True)
#     #
#     # i = 4338
#     # tiles = raw.__getitem__(i)
#     # if "16_11268_26027" in [tile.name() for tile in tiles]:
#     #     print("raw, found the tile", i)
#     #     t = tiles[[tile.name() for tile in tiles].index("16_11268_26027")]
#     #     print(t.name(), t.SAT_img, t.nbr_features)
#     #
#     # raw = RawTileDataset("data/tiles/huge/tasks/pretraining", "data/tiles/huge/images", "train", load_images=True)
#     # for i, tiles in tqdm(enumerate(raw), total=len(raw)):
#     #     if "16_11268_26027" in [tile.name() for tile in tiles]:
#     #         print("raw, found the tile", i)
#     #         t = tiles[[tile.name() for tile in tiles].index("16_11268_26027")]
#     #         print(t.name(), t.SAT_img, t.nbr_features)
#     #
#     # loader = DataLoader(
#     #     raw,
#     #     batch_size=1,
#     #     shuffle=False,
#     #     drop_last=False,
#     #     collate_fn=collate_fn,
#     #     num_workers=6,
#     #     persistent_workers=True,
#     #     pin_memory=False,
#     # )
#     # for batch in tqdm(loader):
#     #     if "16_11268_26027" in [tile.name() for tile in batch]:
#     #         print("loader, found the tile", i)
#     ### IMAGES.h5 NOT CONTAINING TO FOLLOWING TILE CAUSES ERROR; WHY NOT
#     #print(f["16_11268_26027"])
#     # for x in tqdm.tqdm(f):
#     #     c = torch.tensor(f[x]['tile_coord'][:], dtype=torch.int)
#     #     if c[1] == 11268 and c[2] == 26027:
#     #         print("FOUND")
#     #     elif c[1] == 11268 and c[2] == 26026:
#     #         print("FOUND 2")
