import os
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.data.components.tiles import Tile


def search_upwards_for_file(cwd, filename):
    d = cwd
    root = Path(d.root)

    while d != root:
        attempt = d / filename
        if attempt.exists():
            return attempt
        d = d.parent

    return None


class TileDataset(Dataset):
    def __init__(
        self,
        task_dir,
        split="train",
        use_image_transforms=True,
        add_original_image=False,
        load_images=True,
        size="huge",
        cheat=False,
        cache=False,
    ):
        # import resource
        # rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        # resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
        self.cache = cache
        if size != "huge":
            task_dir = task_dir.replace("huge", size)
        if cheat:
            task_dir += "_cheat"
        self.h5_path = Path(task_dir) / (split + ".h5")
        self.file = None
        self.load_images = load_images
        self.img_h5_path = search_upwards_for_file(Path(task_dir), "images.h5")
        if self.img_h5_path is None:
            raise ValueError(
                f"File images.h5 not found in {task_dir} or any of its parent directories"
            )
        self.img_file = None
        self.use_image_transforms = use_image_transforms
        self.add_original_image = add_original_image
        # NAIP image normalization
        self.transform = transforms.Normalize(
            mean=[0.4692, 0.4768, 0.4254], std=[0.1064, 0.0799, 0.0815]
        )
        if not os.path.exists(self.h5_path):
            raise ValueError(f"File {self.h5_path} does not exist")
        with h5py.File(self.h5_path, "r") as file:
            self.len = len(file.keys())
        self.items = {}
        self.nbr_found_images = 0
        self.nbr_not_found_images = 0

    def setup(self):
        self.file = h5py.File(self.h5_path, "r")
        if self.load_images or self.add_original_image:
            self.img_file = h5py.File(self.img_h5_path, "r")

    def __init_worker__(self):
        self.setup()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.file is None:
            raise ValueError("File is not open, run setup() or __init_worker__() first")
        if self.cache and idx in self.items:
            return self.items[idx]
        # if self.file is None:
        #     self.file = h5py.File(self.h5_path, 'r')
        # if self.img_file is None:
        #     self.img_file = h5py.File(self.img_h5_path, 'r')
        group = self.file[str(idx)]

        # Load scalar attributes
        nbr_features = group.attrs["nbr_features"]

        # Load tensors
        tile_coord = torch.tensor(group["tile_coord"][:])
        group_coord = torch.tensor(group["group_coord"][:])
        nodes = torch.tensor(group["nodes"][:])
        bbox_local_coords = torch.tensor(group["bbox_local_coords"][:])
        inter_edges = torch.tensor(group["inter_edges"][:])
        intra_edges = torch.tensor(group["intra_edges"][:])
        node_to_feature = torch.tensor(group["node_to_feature"][:])
        min_boxes = torch.tensor(group["min_boxes"][:])
        # image = torch.tensor(group['SAT_img'][:])
        tags = torch.tensor(group["tags"][:])

        image = torch.tensor([])
        if self.load_images or self.add_original_image:
            # if self.img_file is None:
            #     self.img_file = h5py.File(self.img_h5_path, 'r')
            # try:
            i = str("_".join(map(str, tile_coord.tolist())))
            group = self.img_file[i]
            image = torch.tensor(group["SAT_img"][:])
            # self.nbr_found_images += 1
            # except KeyError:
            #     image = torch.zeros((3, 224, 224), dtype=torch.float32)
            #     self.nbr_not_found_images += 1
            #     log.warning(f"Image not found for tile {tile_coord}, not found {self.nbr_not_found_images}, found {self.nbr_found_images}")

        original_img = torch.tensor([])
        if self.add_original_image:
            original_img = image

        if self.use_image_transforms and self.load_images:
            image = self.transform(image)

        tile = Tile(
            nbr_features=nbr_features,
            group_coord=group_coord,
            tile_coord=tile_coord,
            nodes=nodes,
            bbox_local_coords=bbox_local_coords,
            inter_edges=inter_edges,
            intra_edges=intra_edges,
            node_to_feature=node_to_feature,
            tags=tags,
            min_boxes=min_boxes,
            SAT_img=image,
            original_img=original_img,
        )
        if self.cache:
            self.items[idx] = tile
        return tile


def worker_init_fn(_):
    torch.utils.data.get_worker_info().dataset.setup()
