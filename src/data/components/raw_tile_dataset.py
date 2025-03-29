import typing

from PIL import Image
from einops import einops
from torch.nn.utils.rnn import pad_sequence
import glob
import os
from pathlib import Path
from typing import Tuple, List
import pandas as pd
from torchvision.transforms import transforms
import src.data.components.processed_tile_group_pb2 as osm_schema
import torch

from src import log
from src.data.components.tiles import Tile
from src.modules.tag_encoder import TagIndexer
from torch.utils.data import Dataset
from src.utils.tile_utils import get_sub_tiles


class RawTileDataset(Dataset):
    def __init__(
        self,
        task_dir="data/tiles/tiny/pretraining",
        image_dir="data/tiles/huge/images",
        split="train",
        load_images=True,
        tag_embeddings_file="data/tiles/embeddings.pkl",
    ):
        self.img_dir = image_dir
        self.split = split
        self.tile_dir = task_dir
        self.load_images = load_images

        self.tag_indexer = TagIndexer(tag_embeddings_file)

        # Initialize lists to collect file paths and tile indices
        file_paths = []

        # Collect all .pbf file paths in the specified directory and split
        search_pattern = os.path.join(task_dir, split, "*.pbf")
        self.pbf_files = glob.glob(search_pattern)
        if not self.pbf_files:
            raise FileNotFoundError(f"No .pbf files found in {search_pattern}")

        search_pattern = os.path.join(image_dir, "*.webp")
        self.img_files = set(glob.glob(search_pattern))
        if not self.img_files:
            raise FileNotFoundError(f"No .webp files found in {search_pattern}")

        # Iterate over each file to extract valid tile indices
        for path in self.pbf_files:
            try:
                if "14_2851_6524" in path:
                    print(f"Found 14_2851_6524 in {path}")

                group = osm_schema.TileGroup()
                with open(path, "rb") as f:
                    group.ParseFromString(f.read())
            except Exception as e:
                log.error(f"reading {path}: {e}")
                print(f"ERROR: reading {path}: {e}")
                continue  # Skip corrupted or unreadable files

            if os.path.exists(os.path.join(image_dir, f"14_{group.x}_{group.y}.webp")):
                if group.x == 2851 and group.y == 6524:
                    print(f"Found 14_{group.x}_{group.y} in {image_dir}")
                file_paths.append(path)
            else:
                log.error(f"Image not found for {group.x}_{group.y}")
                print(f"ERROR: Image not found for {group.x}_{group.y}")

        # Create a DataFrame from the collected data
        self.items_df = pd.DataFrame(
            {
                "file_path": file_paths,
            }
        )

        log.info(f"Found {len(self.items_df)} tile_groups")
        if len(self.items_df) == 0:
            raise ValueError("No tile groups found after processing.")

    def __len__(self):
        return len(self.items_df)

    def __getitem__(
        self, idx
    ) -> List[Tile]:  # Replace 'Tile' with your actual Tile class/type
        try:
            if not 0 <= idx < len(self):
                raise IndexError(
                    f"Index {idx} out of range for TileDataset of size {len(self)}."
                )
            row = self.items_df.iloc[idx]
            file_path = row["file_path"]
            tile_group_path = Path(file_path)
            tile_group = osm_schema.TileGroup()
            with open(tile_group_path, "rb") as f:
                tile_group.ParseFromString(f.read())
            g_coord = torch.tensor(
                (tile_group.zoom, tile_group.x, tile_group.y), dtype=torch.int32
            )
            tiles = tile_group.tiles
            for i, tile in enumerate(tiles):
                if tile.x == 11406 and tile.y == 26099:
                    print("RAW: found the tile", tile.name(), i, tile.group_name())
                elif tile_group.x == 2851 and tile_group.y == 6524:
                    print("\nRAW: Found 14_2851_6524 but not 16, count from group: 1")
            image_dict = load_images_from_path(
                os.path.join(self.img_dir, f"14_{g_coord[1]}_{g_coord[2]}.webp"),
                self.load_images,
            )
            items = []
            for tile in tiles:
                if (tile.x, tile.y) not in image_dict:
                    continue
                tile_coord = torch.tensor(
                    (tile.zoom, tile.x, tile.y), dtype=torch.int32
                )
                assert not torch.isnan(tile_coord).any(), "nodes contains NaNs"
                # Extract nodes: reshape flat list to [N, 2]
                nodes = torch.tensor(tile.nodes, dtype=torch.float32).reshape(-1, 2)
                assert not torch.isnan(nodes).any(), "nodes contains NaNs"
                bbox_local_coords = torch.tensor(
                    tile.local_coords, dtype=torch.float32
                ).reshape(-1, 2)
                assert not torch.isnan(bbox_local_coords).any(), (
                    "bbox_local_coords contains NaNs"
                )
                # Extract intra_edges: reshape to [2, E_intra]
                edges = torch.tensor(tile.intra_edges, dtype=torch.long)
                assert not torch.isnan(edges).any(), "intra_edges contains NaNs"
                intra_edges = torch.stack((edges[::2], edges[1::2]))

                # Extract inter_edges: reshape to [2, E_inter]
                edges = torch.tensor(tile.inter_edges, dtype=torch.long)
                assert not torch.isnan(edges).any(), "inter_edges contains NaNs"
                inter_edges = torch.stack((edges[::2], edges[1::2]))

                # Extract min_boxes
                min_boxes = torch.zeros((len(tile.features), 4, 2), dtype=torch.float32)
                areas = torch.zeros((len(tile.features)), dtype=torch.float32)
                widths = torch.zeros((len(tile.features)), dtype=torch.float32)
                heights = torch.zeros((len(tile.features)), dtype=torch.float32)
                rotation = torch.zeros((len(tile.features)), dtype=torch.float32)
                no_min_box = 0
                for j, feature in enumerate(tile.features):
                    if len(feature.min_box) == 0:
                        no_min_box += 1
                        continue
                    min_boxes[j] = torch.tensor(
                        feature.min_box, dtype=torch.float32
                    ).reshape(4, 2)
                    areas[j] = feature.area
                    widths[j] = feature.width
                    heights[j] = feature.height
                    rotation[j] = feature.rotation
                assert not torch.isnan(min_boxes).any(), "min_boxes contains NaNs"
                assert not torch.isnan(areas).any(), "areas contains NaNs"
                assert not torch.isnan(widths).any(), "widths contains NaNs"
                assert not torch.isnan(heights).any(), "heights contains NaNs"
                min_boxes = min_boxes[:, :, [1, 0]]
                # if no_min_box > 0:
                #     log.info(f"{no_min_box} features without min_box")

                node_to_feature = torch.tensor(tile.node_to_feature, dtype=torch.long)
                assert not torch.isnan(node_to_feature).any(), (
                    "node_to_feature index contains NaNs"
                )

                # Extract tags
                tags = []
                for feature in tile.features:
                    temp = []
                    for k, v in zip(feature.tags[::2], feature.tags[1::2]):
                        tag = self.tag_indexer.tag_to_index((k, v))
                        if tag != 0:
                            temp.append(tag)
                    tags.append(torch.tensor(temp, dtype=torch.long))
                if len(tags) == 0:
                    log.info(
                        f"Tile {tile.zoom}_{tile.x}_{tile.y} has no tags, group: {g_coord}"
                    )
                    tags = torch.tensor([[]], dtype=torch.long)
                else:
                    tags = pad_sequence(tags, batch_first=True, padding_value=0)
                # tensor of shape [max_Features, max_Tags]

                items.append(
                    Tile(
                        nbr_features=len(tile.features),
                        group_coord=g_coord,
                        tile_coord=tile_coord,
                        # tile_name=f"{tile.zoom}_{tile.x}_{tile.y}",
                        # -----------
                        nodes=nodes,
                        bbox_local_coords=bbox_local_coords,
                        inter_edges=inter_edges,
                        intra_edges=intra_edges,
                        node_to_feature=node_to_feature,
                        # -----------
                        min_boxes=min_boxes,
                        # box_areas=areas,
                        # box_widths=widths,
                        # box_heights=heights,
                        # box_rotations=rotation,
                        tags=tags,
                        SAT_img=image_dict[(tile.x, tile.y)]
                        if self.load_images
                        else torch.tensor([]),
                        original_img=torch.tensor([]),
                    )
                )
            return items
        except Exception as e:
            log.error(f"Error in __getitem__ at index {idx}: {e}")
            raise e  # Re-raise the exception to let the DataLoader handle it


def load_images_from_path(
    path, load_images=True
) -> typing.Dict[Tuple[int, int], torch.Tensor]:
    if not os.path.exists(path):
        return {}
    transform = transforms.ToTensor()
    try:
        old_coords = list(map(int, path.split(".")[-2].split("/")[-1].split("_")[-3:]))
        coords = get_sub_tiles(old_coords[1], old_coords[2])
        if load_images:
            with Image.open(path) as img:
                image = img.convert("RGB")
                # Transform the image to a tensor
                image = transform(image)
                # Split into 4x4 grid
                images = einops.rearrange(
                    image, "c (g1 h) (g2 w) -> (g1 g2) c h w", h=224, w=224
                )
                img_dict = {k: v for k, v in zip(coords, images)}
        else:
            img_dict = {k: None for k in coords}
        return img_dict
    except Exception as e:
        log.error(f"Error reading image {path}: {e}")
        # Handle the error gracefully, e.g., return an empty dictionary
        return {}
