from dataclasses import dataclass
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from src.utils.graph_utils import batch_geometries


@dataclass()
class Tile:
    nbr_features: int
    # features: List[Feature]
    tile_coord: torch.Tensor  # [3] / (z, x, y)
    group_coord: torch.Tensor  # [3] / (z, x, y)
    # ---------
    nodes: torch.Tensor  # [N, 2]
    bbox_local_coords: torch.Tensor  # [N, 2]
    inter_edges: torch.Tensor  # [2, E_inter]
    intra_edges: torch.Tensor  # [2, E_intra]
    node_to_feature: torch.Tensor  # [N]
    # ---------
    tags: torch.Tensor  # [F, T] - indices
    min_boxes: torch.Tensor  # [F, 4, 2]
    # box_areas: torch.Tensor  # [F]
    # box_widths: torch.Tensor  # [F]
    # box_heights: torch.Tensor  # [F]
    # box_rotations: torch.Tensor  # [F]
    SAT_img: torch.Tensor  # [3, 244, 244]
    original_img: torch.Tensor

    def name(self):
        return "_".join(map(str, self.tile_coord.tolist()))

    def group_name(self):
        return "_".join(map(str, self.group_coord.tolist()))

    def get_node_counts(self):
        # Ensure node_to_feature is a 1D tensor of integer indices
        # Count nodes per feature, ensuring the count vector covers all features
        return torch.bincount(self.node_to_feature, minlength=self.nbr_features)


@dataclass()
class TileBatch:
    size: int
    nbr_features: int
    feature_counts: torch.Tensor  # [B]
    device: torch.device
    # ---------
    group_coords: torch.Tensor  # [B, 3]
    tile_coords: torch.Tensor  # [B, 3]
    # ----------
    nodes: torch.Tensor  # [N_tot, 2]
    bbox_local_coords: torch.Tensor  # [N, 2]
    inter_edges: torch.Tensor  # [2, E_inter_tot]
    intra_edges: torch.Tensor  # [2, E_intra_tot]
    node_to_feature: torch.Tensor  # [N_tot]
    # ----------
    tags: torch.Tensor  # [B, max_Features, max_Tags]
    min_boxes: torch.Tensor  # [B, max_F, 4, 2]
    # box_areas: torch.Tensor  # [B, max_F]
    # box_widths: torch.Tensor  # [B, max_F]
    # box_heights: torch.Tensor  # [B, max_F]
    # box_rotations: torch.Tensor  # [B, max_F]
    SAT_imgs: torch.Tensor  # [B, 3, 244, 244]
    original_imgs: torch.Tensor

    def to(self, device: torch.device):
        self.feature_counts = self.feature_counts.to(device)
        self.group_coords = self.group_coords.to(device)
        self.tile_coords = self.tile_coords.to(device)
        self.nodes = self.nodes.to(device)
        self.bbox_local_coords = self.bbox_local_coords.to(device)
        self.inter_edges = self.inter_edges.to(device)
        self.intra_edges = self.intra_edges.to(device)
        self.node_to_feature = self.node_to_feature.to(device)
        self.min_boxes = self.min_boxes.to(device)
        # self.box_areas = self.box_areas.to(device)
        # self.box_widths = self.box_widths.to(device)
        # self.box_heights = self.box_heights.to(device)
        # self.box_rotations = self.box_rotations.to(device)
        self.SAT_imgs = self.SAT_imgs.to(device)
        self.tags = self.tags.to(device)
        self.device = self.tags.device
        return self

    def names(self):
        return ["_".join(map(str, coords.tolist())) for coords in self.tile_coords]

    def group_names(self):
        return ["_".join(map(str, coords.tolist())) for coords in self.group_coords]

    def get_node_counts(self):
        # Ensure node_to_feature is a 1D tensor of integer indices
        # Count nodes per feature, ensuring the count vector covers all features
        return torch.bincount(self.node_to_feature, minlength=self.nbr_features)


def uncollate_first_tile(tile_batch: TileBatch) -> Tile:
    # Extract the number of features in the first tile
    nbr_features = tile_batch.feature_counts[0].item()
    tile_coord = tile_batch.tile_coords[0]
    group_coord = tile_batch.group_coords[0]

    # Extract node_to_feature mapping and determine N_0 (number of nodes in the first tile)
    node_to_feature = tile_batch.node_to_feature
    nbr_features_0 = nbr_features
    mask = node_to_feature >= nbr_features_0
    indices = mask.nonzero(as_tuple=False).squeeze()
    if indices.numel() == 0:
        N_0 = node_to_feature.size(0)
    else:
        N_0 = indices[0].item()

    # Extract nodes and associated data for the first tile
    nodes_0 = tile_batch.nodes[0:N_0]
    bbox_local_coords_0 = tile_batch.bbox_local_coords[0:N_0]
    node_to_feature_0 = node_to_feature[0:N_0]

    # Extract intra_edges for the first tile
    intra_edges = tile_batch.intra_edges
    mask_intra = (intra_edges[0, :] < N_0) & (intra_edges[1, :] < N_0)
    intra_edges_0 = intra_edges[:, mask_intra]

    # Extract inter_edges for the first tile
    inter_edges = tile_batch.inter_edges
    mask_inter = (inter_edges[0, :] < N_0) & (inter_edges[1, :] < N_0)
    inter_edges_0 = inter_edges[:, mask_inter]

    # Extract tags and min_boxes for the first tile
    tags_0 = tile_batch.tags[0]
    min_boxes_0 = tile_batch.min_boxes[0]

    # Extract images for the first tile
    SAT_img_0 = tile_batch.SAT_imgs[0]
    original_img_0 = tile_batch.original_imgs[0]

    # Create and return the reconstructed Tile
    tile = Tile(
        nbr_features=nbr_features,
        tile_coord=tile_coord,
        group_coord=group_coord,
        nodes=nodes_0,
        bbox_local_coords=bbox_local_coords_0,
        inter_edges=inter_edges_0,
        intra_edges=intra_edges_0,
        node_to_feature=node_to_feature_0,
        tags=tags_0,
        min_boxes=min_boxes_0,
        SAT_img=SAT_img_0,
        original_img=original_img_0,
        # box_areas=tile_batch.box_areas[0],
        # box_widths=tile_batch.box_widths[0],
        # box_heights=tile_batch.box_heights[0],
        # box_rotations=tile_batch.box_rotations[0],
    )
    return tile


def collate_tiles(tiles: List[Tile]) -> TileBatch:
    assert type(tiles[0]) == Tile, f"Expected Tile, got {type(tiles[0])}"
    node_lists = [tile.nodes for tile in tiles]
    intra_edge_lists = [tile.intra_edges for tile in tiles]
    inter_edge_lists = [tile.inter_edges for tile in tiles]
    node_to_feature_lists = [tile.node_to_feature for tile in tiles]
    bbox_local_coords_lists = [tile.bbox_local_coords for tile in tiles]
    nodes, bbox_local_coords, intra_edge, inter_edge, node_to_feature = (
        batch_geometries(
            node_lists,
            bbox_local_coords_lists,
            intra_edge_lists,
            inter_edge_lists,
            node_to_feature_lists,
        )
    )
    feature_counts = torch.tensor([tile.nbr_features for tile in tiles])

    min_boxes = pad_sequence([tile.min_boxes for tile in tiles], batch_first=True)
    # areas = pad_sequence([tile.box_areas for tile in tiles], batch_first=True)
    # widths = pad_sequence([tile.box_widths for tile in tiles], batch_first=True)
    # heights = pad_sequence([tile.box_heights for tile in tiles], batch_first=True)
    # rotations = pad_sequence([tile.box_rotations for tile in tiles], batch_first=True)

    # tags is list of tensors [max_features, max_tags]
    max_length = max(tile.tags.size(1) for tile in tiles)
    padded_tags = [
        torch.nn.functional.pad(
            tile.tags, (0, max_length - tile.tags.size(1))
        )  # Pad to the right
        for tile in tiles
    ]
    tags = torch.nn.utils.rnn.pad_sequence(
        padded_tags, batch_first=True, padding_value=0
    )
    # tags is now [B, max_max_features, max_max_tags]

    return TileBatch(
        size=len(tiles),
        nbr_features=feature_counts.sum().item(),
        feature_counts=feature_counts,
        device=torch.device("cpu"),
        # --------
        group_coords=torch.stack([tile.group_coord for tile in tiles]),
        tile_coords=torch.stack([tile.tile_coord for tile in tiles]),
        # --------
        nodes=nodes,
        bbox_local_coords=bbox_local_coords,
        inter_edges=inter_edge,
        intra_edges=intra_edge,
        node_to_feature=node_to_feature,
        # --------
        tags=tags,
        min_boxes=min_boxes,
        # box_widths=widths,
        # box_heights=heights,
        # box_areas=areas,
        # box_rotations=rotations,
        SAT_imgs=torch.stack([tile.SAT_img for tile in tiles]),
        original_imgs=torch.stack([tile.original_img for tile in tiles]),
    )
