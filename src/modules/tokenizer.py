import time
from enum import IntEnum
from typing import Tuple

import torch
from torch import nn, Tensor
from src.modules.mlp import MLP

from src.lightning_utils.logging_utils import prefix_keys
from src.modules.vision_backbones import ViTB16, ImageSelector
from src.data.components.tiles import TileBatch
from src.modules.geometry_encoder import (
    load_geometry_encoder_pretrained,
)
from src.modules.tag_encoder import TagEncoder
from src.utils.sort_utils import rearrange_tokens, spatial_order


class Modality(IntEnum):
    PAD = 0
    OSM = 1
    GEO = 2  # deprecated
    IMG = 3


class TileTokenizer(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        tag_embedding_file: str = "data/tiles/embeddings.pkl",
        tokenize_images: bool = True,
        tokenize_geometry: bool = True,
        tokenize_tags: bool = True,
        geometry_encoder: nn.Module = None,
        geometry_encoder_out_dim: int = 2048,
        img_encoder_selector: nn.Module = ImageSelector(),
        img_encoder: nn.Module = ViTB16(),
        img_encoder_out_dim: int = 768,
        sort_spatially: bool = False,
        return_indices: bool = False,
        tag_geo_dropout: float = 0.3,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.sort_spatially = sort_spatially
        self.tokenize_images = tokenize_images
        self.tokenize_geometry = tokenize_geometry
        self.tokenize_tags = tokenize_tags
        self.return_indices = return_indices
        self.tag_geo_dropout = tag_geo_dropout
        assert tokenize_images or tokenize_geometry or tokenize_tags, (
            "At least one of the modalities must be tokenized."
        )

        # 1024-d
        if self.tokenize_tags:
            self.tag_encoder = TagEncoder(  # no parameters
                embedding_file=tag_embedding_file, embedding_dim=1024
            )
            self.tag_token_projection = nn.Linear(1024, token_dim)
            self.tag_encoder.requires_grad_(False).eval()

        # 2048-d
        if self.tokenize_geometry:
            if geometry_encoder is None:
                geometry_encoder = load_geometry_encoder_pretrained(
                    "src/models/pretrained/polygnn-ckpt-dec-26"
                )
            self.geometry_encoder = geometry_encoder
            self.geo_token_projection = nn.Linear(geometry_encoder_out_dim, token_dim)
            self.geometry_encoder.requires_grad_(False).eval()
            # self.point_token = nn.Parameter(torch.zeros(1, 1, token_dim))

        # 768-d
        if self.tokenize_images:
            # self.sat_img_encoder = ViTB16()
            self.img_encoder_selector = img_encoder_selector
            self.img_encoder = img_encoder
            # self.sat_img_encoder = SatImgEncoder(res=300 / 224)
            # self.sat_token_projection = nn.Linear(1024, token_dim)
            self.sat_token_projection = nn.Linear(img_encoder_out_dim, token_dim)
            self.img_encoder.requires_grad_(False).eval()

        self.tag_norm = nn.LayerNorm(token_dim)
        self.geo_norm = nn.LayerNorm(token_dim)
        self.sat_norm = nn.LayerNorm(token_dim)

        self.geo_tag_fusion = MLP(token_dim * 2, token_dim * 2, token_dim, bias=True)

        self.tag_emb_time = 0
        self.geo_emb_time = 0
        self.sat_emb_time = 0
        self.geo_tag_fusion_time = 0
        self.sort_time = 0
        self.total_time = 0

    def get_metrics(self, prefix=""):
        d = {
            "tags_time": self.tag_emb_time * 1e3,
            "geo_time": self.geo_emb_time * 1e3,
            "sat_time": self.sat_emb_time * 1e3,
            "sort_time": self.sort_time * 1e3,
            "total_time": self.total_time * 1e3,
        }
        if prefix != "":
            d = prefix_keys(prefix, d)
        self.tag_emb_time = 0
        self.geo_emb_time = 0
        self.sat_emb_time = 0
        self.sort_time = 0
        self.total_time = 0
        return d

    def forward(
        self, batch: TileBatch
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor, Tensor]:
        r"""
        Tokenizes a batch of tiles into:

        tokens: A feature or satellite patch embedding. [B, token_dim]
        bboxes: The bounding boxes of the tokens. [B, 4, 2]
        modalities: The modalities of the tokens. [B, 1] (dtype=torch.int) where 1 is feature and 2 is satellite patch, PAD=0.
        """
        init = time.time()
        empty = torch.tensor([], device=batch.device)
        geo_tokens = empty.clone()
        tag_tokens = empty.clone()

        osm_tokens = empty.clone()
        osm_bboxes = empty.clone()

        img_tokens = empty.clone()
        img_bboxes = empty.clone()

        no_features = batch.nbr_features == 0

        if self.tokenize_tags and not no_features:
            start = time.time()
            # self.tag_encoder = self.tag_encoder.to(batch.device)
            with torch.no_grad():
                tag_embs = self.tag_encoder(batch.tags)
            tag_embs = self.tag_token_projection(tag_embs.to(batch.device))
            tag_embs = self.tag_norm(tag_embs)
            self.tag_emb_time += time.time() - start
            tag_tokens = tag_embs

        if self.tokenize_geometry and not no_features:
            start = time.time()
            # self.geometry_encoder = self.geometry_encoder.to(batch.device)
            with torch.no_grad():
                # runs on ALL features, even points and two-node lines :/
                geo_embs_cat = self.geometry_encoder.forward(
                    batch.bbox_local_coords,
                    batch.intra_edges,
                    batch.inter_edges,
                    batch.node_to_feature,
                )
            geo_embs = split_feat_embs_to_batch(geo_embs_cat, batch)
            geo_embs = self.geo_token_projection(geo_embs)
            geo_embs = self.geo_norm(geo_embs)
            self.geo_emb_time += time.time() - start
            geo_tokens = geo_embs

        start = time.time()
        if not no_features:
            if self.tokenize_tags and self.tokenize_geometry:
                # random tag/geo dropout, 30% do one, 0% do both, 70% do none
                if self.training and torch.rand(1) < self.tag_geo_dropout:
                    if torch.rand(1) > 0.5:
                        osm_tokens = torch.cat(
                            (tag_tokens, torch.zeros_like(geo_tokens)), dim=-1
                        )
                    else:
                        osm_tokens = torch.cat(
                            (torch.zeros_like(tag_tokens), geo_tokens), dim=-1
                        )
                else:
                    osm_tokens = torch.cat((tag_tokens, geo_tokens), dim=-1)
            elif self.tokenize_tags:
                osm_tokens = torch.cat(
                    (tag_tokens, torch.zeros_like(tag_tokens)), dim=-1
                )
            elif self.tokenize_geometry:
                osm_tokens = torch.cat(
                    (torch.zeros_like(geo_tokens), geo_tokens), dim=-1
                )

            if self.tokenize_tags or self.tokenize_geometry:
                osm_bboxes = batch.min_boxes.view(batch.size, -1, 8)
                osm_tokens = self.geo_tag_fusion(osm_tokens)
        self.geo_tag_fusion_time += time.time() - start

        if self.tokenize_images:
            start = time.time()
            # self.img_encoder = self.img_encoder.to(batch.device)
            with torch.no_grad():
                input = self.img_encoder_selector(batch)
                sat_cls_token, sat_patch_tokens = self.img_encoder.forward(input)

            sat_patch_tokens = self.sat_token_projection(sat_patch_tokens)
            sat_patch_tokens = self.sat_norm(sat_patch_tokens)
            sat_patch_bboxes = compute_normalized_bboxes(
                batch.size, 14, device=batch.device
            )

            sat_cls_token = self.sat_token_projection(sat_cls_token)
            sat_cls_token = self.sat_norm(sat_cls_token)
            sat_cls_bbox = (
                torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], device=batch.device)
                .repeat(batch.size)
                .view(-1, 1, 8)
            )
            sat_cls_token = sat_cls_token.unsqueeze(1)

            img_tokens = torch.cat((sat_cls_token, sat_patch_tokens), dim=1)
            img_bboxes = torch.cat((sat_cls_bbox, sat_patch_bboxes), dim=1)
            self.sat_emb_time += time.time() - start

        zeros = torch.zeros(batch.size, device=batch.device, dtype=torch.int)
        num_map_entities = (
            batch.feature_counts
            if self.tokenize_tags or self.tokenize_geometry
            else zeros
        )
        num_imgs = (zeros + 1) * 197 if self.tokenize_images else zeros
        num_tokens = num_map_entities + num_imgs
        max_tokens = num_tokens.max().item()

        tokens = torch.zeros(
            (batch.size, max_tokens, self.token_dim), device=batch.device
        )
        positions = torch.zeros((batch.size, max_tokens, 8), device=batch.device)
        modalities = torch.zeros((batch.size, max_tokens), device=batch.device)
        for i, (map_ent, imgs) in enumerate(zip(num_map_entities, num_imgs)):
            pos = 0
            if self.tokenize_tags or self.tokenize_geometry:
                tokens[i, pos : pos + map_ent] = osm_tokens[i, :map_ent]
                positions[i, pos : pos + map_ent] = osm_bboxes[i, :map_ent]
                modalities[i, pos : pos + map_ent] = Modality.OSM
                pos += map_ent
            if self.tokenize_images:
                tokens[i, pos : pos + imgs] = img_tokens[i, :imgs]
                positions[i, pos : pos + imgs] = img_bboxes[i, :imgs]
                modalities[i, pos : pos + imgs] = Modality.IMG
                pos += imgs

        B, T, _ = tokens.shape

        if self.sort_spatially:
            start = time.time()
            indices = spatial_order(positions, modalities)
            tokens, positions, modalities = rearrange_tokens(
                tokens, positions, modalities, indices
            )
            self.sort_time += time.time() - start
        elif self.return_indices:
            indices = (
                torch.arange(0, tokens.size(1), device=batch.device)
                .unsqueeze(0)
                .expand(B, -1)
            )

        self.total_time = time.time() - init
        # only tokens have gradients, indices are only dependent on positions and modalities -- grads are fine
        if self.return_indices:
            return tokens, positions, modalities, indices
        return tokens, positions, modalities


def compute_normalized_bboxes(batch_size, grid_size, device):
    patch_indices = torch.arange(grid_size * grid_size, device=device).reshape(
        1, grid_size, grid_size
    )
    patch_indices = patch_indices.expand(
        batch_size, -1, -1
    )  # [B, grid_size, grid_size]

    # Calculate normalized coordinates
    step = 1.0 / grid_size
    x_min = (patch_indices % grid_size) * step
    y_min = 1.0 - (patch_indices // grid_size) * step - step
    x_max = x_min + step
    y_max = y_min + step

    bboxes = torch.stack(
        [
            torch.stack([x_min, y_min], dim=-1),  # Top-Left
            torch.stack([x_min, y_max], dim=-1),  # Top-Right
            torch.stack([x_max, y_max], dim=-1),  # Bottom-Left
            torch.stack([x_max, y_min], dim=-1),  # Bottom-Right
        ],
        dim=-2,
    )  # [B, grid_size, grid_size, 8]

    # Reshape to [B, num_patches, 8]
    bboxes = bboxes.view(batch_size, grid_size * grid_size, 8)
    return bboxes


def split_feat_embs_to_batch(embs: Tensor, batch: TileBatch):
    """
    Go from (NUM_FEATURES, TOKEN_DIM) to (B, MAX_TILE_FEATURES, TOKEN_DIM) with zeros as padding.
    """
    assert batch.device == embs.device, (
        "Batch and embeddings must be on the same device."
    )
    assert batch.nbr_features == embs.size(0), (
        "Sum of batch_sizes must equal the number of tokens."
    )
    d = batch.device
    max_features = batch.feature_counts.max().item()
    if embs.is_mps:
        batch_indices = torch.repeat_interleave(
            torch.arange(batch.size, device="cpu"), batch.feature_counts.to("cpu")
        ).to("mps:0")
    else:
        batch_indices = torch.repeat_interleave(
            torch.arange(batch.size, device=d), batch.feature_counts
        )
    cumsum_sizes = torch.cumsum(batch.feature_counts, dim=0)
    start_indices = torch.cat((torch.tensor([0], device=d), cumsum_sizes[:-1]))
    positions = torch.arange(embs.size(0), device=d) - start_indices[batch_indices]
    output_tensor = torch.zeros(
        (batch.size, max_features, embs.shape[1]), dtype=embs.dtype, device=d
    )
    output_tensor[batch_indices, positions] = embs
    return output_tensor
