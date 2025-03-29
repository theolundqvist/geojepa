import os

import torch
from torch import Tensor
from torch import nn
from src.data.components.tiles import TileBatch


class SatImgEmbeddingLookup(nn.Module):
    def __init__(
        self,
        embedding_dir: str = "data/tiles/sthlm/img_embeddings",
        embedding_dim: int = 1024,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_dir = embedding_dir
        files = os.listdir(embedding_dir)
        self.coord_to_file = {}
        for f in files:
            if ".pt" in f:
                # name = z_x_y_sat_emb.pkl
                coord = list(map(int, f.split("_")[:3]))
                self.coord_to_file[tuple(coord)] = f

    def forward(self, batch: TileBatch) -> (Tensor, Tensor):
        """
        @param batch: ZXY coords, Tensor of shape (B, 3)
        @return:
            image cls embeddings of shape (B, 1, 1024)
            image patch embeddings of shape (B, 196, 1024)
        """
        tile_coords = [tuple(c.tolist()) for c in batch.tile_coords]
        cls_embeddings = torch.zeros((len(tile_coords), 1, self.embedding_dim))
        patch_embeddings = torch.zeros((len(tile_coords), 196, self.embedding_dim))
        for i, coord in enumerate(tile_coords):
            fname = self.coord_to_file[coord]
            cls, patches = torch.load(f"{self.embedding_dir}/{fname}.pt")
            cls_embeddings[i] = cls
            patch_embeddings[i] = patches
        return cls_embeddings, patch_embeddings
