from pathlib import Path

import h5py
import torch
import numpy as np
import lightning as L
from torch import Tensor
from typing import Tuple

from src.data.components.tiles import TileBatch


class EmbeddingLookup(L.LightningModule):
    def __init__(
        self,
        dir: str = "data/embeddings/vitb16/pretraining_huge",
        cls_only: bool = False,
        mods: bool = False,
    ):
        super().__init__()
        path = Path(dir)
        self.identifiers_path = path / "identifiers.h5"
        self.cls_path = path / "cls.h5"
        self.feat_path = path / "feat.h5"
        self.cls_only = cls_only
        self.mods = mods

        with h5py.File(self.identifiers_path, "r") as f:
            ids = f["identifiers"][:]

        if isinstance(ids[0], bytes):
            ids = [id.decode("utf-8") for id in ids]
        else:
            ids = [str(id) for id in ids]

        self.id_map = {identifier: idx for idx, identifier in enumerate(ids)}
        del ids
        self.cls_file = h5py.File(self.cls_path, "r")
        if not self.cls_only:
            self.feat_file = h5py.File(self.feat_path, "r")

    def forward(
        self, batch: TileBatch
    ) -> Tuple[Tensor, Tensor, Tensor] | Tuple[Tensor, Tensor] | Tensor:
        names = batch.names()
        try:
            original_indices = np.array(
                [self.id_map[name] for name in names], dtype=np.int64
            )
        except KeyError as e:
            missing_id = e.args
            raise ValueError(
                f"**Identifier '{missing_id}' is missing**., key: {names}, id_map_length: {len(self.id_map)}"
            ) from None

        sort_order = np.argsort(original_indices)
        sorted_indices = original_indices[sort_order]
        inv_order = np.argsort(sort_order)

        cls = self.cls_file["cls"][sorted_indices][inv_order]
        cls_embeddings = torch.from_numpy(cls).to(self.device)
        if self.cls_only:
            return cls_embeddings
        else:
            embs = self.feat_file["features"][sorted_indices][inv_order]
            feat_embeddings = torch.from_numpy(embs).to(self.device)
            if self.mods:
                mods = self.feat_file["mods"][sorted_indices][inv_order]
                mods = torch.from_numpy(mods).to(self.device)
                return cls_embeddings, feat_embeddings, mods
            return cls_embeddings, feat_embeddings

    # def __del__(self):
    #     if self.cls_file:
    #         self.cls_file.close()
    #     if not self.cls_only and self.feat_file:
    #         self.feat_file.close()
