import os
from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor



class TagIndexer:
    def __init__(
        self,
        embedding_file: str = "data/tiles/embeddings.pkl",
        dim=1024,
    ):
        """Initializes the TagIndexer.
        Args:
            embedding_file (str): Path to the pickle file containing embeddings.
        """
        super().__init__()
        assert os.path.exists(embedding_file), (
            f"Embedding file not found at {embedding_file}"
        )
        self.dim = dim

        # Load the embeddings dictionary
        self.loaded_embeddings: dict[Tuple[str, str], torch.Tensor] = (
            pd.read_pickle(embedding_file)
            .set_index(["key", "value"])["embedding"]
            .to_dict()
        )

        for k in list(self.loaded_embeddings.keys()):
            emb: torch.Tensor = self.loaded_embeddings[k]
            emb = emb.clone().detach().float()
            self.loaded_embeddings[k] = emb
            assert type(emb) == torch.Tensor, f"Expected torch.Tensor, got {type(emb)}"
            if str(k[1]) == "nan":
                self.loaded_embeddings[(k[0], "")] = emb
                self.loaded_embeddings.pop(k)

        self.tag_to_index_dict: Dict[Tuple[str, str], int] = {
            (str(k), str(v)): i + 1  # +1 to avoid 0 index
            for i, (k, v) in enumerate(self.loaded_embeddings.keys())
        }

        # add padding vector
        self.loaded_embeddings[("PAD", "PAD")] = torch.zeros(
            dim
        )  # Add padding embedding
        self.tag_to_index_dict[("PAD", "PAD")] = 0

        self.index_to_tag_dict = {i: k for k, i in self.tag_to_index_dict.items()}

        for i, (k, v) in enumerate(self.loaded_embeddings.keys()):
            if (k, "") not in self.tag_to_index_dict:
                self.tag_to_index_dict[(k, "")] = self.tag_to_index_dict.get((k, v))

        self.nbr_unique_tags = len(self.loaded_embeddings)

    def tag_to_index(self, tag: Tuple[str, str]) -> int:
        """Converts a tag to its corresponding index.
        Args:
            tag (Tuple[str, str]): The tag to convert.
        Returns:
            int: The index of the tag.
        """
        i = self.tag_to_index_dict.get(tag, 0)
        if i == 0:
            i = self.tag_to_index_dict.get((tag[0], ""), 0)
        if i == 0:
            while ":" in tag[0] and i == 0:
                tag = (":".join(tag[0].split(":")[:-1]), "")
                i = self.tag_to_index_dict.get(tag, 0)
        return i

    def index_to_tag(self, index: int | Tensor) -> Tuple[str, str]:
        """Converts an index to its corresponding tag.
        Args:
            index (int): The index to convert.
        Returns:
            Tuple[str, str]: The tag of the index.
        """
        if isinstance(index, torch.Tensor):
            assert index.numel() == 1, f"Expected scalar, got {index.shape}"
            index = index.item()
        return self.index_to_tag_dict.get(index, ("", ""))

    def index_to_embedding(self, index: int) -> torch.Tensor:
        """Converts an index to its corresponding embedding.
        Args:
            index (int): The index to convert.
        Returns:
            torch.Tensor: The embedding of the index.
        """
        return self.loaded_embeddings.get(
            self.index_to_tag(index), torch.zeros(self.dim)
        )

    def embedding_matrix(self) -> torch.Tensor:
        """Returns the embedding matrix."""
        return torch.stack(
            [self.index_to_embedding(i) for i in range(len(self.loaded_embeddings))]
        )


class TagEncoder(nn.Module):
    def __init__(
        self,
        embedding_file: str = "data/tiles/embeddings.pkl",
        embedding_dim: int = 1024,
    ):
        """Initializes the TagEncoder.

        Args:
            embedding_file (str): Path to the pickle file containing embeddings.
            embedding_dim (int): Dimension of the embeddings.
            padding_idx (int): Index used for padding. Ensure this index is in [0, num_embeddings-1].
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        tag_indexer = TagIndexer(embedding_file=embedding_file, dim=embedding_dim)
        self.nbr_unique_tags = tag_indexer.nbr_unique_tags

        # Ensure the embeddings are ordered according to tag_indexer
        embedding_matrix = tag_indexer.embedding_matrix()

        # Initialize the embedding layer
        self.embedding = nn.EmbeddingBag.from_pretrained(
            embedding_matrix, mode="sum", freeze=True, padding_idx=0, sparse=True
        )
        self.eval()
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> Tensor:
        assert x.device == self.embedding.weight.device, (
            f"Expected device {self.embedding.weight.device}, got {x.device}"
        )
        assert x.dtype == torch.long, f"Expected torch.long, got {x.dtype} {x}"
        """
        Processes a batch of tag indices and returns aggregated embeddings.

        Args:
            batch (torch.Tensor): Tensor of shape [nbr_tiles (B), max_features, max_tags] containing tag indices.

        Returns:
            Tensor: Aggregated embeddings of shape [B, F, dim].
        """
        # Shape: [B, F, T]
        B, F, T = x.shape
        if F == 0 or T == 0:
            return torch.zeros(B, F, self.embedding_dim)

        x = x.view(B * F, T)
        x = self.embedding(x)
        x = x.view(B, F, self.embedding_dim)

        x = nn.functional.layer_norm(x, x.shape[-1:])
        # Shape: [B, F, embedding_dim]

        return x
