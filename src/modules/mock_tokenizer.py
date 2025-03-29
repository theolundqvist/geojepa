# MockTokenizer.py

import torch
from src.modules.tokenizer import compute_normalized_bboxes, Modality
from src.utils.sort_utils import spatial_order, rearrange_tokens


class MockTokenizer:
    def __init__(
        self,
        token_dim: int = 1,
        tokenize_images: bool = True,
        tokenize_geometry: bool = True,
        tokenize_tags: bool = True,
        sort_spatially: bool = True,
    ):
        self.token_dim = token_dim
        self.tokenize_images = tokenize_images
        self.tokenize_geometry = tokenize_geometry
        self.tokenize_tags = tokenize_tags
        self.sort_spatially = sort_spatially
        self.indices = torch.empty((32, 10), dtype=torch.long)

    def __call__(self, batch):
        """
        Processes the tile_batch and returns tokens, positions, and modalities.

        Args:
            batch: TileBatch object.

        Returns:
            tokens: torch.Tensor of shape (B, T, token_dim)
            positions: torch.Tensor of shape (B, T, 8)  # Positions are flattened bboxes
            modalities: torch.Tensor of shape (B, T)
        """
        B = batch.size  # Batch size
        device = batch.device

        # Number of features per batch item
        feature_counts = batch.feature_counts  # [B]

        # For simplicity, we can create dummy tokens

        tokens_list = []
        positions_list = []
        modalities_list = []

        max_length = 0  # Keep track of the maximum sequence length for padding

        for i in range(B):
            num_feat = feature_counts[i].item()
            tokens = []
            positions = []
            modalities = []

            # Tags tokens
            if self.tokenize_tags:
                # One token per feature
                tag_tokens = torch.zeros((num_feat, self.token_dim), device=device)
                tag_positions = batch.min_boxes[i, :num_feat].view(
                    num_feat, 8
                )  # Use actual positions
                tag_modalities = (
                    torch.ones(num_feat, dtype=torch.long, device=device) * Modality.OSM
                )  # Modality 1
                tokens.append(tag_tokens)
                positions.append(tag_positions)
                modalities.append(tag_modalities)

            # Geometry tokens
            if self.tokenize_geometry:
                # One token per feature
                geo_tokens = torch.zeros((num_feat, self.token_dim), device=device)
                geo_positions = batch.min_boxes[i, :num_feat].view(
                    num_feat, 8
                )  # Use actual positions
                tokens.append(geo_tokens)
                positions.append(geo_positions)

            # Image tokens
            if self.tokenize_images:
                # ViT produces 197 tokens per image (1 class token + 196 patch tokens)
                img_tokens = torch.zeros((197, self.token_dim), device=device)
                img_positions = torch.zeros(
                    (197, 8), device=device
                )  # Positions for image tokens
                # For the image patches, we can compute positions
                # First token is class token, positions can be zeros
                img_positions[0] = torch.tensor(
                    (0, 0, 0, 1, 1, 1, 1, 0), device=device
                )  # Class token

                # For the 196 patches, compute positions
                patch_positions = compute_normalized_bboxes(
                    1, 14, device=device
                ).squeeze(0)  # [196, 8]
                img_positions[1:] = patch_positions

                img_modalities = (
                    torch.ones(197, dtype=torch.long, device=device) * Modality.IMG
                )  # Modality 3
                tokens.append(img_tokens)
                positions.append(img_positions)
                modalities.append(img_modalities)

            # Concatenate
            tokens_i = torch.cat(tokens, dim=0)  # [T_i, token_dim]
            positions_i = torch.cat(positions, dim=0)  # [T_i, 8]
            modalities_i = torch.cat(modalities, dim=0)  # [T_i]

            tokens_list.append(tokens_i)
            positions_list.append(positions_i)
            modalities_list.append(modalities_i)

            if tokens_i.shape[0] > max_length:
                max_length = tokens_i.shape[0]

        # Pad sequences to max_length
        tokens = torch.zeros((B, max_length, self.token_dim), device=device)
        positions = torch.zeros((B, max_length, 8), device=device)
        modalities = torch.zeros((B, max_length), dtype=torch.long, device=device)

        for i in range(B):
            length = tokens_list[i].shape[0]
            tokens[i, :length, :] = tokens_list[i]
            positions[i, :length, :] = positions_list[i]
            modalities[i, :length] = modalities_list[i]

        if self.sort_spatially:
            self.indices = spatial_order(positions, modalities)
            tokens, positions, modalities = rearrange_tokens(
                tokens, positions, modalities, self.indices
            )
        return tokens, positions, modalities
