import math
from abc import abstractmethod, ABC

import torch
from typing import Tuple

from src.modules.tokenizer import Modality


class MaskingStrategy:
    def __init__(
        self,
        num_targets=4,
        min_context=0.1,
    ):
        self.num_targets = num_targets
        self.min_context = min_context
        self.device = None

    def __call__(
        self,
        tokens: torch.Tensor,  # Shape: (B, T, token_dim)
        positions: torch.Tensor,  # Shape: (B, T, 4, 2)
        modalities: torch.Tensor,  # Shape: (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a masking strategy to each sequence in the batch independently.
        Returns mask: (M, B, T) where M is the number of masks per token
        """
        self.device = tokens.device
        self.prepare()
        M = self.num_targets
        tgt_masks = []

        # context starts with all non-padding tokens
        valid_mask = modalities != 0  # Shape: (B, T)
        nbr_tokens = valid_mask.sum(dim=1)  # Shape: (B)
        ctx_mask = valid_mask.clone()

        for tgt in range(M):
            selected = self._select_target(
                tokens, positions, modalities, valid_mask, nbr_tokens
            )
            if selected.sum() == 0:
                continue
            # we then remove each target from the context
            ctx_mask &= ~selected
            tgt_masks.append(selected)

        tgt_masks = torch.stack(tgt_masks, dim=0)
        # if context is too small, add a random contiguous token sequence too it.
        ctx_mask = self._min_ctx(valid_mask, ctx_mask, nbr_tokens)
        if M == 1:
            tgt_masks = tgt_masks.squeeze(0)
        return ctx_mask, tgt_masks

    def _min_ctx(
        self,
        valid_mask: torch.Tensor,  # (B, T)
        ctx_mask: torch.Tensor,  # (B, T)
        nbr_tokens: torch.Tensor,
    ):
        B, T = valid_mask.shape
        # if ctx is too small, add some more tokens to it.

        ctx_fraction = ctx_mask.sum(dim=1) / nbr_tokens
        too_small_mask = ctx_fraction < self.min_context
        increase = (self.min_context - ctx_fraction) * too_small_mask
        nbr_tokens_to_add = (increase * nbr_tokens).ceil().long()

        start_idx = (
            torch.rand(B, device=self.device) * (nbr_tokens - nbr_tokens_to_add)
        ).long()
        start_idx = start_idx.unsqueeze(1)  # Shape: (B, 1)
        nbr_tokens_to_add = nbr_tokens_to_add.unsqueeze(1)  # Shape: (B, 1)
        end_idx = start_idx + nbr_tokens_to_add  # Shape: (B, 1)
        indices = (
            torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)
        )  # Shape: (B, T) repeat over batches without copies
        add_mask = (indices >= start_idx) & (indices < end_idx)  # Shape: (B, T)
        add_mask *= too_small_mask.unsqueeze(-1).expand(-1, T)
        return add_mask | ctx_mask

    @abstractmethod
    def _select_target(
        self,
        tokens: torch.Tensor,  # (B, T, token_dim)
        positions: torch.Tensor,  # (B, T, 4, 2)
        modalities: torch.Tensor,  # (B, T)
        valid_mask: torch.Tensor,  # (B, T)
        nbr_tokens: torch.Tensor,  # (B)
    ) -> torch.Tensor:
        pass

    def prepare(self):
        pass


class RandomMask(MaskingStrategy, ABC):
    def __init__(self, target_size: float = 0.2, num_targets=4, min_context=0.1):
        super().__init__(num_targets, min_context)
        self.target_size = target_size

    def _select_target(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        modalities: torch.Tensor,
        valid_mask: torch.Tensor,
        nbr_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = tokens.shape
        device = tokens.device

        # Determine the number of tokens to select per sample based on the ratio
        num_to_select = torch.clamp(
            (nbr_tokens * self.target_size).long(), min=1
        )  # Shape: (B,)

        # Generate random values and set invalid positions to -inf
        rand_values = torch.rand(B, T, device=device)  # Shape: (B, T)
        rand_values[~valid_mask] = float("-inf")

        # Find the maximum number of tokens to select across the batch
        max_num_to_select = num_to_select.max().item()

        # Use torch.topk to select the top 'max_num_to_select' tokens for each sample
        topk_values, topk_indices = torch.topk(rand_values, k=max_num_to_select, dim=1)

        # Create a mask to filter out extra tokens for samples with fewer tokens to select
        mask = torch.arange(max_num_to_select, device=device).unsqueeze(
            0
        ) < num_to_select.unsqueeze(1)

        # Select the valid indices based on the mask
        selected_indices = topk_indices[mask]

        # Create the target mask
        target_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        batch_indices = (
            torch.arange(B, device=device)
            .unsqueeze(1)
            .expand(B, max_num_to_select)[mask]
        )
        target_mask[batch_indices, selected_indices] = True

        return target_mask


class ContiguousMask(MaskingStrategy):
    def __init__(self, target_size: float = 0.2, num_targets=4, min_context=0.1):
        super().__init__(num_targets, min_context)
        self.target_size = target_size

    def _select_target(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        modalities: torch.Tensor,
        valid_mask: torch.Tensor,
        nbr_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = tokens.shape
        d = tokens.device

        tgt_num_tokens = self.target_size * nbr_tokens
        nbr_tokens_to_add = torch.clamp(tgt_num_tokens.long(), 1)
        start_idx = (torch.rand(B, device=d) * (nbr_tokens - tgt_num_tokens)).long()
        start_idx = start_idx.unsqueeze(1)  # Shape: (B, 1)
        nbr_tokens_to_add = nbr_tokens_to_add.unsqueeze(1)  # Shape: (B, 1)

        end_idx = start_idx + nbr_tokens_to_add  # Shape: (B, 1)
        indices = (
            torch.arange(T, device=d).unsqueeze(0).expand(B, T)
        )  # Shape: (B, T) repeat over batches without copies
        tgt_mask = (indices >= start_idx) & (indices < end_idx)  # Shape: (B, T)
        return tgt_mask


class ModalityMask(MaskingStrategy):
    def __init__(self, min_context=0.1):
        super().__init__(0, min_context)

    def __call__(
        self,
        tokens: torch.Tensor,  # Shape: (B, T, token_dim)
        positions: torch.Tensor,  # Shape: (B, T, 4, 2)
        modalities: torch.Tensor,  # Shape: (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a masking strategy to each sequence in the batch independently.
        Returns mask: (M, B, T) where M is the number of masks per token
        """
        B, T, C = tokens.size()
        mods = modalities.unique()
        self.device = tokens.device
        mods = mods[(mods != Modality.PAD)]  # ignore padding mod and class token
        ctx_mod = mods[torch.randperm(mods.size(0), device=self.device)[0]]
        tgt_mods = mods[mods != ctx_mod]
        M = tgt_mods.numel()
        if M == 0:
            return ContiguousMask()(tokens, positions, modalities)
        tgt_masks = torch.zeros((M, B, T), dtype=torch.bool, device=self.device)

        # context starts with all non-padding tokens
        valid_mask = modalities != 0  # Shape: (B, T)
        nbr_tokens = valid_mask.sum(dim=1)  # Shape: (B)

        ctx_mask = modalities == ctx_mod

        for i, tgt_mod in enumerate(tgt_mods):
            tgt_masks[i] = modalities == tgt_mod

        # if context is too small, add a random contiguous token sequence too it.
        ctx_mask = self._min_ctx(valid_mask, ctx_mask, nbr_tokens)
        return ctx_mask, tgt_masks


class AreaMask(MaskingStrategy):
    def __init__(
        self,
        target_size: float = 0.2,
        min_ar=0.66,
        max_ar=1.5,
        num_targets=4,
        min_context=0.1,
    ):
        super().__init__(num_targets, min_context)
        self.target_size = target_size
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.box_side = math.sqrt(
            target_size
        )  # Not directly used, but kept for consistency
        self.fallback = RandomMask(target_size, num_targets, min_context)

    def sample_target_box(self, B: int) -> torch.Tensor:
        """
        Samples B random bounding boxes within [0.0, 1.0].

        Returns:
            boxes: Tensor of shape (B, 4) where each box is (x_min, y_min, x_max, y_max)
        """
        ar = (
            torch.rand(B, device=self.device) * (self.max_ar - self.min_ar)
            + self.min_ar
        )
        height = math.sqrt(self.target_size) / ar
        width = self.target_size / height

        # Ensure width and height do not exceed 1.0
        width = torch.clamp(width, max=1.0)
        height = torch.clamp(height, max=1.0)

        # Sample top-left corner such that the box fits within [0.0, 1.0]
        x_min = torch.rand(B, device=self.device) * (1.2 - width) - 0.1
        y_min = torch.rand(B, device=self.device) * (1.2 - height) - 0.1

        x_max = x_min + width
        y_max = y_min + height

        boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)  # Shape: (B, 4)
        return boxes

    def _select_target(
        self,
        tokens: torch.Tensor,  # (B, T, token_dim)
        positions: torch.Tensor,  # (B, T, 4, 2)
        modalities: torch.Tensor,  # (B, T)
        valid_mask: torch.Tensor,  # (B, T)
        nbr_tokens: torch.Tensor,  # (B)
    ) -> torch.Tensor:
        B, T, C = tokens.shape

        # Sample B bounding boxes
        mask_boxes = self.sample_target_box(B)  # (B, 4)

        # Compute centroids of tokens
        bboxes = positions.view(B, T, 4, 2)
        centroids = bboxes.mean(dim=2)  # (B, T, 2)
        x_corner = bboxes[..., 0]
        y_corner = bboxes[..., 1]

        # Extract coordinates
        x_centroid = centroids[..., 0]  # (B, T)
        y_centroid = centroids[..., 1]  # (B, T)

        x_min = mask_boxes[:, 0].unsqueeze(1)  # (B, 1)
        y_min = mask_boxes[:, 1].unsqueeze(1)  # (B, 1)
        x_max = mask_boxes[:, 2].unsqueeze(1)  # (B, 1)
        y_max = mask_boxes[:, 3].unsqueeze(1)  # (B, 1)

        # Determine which centroids are within the bounding boxes
        within_x = (x_centroid >= x_min) & (x_centroid <= x_max)  # (B, T)
        within_y = (y_centroid >= y_min) & (y_centroid <= y_max)  # (B, T)

        # corner_within_x = (x_corner >= x_min) & (x_corner <= x_max)
        # corner_within_y = (y_corner >= y_min) & (y_corner <= y_max)
        # corners_within = torch.sum((corner_within_x & corner_within_y), dim=-1) > 1.9

        tgt_mask = within_x & within_y  # (B, T)

        # Ensure that only valid tokens are masked
        tgt_mask &= valid_mask
        if tgt_mask.sum() == 0:
            return self.fallback._select_target(
                tokens, positions, modalities, valid_mask, nbr_tokens
            )

        return tgt_mask


def apply_mask(
    tensor: torch.Tensor, mask: torch.Tensor, padding_value: float = 0
) -> torch.Tensor:
    """
    Removes masked tokens and pads minimally to maintain a consistent matrix shape.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, T, *], where
                               B = Batch size,
                               T = Sequence length (number of tokens),
                               * = Any number of trailing dimensions (e.g., C, 8).
        mask (torch.Tensor): Binary mask tensor of shape [B, T], where
                             1 indicates a valid token and 0 indicates a masked token.
        padding_value (float, optional): Value to use for padding. Defaults to 0.

    Returns:
        torch.Tensor: Padded tensor of shape [B, X, *], where X is the maximum
                      number of unmasked tokens across the batch.
    """
    tensor = tensor.clone()
    B, T = mask.shape
    trailing_dims = tensor.shape[2:]  # Capture any trailing dimensions (e.g., C, 8)

    # Step 1: Determine X (maximum number of unmasked tokens in the batch)
    X = mask.sum(dim=1).max().item()

    # Step 2: Sort the mask in descending order to bring unmasked tokens to the front
    # sorted_mask, sorted_indices = mask.sort(dim=1, descending=True)  # Both [B, T]
    sorted_indices = mask.int().argsort(dim=1, descending=True)

    # Step 3: Expand sorted indices to match the tensor's trailing dimensions for gathering
    if trailing_dims:
        # Create a shape that can be broadcasted for gather
        sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(
            -1, -1, *trailing_dims
        )  # [B, T, *]
    else:
        sorted_indices_expanded = sorted_indices  # [B, T]

    # Step 4: Gather the tokens based on sorted indices
    sorted_tensor = torch.gather(
        tensor, dim=1, index=sorted_indices_expanded
    )  # [B, T, *]

    # Step 5: Truncate the sorted tensor to the first X tokens
    tokens_trimmed = sorted_tensor[:, :X]  # [B, X, *] or [B, X] if * is empty

    # Step 6: Create a mask for valid tokens within the first X positions
    # Generate a range tensor [0, 1, 2, ..., X-1] and compare with mask sums
    range_tensor = (
        torch.arange(X, device=mask.device).unsqueeze(0).expand(B, X)
    )  # [B, X]

    # valid_mask: [B, X], True if the position is within the valid token count
    valid_mask = range_tensor < mask.sum(dim=1).unsqueeze(1)  # [B, X]

    if trailing_dims:
        # Expand valid_mask to match the trailing dimensions
        # Example: [B, X] -> [B, X, 1, 1, ...] and then broadcast
        valid_mask = valid_mask.view(
            B, X, *([1] * len(trailing_dims))
        )  # [B, X, 1, 1, ...]
        valid_mask = valid_mask.expand(-1, -1, *trailing_dims)  # [B, X, *]
    # else: valid_mask remains [B, X]

    # Step 7: Apply the padding value to positions beyond the valid tokens
    tokens_padded = tokens_trimmed.masked_fill(~valid_mask, padding_value)  # [B, X, *]

    return tokens_padded
