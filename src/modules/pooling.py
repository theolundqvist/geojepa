import torch
import torch.nn as nn


class AvgMaxPool(nn.Module):
    def __init__(self, top_k=20):
        super(AvgMaxPool, self).__init__()
        self.top_k = top_k

    def forward(self, features, mask):
        """
        Args:
            features: Tensor of shape (Batch, T, 256)
            mask: Tensor of shape (Batch, T) with 0 for padding
        Returns:
            Pooled tensor of shape (Batch, 256 * 4) -> [max, avg, top_k_avg, cls]
        """
        # Ensure padding_mask is of type bool
        padding_mask = ~(mask.bool())  # (B, T)

        # Max Pooling
        masked_features_max = features.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )  # (B, T, 256)
        max_pool, _ = masked_features_max.max(dim=1)  # (B, 256)
        # Replace -inf with 0 to avoid issues with min()
        max_pool = torch.where(
            max_pool == float("-inf"), torch.zeros_like(max_pool), max_pool
        )

        # Average Pooling
        masked_features_avg = features.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )  # (B, T, 256)
        sum_pool = masked_features_avg.sum(dim=1)  # (B, 256)
        # Compute the number of valid (non-padded) tokens per batch
        valid_counts = (~padding_mask).sum(dim=1).unsqueeze(-1).clamp(min=1)  # (B, 1)
        avg_pool = sum_pool / valid_counts  # (B, 256)

        # Top-k Average Pooling
        masked_features_topk = features.masked_fill(
            padding_mask.unsqueeze(-1), float("-inf")
        )  # (B, T, 256)
        # Get top k values along the temporal dimension
        k = min(self.top_k, masked_features_topk.size(1))
        topk_vals, _ = torch.topk(masked_features_topk, k, dim=1)  # (B, k, 256)
        # If there are fewer than k valid tokens, replace -inf with 0 before averaging
        topk_vals = torch.where(
            topk_vals == float("-inf"), torch.zeros_like(topk_vals), topk_vals
        )
        topk_avg_pool = topk_vals.mean(dim=1)  # (B, 256)

        pooled = torch.cat([max_pool, avg_pool, topk_avg_pool], dim=1)  # (B, 256 * 3)

        return pooled
