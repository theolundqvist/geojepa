from typing import Tuple

import torch
import torch.nn.functional as F

import torch.nn as nn
from torch import Tensor


class WeightedMSELoss(nn.Module):
    def __init__(self, zero_fraction=0.9):
        """
        Initializes the WeightedMSELoss.
        Args:
            zero_fraction (float): The fraction of zero-valued targets in the dataset.
                                    Must be between 0 and 1.
        """
        super(WeightedMSELoss, self).__init__()
        if not 0.0 < zero_fraction < 1.0:
            raise ValueError("zero_fraction must be between 0 and 1.")
        self.non_zero_weight = zero_fraction / (1.0 - zero_fraction)

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the weighted Mean Squared Error loss.
        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The ground truth target values.
        Returns:
            torch.Tensor: The computed loss.
        """
        weights = torch.where(targets != 0, self.non_zero_weight, 1.0).to(
            predictions.device
        )
        loss = weights * (predictions - targets) ** 2
        return loss.mean()


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 2
    dim = x.shape[0]
    mask = ~torch.eye(dim, dtype=torch.bool, device=x.device)
    return x[mask]


def vectorised_masked_vicreg_loss(
    x: torch.Tensor, padding_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes the VICReg loss.

    ---
    Args:
        x: Features map.
            Shape of [B, T, dim] or [T, dim].
        padding_mask: Features map.
            Shape of [B, T] or [T]. 1 - padding, 0 - valid

    ---
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of (std_loss, cov_loss)
    """
    x = x.float()
    if len(x.shape) == 3:  # (B, T, dim)
        x = x.permute(2, 0, 1)
        dim, B, T = x.shape
        x = x.reshape(dim, B * T)
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(B * T)
    else:
        B, dim = x.shape
        x = x.permute(1, 0)
        x = x.reshape(dim, B)

    if padding_mask is not None:
        x = x[:, padding_mask == 0]
    dim, N = x.shape

    # Compute standard deviation loss
    x = x - x.mean(dim=1, keepdim=True)  # subtract mean per feature across N samples
    std_x = torch.sqrt(x.var(dim=1, unbiased=False) + 1e-4)
    std_loss = torch.mean(F.relu(1 - std_x)) * 25

    # If not enough data to compute covariance meaningfully, just return std_loss
    if B < 2 or dim < 2:
        cov_loss = torch.tensor(0.0, device=x.device)
        return std_loss, cov_loss

    cov_x = torch.cov(x, correction=0)  # (dim, dim)
    # Convert population covariance (denominator N) to sample covariance (denominator N-1)
    cov_x *= float(N) / (N - 1)

    off_diag = off_diagonal(cov_x)
    cov_loss = off_diag.pow(2).sum().div(dim) * 1.0
    if not cov_loss.isfinite():
        cov_loss = torch.tensor(0.0, device=x.device)

    return std_loss, cov_loss


def vectorised_masked_smooth_l1_loss(pred, tgt, padding_mask, beta=1.0):
    """
    pred, tgt: [B, L, D]
    mask: [B, L] with 1-padding, 0-valid
    beta: beta scalar
    """
    mask = ~(padding_mask.bool())
    # Flatten B and L into a single dimension
    pred_flat = pred.view(-1, pred.size(-1))  # [B*L, D]
    tgt_flat = tgt.view(-1, tgt.size(-1))  # [B*L, D]
    mask_flat = mask.view(-1)  # [B*L]

    # Gather valid positions
    valid_pred = pred_flat[mask_flat]  # [T, D]
    valid_tgt = tgt_flat[mask_flat]  # [T, D]

    # If no valid positions, return zero
    if valid_pred.numel() == 0:
        return pred.new_tensor(0.0)

    # Smooth L1 loss over valid positions
    total_loss = torch.nn.functional.smooth_l1_loss(valid_pred, valid_tgt, beta=beta)

    return total_loss


def masked_avg_pairwise_sim(X, Y, m):
    total_pairs = torch.tensor(0.0, device=X.device)
    sim_sum = torch.tensor(0.0, device=X.device)
    for b in range(X.size(0)):
        x = X[b][m[b]]
        y = Y[b][m[b]]
        assert x.shape == y.shape
        if x.size(0) == 0:
            continue
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        sim_m = x @ y.T  # [T, T]
        sim_sum += sim_m.sum()
        total_pairs += sim_m.numel()
    if total_pairs < 1:
        return torch.tensor(0.0, device=X.device)
    return sim_sum / total_pairs


def masked_smooth_l1_loss(pred, tgt, mask, beta=1.0):
    """
    pred, tgt: [B, L, D]
    mask: [B, L] with True for valid tokens, False for pad
    beta: beta scalar
    """
    B, _, _ = pred.shape
    device = pred.device

    total_loss = torch.tensor(0.0, device=device)
    valid_tokens = 0

    for b in range(B):
        # --- 2. Gather valid pred/tgt
        sub_pred = pred[b][mask[b]]  # [T, D]
        sub_tgt = tgt[b][mask[b]]  # [T, D]

        T, D = sub_pred.shape

        if T == 0:
            # skip if no valid tokens at all
            continue

        valid_tokens += T * D
        total_loss += torch.nn.functional.smooth_l1_loss(
            sub_pred, sub_tgt, beta=beta, reduction="sum"
        )

    # --- 8. Average (or sum) over valid samples
    if valid_tokens > 0:
        total_loss /= valid_tokens

    return total_loss


def masked_info_nce_loss(pred, tgt, mask, tau=0.07):
    """
    pred, tgt: [B, L, D]
    mask: [B, L] with True for valid tokens, False for pad
    tau: temperature scalar
    """
    B, _, _ = pred.shape
    device = pred.device

    total_loss = torch.tensor(0.0, device=device)
    valid_tokens = 0

    for b in range(B):
        # --- 2. Gather valid pred/tgt
        sub_pred = pred[b][mask[b]]  # [T, D]
        sub_tgt = tgt[b][mask[b]]  # [T, D]

        T, D = sub_pred.shape

        if T == 0:
            # skip if no valid tokens at all
            continue
        if T == 1:
            # no contrastive loss, skip
            continue

        # --- 3. (Optional) subtract mean/std for each sample
        mu = sub_pred.mean(dim=0, keepdim=True)  # shape: [1, D]
        std = sub_pred.std(dim=0, keepdim=True)  # shape: [1, D]
        sub_pred = (sub_pred - mu) / (std + 1e-4)

        # --- 4. Normalize embeddings
        sub_pred = F.normalize(sub_pred, p=2, dim=-1)
        sub_tgt = F.normalize(sub_tgt, p=2, dim=-1)

        # --- 5. Compute pairwise similarities: [T, T]
        scores = (sub_pred @ sub_tgt.T) / tau  # [T, T]

        # --- 6. Construct labels = [0..T-1], meaning
        #         the "correct" (positive) for position p is p itself.
        labels = torch.arange(T, device=device, dtype=torch.long)

        # --- 7. Cross entropy for this sample
        #         shape: [T, T] -> [T],
        #         labels -> [T]
        loss = F.cross_entropy(scores, labels, reduction="sum") * (tau * 2)

        valid_tokens += T
        total_loss += loss

    # --- 8. Average over valid tokens
    if valid_tokens > 0:
        total_loss /= valid_tokens

    return total_loss


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction="mean", negative_mode="unpaired"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(
            query,
            positive_key,
            negative_keys,
            temperature=self.temperature,
            reduction=self.reduction,
            negative_mode=self.negative_mode,
        )


def info_nce(
    query,
    positive_key,
    negative_keys=None,
    temperature=0.1,
    reduction="mean",
    negative_mode="unpaired",
):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError("<query> must have 2 dimensions.")
    if positive_key.dim() != 2:
        raise ValueError("<positive_key> must have 2 dimensions.")
    if negative_keys is not None:
        if negative_mode == "unpaired" and negative_keys.dim() != 2:
            raise ValueError(
                "<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'."
            )
        if negative_mode == "paired" and negative_keys.dim() != 3:
            raise ValueError(
                "<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'."
            )

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError(
            "<query> and <positive_key> must must have the same number of samples."
        )
    if negative_keys is not None:
        if negative_mode == "paired" and len(query) != len(negative_keys):
            raise ValueError(
                "If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>."
            )

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError(
            "Vectors of <query> and <positive_key> should have the same number of components."
        )
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError(
                "Vectors of <query> and <negative_keys> should have the same number of components."
            )

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == "unpaired":
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == "paired":
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
