import torch
# from hilbertcurve.hilbertcurve import HilbertCurve



def compute_morton_code(x, y, num_bits=16):
    """
    Compute Morton (Z-order) codes for given x and y coordinates.

    Args:
        x (torch.Tensor): Tensor of x coordinates, shape (B, T).
        y (torch.Tensor): Tensor of y coordinates, shape (B, T).
        num_bits (int): Number of bits to represent each coordinate (default: 16).

    Returns:
        torch.Tensor: Morton codes, shape (B, T), dtype=torch.long.
    """

    def split_by_1bits(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    # Ensure coordinates are integers
    x = x.long()
    y = y.long()

    # Mask to fit within num_bits
    mask = (1 << num_bits) - 1
    x = x & mask
    y = y & mask

    # Split bits and interleave
    x_split = split_by_1bits(x)
    y_split = split_by_1bits(y)

    morton_code = (y_split << 1) | x_split  # Interleave y and x bits

    return morton_code


def compute_morton_order(x, y, num_bits=16):
    """
    Compute Morton order for given x and y coordinates.

    Args:
        x (torch.Tensor): Tensor of x coordinates, shape (B, T)
        y (torch.Tensor): Tensor of y coordinates, shape (B, T)
        num_bits (int): Number of bits to represent each coordinate.

    Returns:
        torch.Tensor: Morton order values, shape (B, T)
    """
    B, T = x.shape

    # Normalize coordinates to fit within the grid defined by num_bits
    max_val_x = x.max(dim=1, keepdim=True)[0].float()
    max_val_y = y.max(dim=1, keepdim=True)[0].float()

    # Avoid division by zero
    max_val_x = max_val_x.clamp(min=1)
    max_val_y = max_val_y.clamp(min=1)

    x_norm = (x.float() / max_val_x) * ((1 << num_bits) - 1)
    y_norm = (y.float() / max_val_y) * ((1 << num_bits) - 1)

    # Clamp to ensure coordinates are within the grid
    x_norm = x_norm.clamp(0, (1 << num_bits) - 1).long()
    y_norm = y_norm.clamp(0, (1 << num_bits) - 1).long()

    # Compute Morton codes
    morton_codes = compute_morton_code(x_norm, y_norm, num_bits=num_bits)

    return morton_codes


def compute_hilbert_order(x, y, p=16, n=2):
    """
    Compute Hilbert order for given x and y coordinates.

    Args:
        x (torch.Tensor): Tensor of x coordinates, shape (B, T)
        y (torch.Tensor): Tensor of y coordinates, shape (B, T)
        p (int): Number of iterations (defines the size of the grid as 2^p)
        n (int): Number of dimensions (2 for 2D)

    Returns:
        torch.Tensor: Hilbert order values, shape (B, T)
    """
    B, T = x.shape
    hilbert = HilbertCurve(p, n)

    # Normalize coordinates to fit the Hilbert curve grid
    max_val_x = x.max(dim=1, keepdim=True)[0].float()
    max_val_y = y.max(dim=1, keepdim=True)[0].float()

    # Avoid division by zero
    max_val_x = max_val_x.clamp(min=1)
    max_val_y = max_val_y.clamp(min=1)

    x_norm = (x.float() / max_val_x) * (2**p - 1)
    y_norm = (y.float() / max_val_y) * (2**p - 1)

    # Clamp to ensure coordinates are within the grid
    x_norm = x_norm.clamp(0, 2**p - 1).long()
    y_norm = y_norm.clamp(0, 2**p - 1).long()

    # Stack and reshape for processing
    points = torch.stack([x_norm, y_norm], dim=-1).reshape(-1, 2)

    # Compute Hilbert distances
    hilbert_indices = hilbert.distance_from_points(points)

    # Reshape back to (B, T)
    hilbert_tensor = hilbert_indices.reshape(B, T)

    return hilbert_tensor


def spatial_order(positions, modalities, num_bits=16) -> torch.Tensor:
    """
    Generate sort indices based on Morton order for given positions,
    ensuring that padding tokens (modalities == 0) are sorted to the end.

    Args:
        positions (torch.Tensor): Bounding boxes, shape (B, T, 8).
        modalities (torch.Tensor): Modalities tensor, shape (B, T).
        num_bits (int): Number of bits to represent each coordinate for Morton order.

    Returns:
        torch.Tensor: Sort indices, shape (B, T).
    """
    B, T, _ = positions.shape
    bbox = positions.reshape(B, T, 4, 2)

    # Compute Bounding Box Centers
    bbox_centers = bbox.mean(dim=2)  # Shape: (B, T, 2)
    x = bbox_centers[:, :, 0]  # Shape: (B, T)
    y = bbox_centers[:, :, 1]  # Shape: (B, T)

    # Compute Morton codes
    order = compute_morton_order(x, y, num_bits=num_bits)  # Shape: (B, T)

    # Create a padding mask where modalities == 0 (True for padding tokens)
    padding_mask = modalities == 0  # Shape: (B, T)

    # Assign a large value to padding tokens to sort them to the end
    large_value = order.max(dim=1, keepdim=True)[0] + 1  # Shape: (B, 1)
    order = torch.where(padding_mask, large_value, order)

    # Compute sort indices
    sort_indices = order.argsort(dim=1)

    return sort_indices  # Shape: (B, T)


def rearrange_tokens(tokens, positions, modalities, indices):
    """
    Sort tokens and associated tensors based on provided indices.

    Args:
        tokens (torch.Tensor): Token embeddings, shape (B, T, C).
        positions (torch.Tensor): Bounding boxes, shape (B, T, 8).
        modalities (torch.Tensor): Modalities tensor, shape (B, T).
        indices (torch.Tensor): Indices to sort by, shape (B, T).

    Returns:
        tuple: Sorted tokens, sorted bounding boxes, sorted modalities.
    """
    B, T, C = tokens.shape  # Corrected: Extract C from tokens

    # Sort tokens
    sort_indices = indices.unsqueeze(-1).expand(-1, -1, C)
    tokens_sorted = tokens.gather(1, sort_indices)  # Shape: (B, T, C)

    # Sort bounding boxes
    # positions shape: (B, T, 8)
    sort_indices_expanded = indices.unsqueeze(-1).expand(-1, -1, 8)
    positions_sorted = positions.gather(1, sort_indices_expanded)  # Shape: (B, T, 8)

    # Sort modalities
    modalities_sorted = modalities.gather(1, indices)  # Shape: (B, T)

    return tokens_sorted, positions_sorted, modalities_sorted


def restore_tensor_order(tensor, sort_indices, dim=1):
    """
    Restore the original order of a tensor based on the sort_indices used during sorting along a specified dimension.

    Args:
        tensor (torch.Tensor): The sorted tensor to be restored.
        sort_indices (torch.Tensor): Indices used to sort the tensor, with the same shape as `tensor` except at `dim`.
        dim (int): The dimension along which the sorting was performed. Default is 1.

    Returns:
        torch.Tensor: The tensor with the original ordering restored.
    """
    # Adjust negative dimensions
    if dim < 0:
        dim += tensor.dim()

    # Validate the dimension
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(
            f"dim={dim} is out of range for tensor with {tensor.dim()} dimensions"
        )

    # Size along the sorting dimension
    sorted_dim_size = tensor.size(dim)

    # Create a shape list with 1s, except for the sorting dimension
    # This is for broadcasting the original indices
    shape = [1] * sort_indices.dim()
    shape[dim] = sorted_dim_size

    # Generate original indices and reshape for broadcasting
    original_idx = torch.arange(sorted_dim_size, device=sort_indices.device).view(shape)

    # Expand original_idx to match sort_indices
    original_idx = original_idx.expand_as(sort_indices)

    # Compute inverse permutation using scatter
    inverse_indices = torch.empty_like(sort_indices)
    inverse_indices = inverse_indices.scatter(dim, sort_indices, original_idx)

    # Automatically expand inverse_indices to match tensor's dimensions
    for _ in range(tensor.dim() - sort_indices.dim()):
        inverse_indices = inverse_indices.unsqueeze(-1)

    # Expand to match tensor's shape
    inverse_indices = inverse_indices.expand_as(tensor)

    # Restore the original order by gathering with inverse_indices
    restored_tensor = torch.gather(tensor, dim, inverse_indices)

    return restored_tensor
