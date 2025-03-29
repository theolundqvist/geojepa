# tests/test_random_mask.py

import pytest
import torch

from torch import Tensor

# Import the RandomMask class from its module
from src.modules.masks import RandomMask


@pytest.fixture
def random_mask():
    """Fixture to instantiate the RandomMask class."""
    return RandomMask()


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 100, 64, (4, 2)),
        (2, 50, 128, (4, 2)),
        (4, 200, 256, (4, 2)),
    ],
)
def test_mask_percentage(
    random_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """Test that approximately 20% of the tokens are masked."""
    torch.manual_seed(42)  # Set seed for reproducibility
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 10, (batch_size, sequence_length))

    target_indices, context_indices = random_mask(tokens, bboxes, modalities)

    expected_target_size = int(sequence_length * random_mask.mask_fraction)
    expected_context_size = sequence_length - expected_target_size

    assert target_indices.shape == (batch_size, expected_target_size), (
        f"Expected target_indices shape {(batch_size, expected_target_size)}, "
        f"got {target_indices.shape}."
    )
    assert context_indices.shape == (batch_size, expected_context_size), (
        f"Expected context_indices shape {(batch_size, expected_context_size)}, "
        f"got {context_indices.shape}."
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 50, 64, (4, 2)),
        (3, 75, 128, (4, 2)),
        (5, 150, 256, (4, 2)),
    ],
)
def test_no_overlap(random_mask, batch_size, sequence_length, token_dim, bbox_dims):
    """Ensure there is no overlap between target_indices and context_indices."""
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 5, (batch_size, sequence_length))

    target_indices, context_indices = random_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        target_set = set(target_indices[i].tolist())
        context_set = set(context_indices[i].tolist())
        intersection = target_set.intersection(context_set)
        assert len(intersection) == 0, (
            f"Batch {i}: Overlapping indices found: {intersection}."
        )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 30, 64, (4, 2)),
        (2, 60, 128, (4, 2)),
        (4, 120, 256, (4, 2)),
    ],
)
def test_complete_coverage(
    random_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """Confirm that all original indices are present in either target or context indices."""
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 3, (batch_size, sequence_length))

    target_indices, context_indices = random_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        combined_indices = torch.cat((target_indices[i], context_indices[i]))
        combined_sorted, _ = torch.sort(combined_indices)
        expected_indices = torch.arange(0, sequence_length)
        assert torch.equal(combined_sorted, expected_indices), (
            f"Batch {i}: Not all indices are covered."
        )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims, seed",
    [
        (1, 40, 64, (4, 2), 123),
        (2, 80, 128, (4, 2), 456),
        (3, 160, 256, (4, 2), 789),
    ],
)
def test_reproducibility(
    random_mask, batch_size, sequence_length, token_dim, bbox_dims, seed
):
    """Check that masking is deterministic when a fixed random seed is set."""
    # First run
    torch.manual_seed(seed)
    tokens1 = torch.randn(batch_size, sequence_length, token_dim)
    bboxes1 = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities1 = torch.randint(0, 7, (batch_size, sequence_length))

    target_indices1, context_indices1 = random_mask(tokens1, bboxes1, modalities1)

    # Second run with the same seed
    torch.manual_seed(seed)
    tokens2 = torch.randn(batch_size, sequence_length, token_dim)
    bboxes2 = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities2 = torch.randint(0, 7, (batch_size, sequence_length))

    target_indices2, context_indices2 = random_mask(tokens2, bboxes2, modalities2)

    assert torch.equal(target_indices1, target_indices2), (
        "Target indices should be identical for the same seed across the entire batch."
    )
    assert torch.equal(context_indices1, context_indices2), (
        "Context indices should be identical for the same seed across the entire batch."
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 0, 64, (4, 2)),
        (2, 0, 128, (4, 2)),
    ],
)
def test_empty_tokens(random_mask, batch_size, sequence_length, token_dim, bbox_dims):
    """Handle the edge case where the input tokens tensor is empty."""
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 2, (batch_size, sequence_length))

    target_indices, context_indices = random_mask(tokens, bboxes, modalities)

    assert target_indices.shape == (batch_size, 0), (
        f"target_indices should have shape ({batch_size}, 0), got {target_indices.shape}."
    )
    assert context_indices.shape == (batch_size, 0), (
        f"context_indices should have shape ({batch_size}, 0), got {context_indices.shape}."
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 3, 64, (4, 2)),
        (2, 4, 128, (4, 2)),
        (3, 5, 256, (4, 2)),
    ],
)
def test_small_token_size(
    random_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """Test behavior when the number of tokens is too small to apply the masking percentage meaningfully."""
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 2, (batch_size, sequence_length))

    target_indices, context_indices = random_mask(tokens, bboxes, modalities)

    expected_target_size = int(sequence_length * random_mask.mask_fraction)
    expected_context_size = sequence_length - expected_target_size

    assert target_indices.shape == (batch_size, expected_target_size), (
        f"Expected target_indices shape {(batch_size, expected_target_size)}, "
        f"got {target_indices.shape}."
    )
    assert context_indices.shape == (batch_size, expected_context_size), (
        f"Expected context_indices shape {(batch_size, expected_context_size)}, "
        f"got {context_indices.shape}."
    )


from src.modules.masks import ModalityMask


@pytest.fixture
def random_modality_mask():
    """Fixture to instantiate the ModalityMask class."""
    return ModalityMask()


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 100, 64, (4, 2)),
        (2, 50, 128, (4, 2)),
        (4, 200, 256, (4, 2)),
    ],
)
def test_modality_masking(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Test that all target indices correspond to the selected modality.
    """
    torch.manual_seed(42)  # Set seed for reproducibility
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(
        0, 4, (batch_size, sequence_length)
    )  # Modalities between 0-3

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        if sequence_length == 0:
            assert target_indices[i].numel() == 0
            assert context_indices[i].numel() == 0
            continue

        # Retrieve the selected modality by checking one of the target indices
        if target_indices[i].numel() == 0:
            continue  # No target indices, possibly no modalities
        selected_modality = modalities[i, target_indices[i][0]].item()

        # Verify all target indices have the selected modality
        assert torch.all(modalities[i, target_indices[i]] == selected_modality), (
            f"Batch {i}: Not all target indices have the selected modality {selected_modality}."
        )

        # Verify context indices do not have the selected modality
        if context_indices[i].numel() > 0:
            assert torch.all(modalities[i, context_indices[i]] != selected_modality), (
                f"Batch {i}: Some context indices have the selected modality {selected_modality}."
            )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 50, 64, (4, 2)),
        (3, 75, 128, (4, 2)),
        (5, 150, 256, (4, 2)),
    ],
)
def test_no_overlap(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Ensure there is no overlap between target_indices and context_indices.
    """
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        target_set = set(target_indices[i].tolist())
        context_set = set(context_indices[i].tolist())
        print(target_set, context_set)
        intersection = target_set.intersection(context_set)
        assert len(intersection) == 0, (
            f"Batch {i}: Overlapping indices found: {intersection}."
        )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 30, 64, (4, 2)),
        (2, 60, 128, (4, 2)),
        (4, 120, 256, (4, 2)),
    ],
)
def test_complete_coverage(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Confirm that all original indices are present in either target or context indices.
    """
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        combined_indices = torch.cat((target_indices[i], context_indices[i]))
        combined_sorted, _ = torch.sort(combined_indices)
        expected_indices = torch.arange(
            0, sequence_length, device=combined_sorted.device
        )
        # Remove padding (-1) from combined_sorted
        valid_indices = combined_sorted[combined_sorted != 0]
        assert torch.equal(valid_indices, expected_indices), (
            f"Batch {i}: Not all indices are covered."
        )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims, seed",
    [
        (1, 40, 64, (4, 2), 123),
        (2, 80, 128, (4, 2), 456),
        (3, 160, 256, (4, 2), 789),
    ],
)
def test_reproducibility(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims, seed
):
    """
    Check that masking is deterministic when a fixed random seed is set.
    """
    # First run
    torch.manual_seed(seed)
    tokens1 = torch.randn(batch_size, sequence_length, token_dim)
    bboxes1 = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities1 = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices1, context_indices1 = random_modality_mask(
        tokens1, bboxes1, modalities1
    )

    # Second run with the same seed and same inputs
    torch.manual_seed(seed)
    tokens2 = torch.randn(batch_size, sequence_length, token_dim)
    bboxes2 = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities2 = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices2, context_indices2 = random_modality_mask(
        tokens2, bboxes2, modalities2
    )

    assert torch.equal(target_indices1, target_indices2), (
        "Target indices should be identical for the same seed across the entire batch."
    )
    assert torch.equal(context_indices1, context_indices2), (
        "Context indices should be identical for the same seed across the entire batch."
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 0, 64, (4, 2)),
        (2, 0, 128, (4, 2)),
    ],
)
def test_empty_tokens(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Handle the edge case where the input tokens tensor is empty.
    """
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    modalities = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    assert target_indices.shape == (batch_size, 0), (
        f"target_indices should have shape ({batch_size}, 0), got {target_indices.shape}."
    )
    assert context_indices.shape == (batch_size, 0), (
        f"context_indices should have shape ({batch_size}, 0), got {context_indices.shape}."
    )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 3, 64, (4, 2)),
        (2, 4, 128, (4, 2)),
        (3, 5, 256, (4, 2)),
    ],
)
def test_single_modality(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Test behavior when only one modality is present in the sequence.
    All tokens should be masked as target indices.
    """
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    # All tokens have the same modality, e.g., 2
    modalities = torch.full((batch_size, sequence_length), 2, dtype=torch.long)

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        if sequence_length == 0:
            assert target_indices[i].numel() == 0
            assert context_indices[i].numel() == 0
            continue

        # All tokens should be in target_indices
        assert target_indices[i].numel() == sequence_length, (
            f"Batch {i}: All tokens should be masked, expected {sequence_length} target indices."
        )
        assert context_indices[i].numel() == 0, (
            f"Batch {i}: No context indices should be present."
        )


@pytest.mark.parametrize(
    "batch_size, sequence_length, token_dim, bbox_dims",
    [
        (1, 10, 64, (4, 2)),
        (2, 20, 128, (4, 2)),
        (3, 30, 256, (4, 2)),
    ],
)
def test_multiple_modalities(
    random_modality_mask, batch_size, sequence_length, token_dim, bbox_dims
):
    """
    Test sequences with multiple modalities to ensure correct masking.
    """
    torch.manual_seed(42)
    tokens = torch.randn(batch_size, sequence_length, token_dim)
    bboxes = torch.randn(batch_size, sequence_length, *bbox_dims)
    # Ensure multiple modalities are present
    modalities = torch.randint(0, 4, (batch_size, sequence_length))

    target_indices, context_indices = random_modality_mask(tokens, bboxes, modalities)

    for i in range(batch_size):
        if sequence_length == 0:
            assert target_indices[i].numel() == 0
            assert context_indices[i].numel() == 0
            continue

        unique_modalities = torch.unique(modalities[i])
        if unique_modalities.numel() == 0:
            assert target_indices[i].numel() == 0
            assert context_indices[i].numel() == 0
            continue

        # Retrieve the selected modality by checking one of the target indices
        if target_indices[i].numel() == 0:
            continue  # No target indices, possibly no modalities
        selected_modality = modalities[i, target_indices[i][0]].item()
        print("mods", modalities[i])
        print("selected_mod", selected_modality)

        # Verify all target indices have the selected modality
        target_modalities = modalities[i, target_indices[i]]
        print("target", target_modalities)
        m: Tensor = (target_modalities == selected_modality) | (target_modalities == 0)
        assert torch.all(m), (
            f"Batch {i}: Not all target indices have the selected modality {selected_modality}."
        )

        # Verify context indices do not have the selected modality
        ctx_mods = modalities[i, context_indices[i]]
        if context_indices[i].numel() > 0:
            print("context", ctx_mods)
            assert torch.any((ctx_mods == selected_modality)), (
                f"Batch {i}: Some context indices have the selected modality {selected_modality}."
            )
