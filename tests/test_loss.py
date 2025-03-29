
import pytest

from src.modules.losses import (
    WeightedMSELoss,
    vectorised_masked_vicreg_loss,
    vectorised_masked_smooth_l1_loss,
)

# test_weighted_mse_loss.py

import torch


@pytest.fixture
def set_seed():
    """Fixture to set the random seed for reproducibility."""
    torch.manual_seed(0)


def test_weighted_mse_loss_high_zero_fraction(set_seed):
    """
    Test WeightedMSELoss with a high zero fraction (e.g., 0.95).
    Ensures that non-zero targets are appropriately weighted.
    """
    zero_fraction = 0.95
    loss_fn = WeightedMSELoss(zero_fraction=zero_fraction)

    # Verify non_zero_weight calculation
    expected_non_zero_weight = zero_fraction / (
        1.0 - zero_fraction
    )  # 0.95 / 0.05 = 19.0
    assert loss_fn.non_zero_weight == expected_non_zero_weight, (
        f"Expected non_zero_weight to be {expected_non_zero_weight}, but got {loss_fn.non_zero_weight}"
    )

    # Create synthetic data
    batch_size = 100
    num_non_zero = int(batch_size * (1 - zero_fraction))  # 5
    num_zero = batch_size - num_non_zero  # 95

    # Targets: first 5 are 1.0, rest are 0.0
    targets = torch.zeros(batch_size)
    targets[:num_non_zero] = 1.0

    # Predictions: first 5 are 2.0, rest are 0.0
    predictions = torch.zeros(batch_size)
    predictions[:num_non_zero] = 2.0

    # Expected weights: first 5 are 19.0, rest are 1.0
    expected_weights = torch.ones(batch_size)
    expected_weights[:num_non_zero] = expected_non_zero_weight

    # Compute expected loss manually
    squared_errors = (predictions - targets) ** 2  # 1.0 for non-zero, 0.0 for zero
    weighted_squared_errors = (
        expected_weights * squared_errors
    )  # 19.0 for non-zero, 0.0 for zero
    expected_loss = weighted_squared_errors.mean()  # (5 * 19) / 100 = 95 / 100 = 0.95

    # Compute loss using WeightedMSELoss
    computed_loss = loss_fn(predictions, targets)

    # Assert that the computed loss matches the expected loss
    assert torch.isclose(computed_loss, torch.tensor(expected_loss), atol=1e-6), (
        f"Computed loss {computed_loss.item()} does not match expected loss {expected_loss}."
    )


def test_weighted_mse_loss_all_zero():
    """
    Test that initializing WeightedMSELoss with zero_fraction=1.0 raises a ValueError.
    """
    with pytest.raises(ValueError, match="zero_fraction must be between 0 and 1."):
        WeightedMSELoss(zero_fraction=1.0)


def test_weighted_mse_loss_no_zero():
    """
    Test that initializing WeightedMSELoss with zero_fraction=0.0 raises a ValueError.
    """
    with pytest.raises(ValueError, match="zero_fraction must be between 0 and 1."):
        WeightedMSELoss(zero_fraction=0.0)


def test_weighted_mse_loss_mixed_values():
    """
    Test WeightedMSELoss with mixed zero and non-zero targets and varied predictions.
    """
    zero_fraction = 0.8
    loss_fn = WeightedMSELoss(zero_fraction=zero_fraction)

    # Verify non_zero_weight calculation
    expected_non_zero_weight = zero_fraction / (1.0 - zero_fraction)  # 0.8 / 0.2 = 4.0
    assert loss_fn.non_zero_weight == expected_non_zero_weight, (
        f"Expected non_zero_weight to be {expected_non_zero_weight}, but got {loss_fn.non_zero_weight}"
    )

    # Create synthetic data
    batch_size = 10
    targets = torch.tensor([0, 1, 0, 2, 0, 0, 3, 0, 0, 4], dtype=torch.float32)
    predictions = torch.tensor(
        [0, 1.5, 0, 1.8, 0, 0, 2.5, 0, 0, 3.5], dtype=torch.float32
    )

    # Expected weights: 1.0 for zeros, 4.0 for non-zeros
    expected_weights = torch.ones(batch_size)
    expected_weights[targets != 0] = expected_non_zero_weight

    # Compute expected loss manually
    squared_errors = (predictions - targets) ** 2
    weighted_squared_errors = expected_weights * squared_errors
    expected_loss = weighted_squared_errors.mean()

    # Compute loss using WeightedMSELoss
    computed_loss = loss_fn(predictions, targets)

    # Assert that the computed loss matches the expected loss
    assert torch.isclose(computed_loss, torch.tensor(expected_loss), atol=1e-6), (
        f"Computed loss {computed_loss.item()} does not match expected loss {expected_loss}."
    )


# test_losses.py
import pytest
import torch.nn.functional as F


#
# ------------------- TESTS FOR vectorised_masked_vicreg_loss -------------------
#


def test_vicreg_loss_no_padding_2d_input():
    """Test VICReg with a simple 2D [B, dim] input (no padding)."""
    torch.manual_seed(42)
    B, dim = 5, 3
    x = torch.randn(B, dim)  # shape [5, 3]

    std_loss, cov_loss = vectorised_masked_vicreg_loss(x, padding_mask=None)

    # Basic sanity checks
    assert std_loss.shape == ()
    assert cov_loss.shape == ()
    assert std_loss >= 0
    assert cov_loss >= 0  # typically non-negative in VICReg


def test_vicreg_loss_no_padding_3d_input():
    """Test VICReg with a 3D [B, T, dim] input (no padding)."""
    torch.manual_seed(42)
    B, T, dim = 4, 6, 3
    x = torch.randn(B, T, dim)  # shape [4, 6, 3]

    std_loss, cov_loss = vectorised_masked_vicreg_loss(x, padding_mask=None)

    assert std_loss.shape == ()
    assert cov_loss.shape == ()
    assert std_loss >= 0
    assert cov_loss >= 0


def test_vicreg_loss_with_padding():
    """Test VICReg with a 3D [B, T, dim] input with some padding."""
    torch.manual_seed(42)
    B, T, dim = 3, 5, 2
    x = torch.randn(B, T, dim)

    # Let padding_mask = 1 means "padding"
    # We'll pad the last 2 time steps for each batch
    padding_mask = torch.zeros(B, T)
    padding_mask[:, -2:] = 1  # last two positions are padded

    std_loss, cov_loss = vectorised_masked_vicreg_loss(x, padding_mask)

    assert std_loss.shape == ()
    assert cov_loss.shape == ()
    # We removed 2 time steps per batch, so effectively N = B*(T-2).
    # Should still be valid for VICReg, but the covariance might change from no-padding version.
    assert std_loss >= 0
    assert cov_loss >= 0


def test_vicreg_loss_all_padding():
    """If everything is padded, effectively no valid samples -> cov_loss = 0, std_loss is computed on empty?
    The function might return a standard deviation with no valid samples.
    In the user code, if B < 2 or dim < 2 or no valid samples, cov_loss = 0.
    Let's see the behavior for standard deviation loss as well.
    """
    torch.manual_seed(42)
    B, T, dim = 3, 4, 2
    x = torch.randn(B, T, dim)

    # Everything is padding
    padding_mask = torch.ones(B, T)

    std_loss, cov_loss = vectorised_masked_vicreg_loss(x, padding_mask)
    # In the code above, if there are no valid positions, we still compute std_x but N=0.
    # The snippet handles the dimension checks. The cov_loss is set to 0,
    # and the std_loss can be something but let's see if it is finite.

    # We at least check that no error is thrown, and that the output is a valid float
    assert std_loss.numel() == 1
    assert cov_loss.numel() == 1
    # Typically you'd want to define the exact behavior. The code as given might produce a normal float
    # for std_loss or do an average of an empty tensor, which could be NaN.
    # If it is NaN, you might want to clamp or handle it in production code.
    # Here we just check it's a finite float.
    assert std_loss.isfinite()
    assert cov_loss == 0.0


#
# ------------------- TESTS FOR vectorised_masked_smooth_l1_loss -------------------
#


def test_smooth_l1_loss_no_padding():
    """Test smooth L1 with no padding."""
    torch.manual_seed(42)
    B, L, D = 2, 3, 4
    pred = torch.randn(B, L, D)
    tgt = torch.randn(B, L, D)
    padding_mask = torch.zeros(B, L)  # no padding
    beta = 1.0

    loss = vectorised_masked_smooth_l1_loss(pred, tgt, padding_mask, beta)
    assert loss.shape == ()
    assert loss >= 0
    # Compare with standard smooth_l1_loss on the entire input
    expected_loss = F.smooth_l1_loss(pred, tgt, beta=beta)
    torch.testing.assert_close(loss, expected_loss)


def test_smooth_l1_loss_some_padding():
    """Test smooth L1 with partial padding. Make sure the padded positions are ignored."""
    torch.manual_seed(42)
    B, L, D = 2, 5, 3
    pred = torch.randn(B, L, D)
    tgt = torch.randn(B, L, D)

    # Let the last 2 positions in each sequence be padding
    padding_mask = torch.zeros(B, L)
    padding_mask[:, -2:] = 1  # these are padding

    beta = 1.0

    loss = vectorised_masked_smooth_l1_loss(pred, tgt, padding_mask, beta)

    # Manually compute the "valid" positions' smooth L1
    # We'll only compare the first L-2 positions
    valid_pred = pred[:, :-2, :].reshape(-1, D)
    valid_tgt = tgt[:, :-2, :].reshape(-1, D)
    manual_loss = F.smooth_l1_loss(valid_pred, valid_tgt, beta=beta)

    torch.testing.assert_close(loss, manual_loss)


def test_smooth_l1_loss_all_padding():
    """If everything is padded, the function returns 0.0."""
    torch.manual_seed(42)
    B, L, D = 2, 3, 2
    pred = torch.randn(B, L, D)
    tgt = torch.randn(B, L, D)

    padding_mask = torch.ones(B, L)  # everything is padding

    loss = vectorised_masked_smooth_l1_loss(pred, tgt, padding_mask, beta=1.0)
    assert loss == 0.0


def test_smooth_l1_loss_empty_batch():
    """Edge case: B=0 or L=0. Usually unusual in training, but let's see if code handles it."""
    # This might cause a shape mismatch or an immediate 'no valid positions.'
    # We'll do B=0 to simulate an empty batch.
    pred = torch.empty(0, 3, 2)
    tgt = torch.empty(0, 3, 2)
    padding_mask = torch.empty(0, 3)

    loss = vectorised_masked_smooth_l1_loss(pred, tgt, padding_mask, beta=1.0)
    assert loss.numel() == 1
    # Usually should be 0.0, but check if it doesn't crash.
    assert loss == 0.0


# To run the tests, save this script as `test_weighted_mse_loss.py` and execute:
# pytest test_weighted_mse_loss.py
