import matplotlib.pyplot as plt
import numpy as np


def plot_weighted_mse_weights():
    """
    Generates a plot showing how WeightedMSELoss assigns weights to zero and non-zero targets
    based on different zero_fraction values.
    """
    # Define a range of zero_fraction values from 1% to 99%
    zero_fractions = np.linspace(0.01, 0.99, 500)  # 500 points for smoothness

    # Calculate corresponding non_zero_weights
    non_zero_weights = zero_fractions / (1 - zero_fractions)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot non-zero weights
    plt.plot(zero_fractions, non_zero_weights, label="Non-Zero Weight", color="blue")

    # Plot zero weights (constant at 1.0)
    plt.plot(
        zero_fractions,
        np.ones_like(zero_fractions),
        label="Zero Weight",
        color="orange",
        linestyle="--",
    )

    # Highlight specific zero_fraction values if desired
    highlight_zero_fractions = [0.1, 0.5, 0.95]
    colors = ["green", "red", "purple"]
    for zf, color in zip(highlight_zero_fractions, colors):
        weight = zf / (1 - zf)
        plt.scatter(
            zf,
            weight,
            color=color,
            label=f"zero_fraction={zf:.2f} (weight={weight:.2f})",
        )
        plt.annotate(
            f"({zf:.2f}, {weight:.2f})",
            xy=(zf, weight),
            xytext=(zf, weight + 5),
            arrowprops=dict(facecolor=color, shrink=0.05),
            fontsize=9,
            color=color,
        )

    # Set plot labels and title
    plt.xlabel("Zero Fraction (Proportion of Zero Targets)")
    plt.ylabel("Weight Assigned to Targets")
    plt.title("WeightedMSELoss: Weight Assignment Based on Zero Fraction")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_weighted_mse_weights()
