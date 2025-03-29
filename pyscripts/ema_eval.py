import math
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from src.modules.linear_ema import LinearEMA as LinearEMA
from torch import nn

# Assuming the EMA class code is already defined as provided.
# If it's in a separate module, you can import it accordingly.
# from ema_module import EMA

# Define the EMA class (as provided in your code)
# [Insert the EMA class code here or assume it's already defined]

# For the purpose of this example, we assume the EMA class is already defined above.


# Define a simple model with a single parameter
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(0.0))  # Initialize at 0.0

    def forward(self):
        return self.param


def simulate_training(model, ema_instances, num_steps=100, update_func=None):
    """Simulate training by updating the model's parameter and EMA instances.

    Args:
        model (nn.Module): The model to train.
        ema_instances (list of EMA): List of EMA instances to update.
        num_steps (int): Number of training steps to simulate.
        update_func (callable): Function to update the model's parameter.
    Returns:
        dict: Dictionary containing the parameter history and EMA histories.
    """
    history = {
        "step": [],
        "param": [],
        "decays": [],
    }
    for idx, ema in enumerate(ema_instances):
        history[f"EMA_{idx}"] = []

    for step in range(num_steps):
        # Update the model's parameter using the update function
        if update_func:
            update_func(model, step)
        else:
            # Default update: sine wave
            new_value = math.sin(step * 0.1) * 20 + step * 0.2
            model.param.data = torch.tensor(new_value)

        # Record the current parameter value
        history["step"].append(step)
        history["param"].append(model.param.item())
        history["decays"].append(ema_instances[0].get_current_decay())

        # Update each EMA instance and record their parameter values
        for idx, ema in enumerate(ema_instances):
            ema.update()
            ema_value = ema.ema_model.param.item()
            history[f"EMA_{idx}"].append(ema_value)

    return history


def plot_decay(decay_values):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(decay_values)), decay_values, label="Decay Values")
    plt.xlabel("Training Step")
    plt.ylabel("EMA Decay Value")
    plt.title("EMA Decay Values Over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ema_contributions(history, ema_instances):
    """Plot the parameter and its EMA contributions over training steps.

    Args:
        history (dict): Dictionary containing the parameter and EMA histories.
        ema_params (list): List of EMA parameter settings for labeling.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        history["step"], history["param"], label="Original Parameter", color="blue"
    )

    for idx, instance in enumerate(ema_instances):
        plt.plot(
            history["step"], history[f"EMA_{idx}"], label=f"EMA {idx}", linestyle="--"
        )

    plt.xlabel("Training Step")
    plt.ylabel("Parameter Value")
    plt.title("EMA Contribution Over Training Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # Initialize the model
    model = SimpleModel()

    # Create EMA instances with different betas
    ema_instances = []
    # Deepcopy the model for each EMA to ensure separate EMA models
    linear_ema = LinearEMA(
        model=model,
        ema_model=deepcopy(model),
        training_steps=500,
        model_influence_start=1 - 0.9995,
        model_influence_end=1 - 0.99999,
    )

    ema_instances.append(linear_ema)

    # Simulate training
    history = simulate_training(model, ema_instances, num_steps=500)

    # Plot the EMA contributions
    plot_decay(history["decays"][2:])
    plot_ema_contributions(history, ema_instances)

    def plot_momentum_schedule(ema_start, ema_end, total_steps):
        momentum_values = [
            ema_start + i * (ema_end - ema_start) / total_steps
            for i in range(total_steps + 1)
        ]
        plt.plot(momentum_values)
        plt.xlabel("Training Step")
        plt.ylabel("Momentum")
        plt.title("EMA Momentum Schedule")
        plt.grid(True)
        plt.show()

    # Example usage
    plot_momentum_schedule(ema_start=0.9, ema_end=0.999, total_steps=10000)

    def plot_momentum_schedule(
        inv_gamma: float,
        power: float,
        min_value: float,
        beta: float,
        update_after_step: int,
        total_steps: int,
    ):
        """Plots the EMA decay (momentum) schedule over training steps.

        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup.
            power (float): Exponential factor of EMA warmup.
            min_value (float): The minimum EMA decay rate.
            beta (float): The maximum EMA decay rate.
            update_after_step (int): Number of initial steps to skip EMA updates.
            total_steps (int): Total number of training steps to simulate.
        """
        decays = []
        steps = list(range(1, total_steps + 1))

        for step in steps:
            epoch = max(step - update_after_step - 1, 0)
            if epoch <= 0:
                current_decay = 0.0
            else:
                value = 1 - (1 + epoch / inv_gamma) ** -power
                current_decay = min(max(value, min_value), beta)
            decays.append(current_decay)

        plt.figure(figsize=(10, 6))
        plt.plot(steps, decays, label="EMA Decay (current_decay)")
        plt.axvline(x=update_after_step, color="r", linestyle="--", label="EMA Start")
        plt.xlabel("Training Steps")
        plt.ylabel("Decay Factor (current_decay)")
        plt.title("EMA Decay Schedule")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Define EMA parameters
    inv_gamma = 20.0
    power = 1.0
    min_value = 0.9
    beta = 0.9999
    update_after_step = 100
    total_steps = 10000  # Total training steps to plot

    # Plot the momentum schedule
    plot_momentum_schedule(
        inv_gamma=inv_gamma,
        power=power,
        min_value=min_value,
        beta=beta,
        update_after_step=update_after_step,
        total_steps=total_steps,
    )


if __name__ == "__main__":
    main()
