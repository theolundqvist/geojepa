import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.modules.schedulers import CosineAnnealingWarmupRestarts


def create_warmup_annealing(opt, conf):
    warmup_epochs = int(conf["max_epochs"] * conf["warmup_fraction"])
    warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    annealing = CosineAnnealingLR(
        opt, T_max=conf["max_epochs"] - warmup_epochs, eta_min=conf["min_lr"]
    )
    return SequentialLR(opt, schedulers=[warmup, annealing], milestones=[warmup_epochs])


def create_warmup_restarts(opt, conf):
    return CosineAnnealingWarmupRestarts(
        opt,
        first_cycle_steps=conf["first_cycle_steps"],
        cycle_mult=conf["cycle_mult"],
        max_lr=conf["max_lr"],
        min_lr=conf["min_lr"],
        warmup_steps=conf["warmup_steps"],
        gamma=conf["gamma"],
    )


def run_test():
    # Define different scheduler configurations
    max_epochs = 200
    batches_per_epoch = 500
    total_steps = batches_per_epoch * max_epochs
    scheduler_configs = [
        {
            "name": "Cosine Annealing Warmup Restarts",
            "first_cycle_steps": total_steps * 0.3,
            "cycle_mult": 1.2,
            "max_lr": 2e-2,
            "min_lr": 1e-6,
            "opt_lr": 1e-2,
            "warmup_steps": total_steps * 0.05,
            "gamma": 0.6,
            "scheduler": create_warmup_restarts,
        },
        {
            "name": "Linear Warmup Cosine Annealing",
            "max_epochs": total_steps,
            "warmup_fraction": 0.1,
            "opt_lr": 1e-2,
            "min_lr": 1e-7,
            "scheduler": create_warmup_annealing,
        },
    ]

    # Prepare to store learning rates for each config
    lr_history = {config["name"]: [] for config in scheduler_configs}

    # Run the simulation for each scheduler configuration
    for config in scheduler_configs:
        # Define a dummy model parameter
        dummy_param = torch.nn.Parameter(torch.zeros(1))

        # Define optimizer
        opt = torch.optim.SGD([dummy_param], lr=config["opt_lr"])
        sched = config["scheduler"](opt, config)

        # Initialize scheduler

        # Simulate 200 epochs
        for epoch in range(max_epochs):
            # Normally, you would train your model here
            # For this test, we only step the scheduler
            for _ in range(batches_per_epoch):
                sched.step()
                # Get current learning rate
                current_lr = opt.param_groups[0]["lr"]
                lr_history[config["name"]].append(current_lr)

    # Plot the learning rates
    plt.figure(figsize=(12, 8))
    for config_name, lrs in lr_history.items():
        plt.plot(range(1, 200 * 500 + 1), lrs, label=config_name)

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Cosine Annealing Warmup Restarts Learning Rate Schedules")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_test()
