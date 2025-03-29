class UnfreezeParamGroupsCallback:
    def __init__(self, fraction=0.85):
        """
        Initializes the callback.

        Args:
            fraction (float): The fraction of total training steps after which to unfreeze the parameters.
                              Default is 0.85 (i.e., unfreeze during the last 15% of training).
        """
        super().__init__()
        self.fraction = fraction
        self.unfroze = False

    def on_train_epoch_end(self, trainer, pl_module):
        if self.unfroze:
            return

        total_steps = trainer.estimated_stepping_batches
        current_step = trainer.global_step

        if current_step >= int(self.fraction * total_steps):
            optimizer = trainer.optimizers[0]  # Assuming a single optimizer
            for group in pl_module.unfreeze_param_groups:
                # Unfreeze the parameters
                for param in group["params"]:
                    param.requires_grad = True

                # Calculate the new learning rate
                base_lr = optimizer.param_groups[0][
                    "lr"
                ]  # Assuming the base LR is in the first param group
                new_lr = base_lr * group.get("lr_modifier", 1.0)

                # Add the parameters to the optimizer with the new learning rate
                optimizer.add_param_group({"params": group["params"], "lr": new_lr})

                # Logging
                pl_module.log(f"opt/unfreeze_{group['name']}_lr", new_lr, prog_bar=True)
                pl_module.log(
                    f"info/unfreeze_{group['name']}_step",
                    float(current_step),
                    prog_bar=False,
                )

                print(
                    f"[UnfreezeParamGroupsCallback] Unfroze '{group['name']}' at step {current_step} with LR {new_lr}"
                )

            self.unfroze = True  # Ensure this block runs only once
