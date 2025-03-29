from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.task_datamodule import TaskDataset, TaskBatch, collate_tasks


def calculate_mean_std(loader, num_images=200000):
    # Initialize sums and count
    mean_sum = torch.zeros(3)
    std_sum = torch.zeros(3)
    total_pixels = 0

    # Loop over images in the dataloader
    for i, batch in tqdm(enumerate(loader)):
        assert type(batch) == TaskBatch, f"Expected TaskBatch, got {type(batch)}"
        images = batch.tiles.SAT_imgs
        # Break if reaching the desired number of images
        if i * images.size(0) >= num_images:
            break

        # Calculate mean and std per batch
        batch_mean = images.mean(dim=[0, 2, 3])  # Mean across batch, height, and width
        batch_std = images.std(
            dim=[0, 2, 3]
        )  # Std deviation across batch, height, and width

        # Accumulate mean and std
        mean_sum += batch_mean * images.size(0)
        std_sum += batch_std * images.size(0)
        total_pixels += images.size(0)

    # Final mean and std
    mean = mean_sum / total_pixels
    std = std_sum / total_pixels

    return mean, std


script_dir = Path(__file__).resolve().parent
print("Loading dataset")
dataset = TaskDataset(
    script_dir / "../data/tiles/huge/tasks/pretraining", use_image_transforms=False
)
print(f"Found {len(dataset)} images")
dataloader = DataLoader(
    dataset, collate_fn=collate_tasks, num_workers=12, shuffle=True, batch_size=100
)
# Usage
mean, std = calculate_mean_std(dataloader)
print(f"Mean: {mean}")
print(f"Std: {std}")
