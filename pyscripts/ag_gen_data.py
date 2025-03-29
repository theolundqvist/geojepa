import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torchvision.transforms
from torch.utils.data import DataLoader

from src.data.task_datamodule import TaskDataModule, TaskBatch

script_path = os.path.dirname(os.path.realpath(__file__))

huge_path = script_path + "/../data/tiles/huge"
tiny_path = script_path + "/../data/tiles/tiny"
ag_data_path = script_path + "/../data/autogluon"
image_dir = ag_data_path + "/images"
os.makedirs(image_dir, exist_ok=True)


def create_split(loader: DataLoader, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    data = []
    print(
        f"Saving {len(loader) * loader.batch_size} tiles from {'/'.join(out_dir.split('/')[-3:])}..."
    )
    transform = torchvision.transforms.ToPILImage()
    for idx, batch in enumerate(loader):
        batch: TaskBatch = batch
        # Convert the image tensor to PIL Image
        # Assuming image_tensor is in CHW format and in range [0, 1] or [0, 255]
        for img_i in range(batch.tiles.size):
            label = batch.labels[img_i]

            # Define a unique filename
            image_filename = os.path.join(
                image_dir, f"{batch.tiles.names()[img_i]}.png"
            )
            if not os.path.exists(image_filename):
                image_tensor = batch.tiles.SAT_imgs[img_i]
                img = transform(image_tensor)
                img.save(image_filename)

            # Assuming label is a single float for regression
            data.append({"image": image_filename, "label": label.item()})
    # Create a DataFrame
    df = pd.DataFrame(data)
    pd.to_pickle(df, out_dir + "/data.pkl")
    return df


def build_task(task_dir, tiny_only, load_images):
    assert task_dir == Path(task_dir).name, f"Expected {task_dir} to be a directory"
    start = time.time()

    tiny_loader = TaskDataModule(
        tiny_path + "/tasks/" + task_dir,
        batch_size=30,
        num_workers=0,
        use_image_transforms=False,
        load_images=load_images,
        cache=False,
    )
    tiny_loader.setup()
    d = ag_data_path + "/tiny/" + task_dir
    create_split(tiny_loader.test_dataloader(), d + "/test")
    if tiny_only:
        return
    huge_loader = TaskDataModule(
        huge_path + "/tasks/" + task_dir,
        batch_size=100,
        num_workers=0,
        use_image_transforms=False,
        cache=False,
        load_images=load_images,
    )
    huge_loader.setup()
    d = ag_data_path + "/huge/" + task_dir
    create_split(huge_loader.train_dataloader(), d + "/train")
    create_split(huge_loader.val_dataloader(), d + "/val")
    create_split(huge_loader.test_dataloader(), d + "/test")
    print(f"{task_dir} finished in: {time.time() - start}s")


def main(task, tiny_only=False):
    if task is not None:
        build_task(task, tiny_only, load_images=True)

    start = time.time()
    with ProcessPoolExecutor() as executor:
        futures = []
        futures.append(
            executor.submit(
                build_task,
                task_dir="pretraining",
                tiny_only=tiny_only,
                load_images=True,
            )
        )
        for task_dir in os.listdir(huge_path + "/tasks"):
            if task_dir == "pretraining":
                continue
            futures.append(
                executor.submit(
                    build_task,
                    task_dir=task_dir,
                    tiny_only=tiny_only,
                    load_images=False,
                )
            )

    # Monitor task completion
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"Generated an exception: {exc}")

    elapsed = time.time() - start
    print(f"Time elapsed: {elapsed}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=False, default=None)
    parser.add_argument("--tiny-only", required=False, action="store_true")
    args = parser.parse_args()
    main(args.task, args.tiny_only)
