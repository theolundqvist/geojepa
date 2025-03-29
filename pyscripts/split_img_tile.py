import glob
import os

from einops import einops
from rootutils import rootutils
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import src.utils.tile_utils as utils

root = rootutils.setup_root(
    os.path.abspath(""), indicator=".project-root", pythonpath=True
)

# def run(folder, output_dir, max_images):
#     files = [name for name in glob.glob(f'{folder}/**/*.webp', recursive=True)]
#     if max_images:
#         files = files[:max_images]
#     for file in tqdm(files):
#         image = Image.open(file).convert('RGB')
#         transform = transforms.ToTensor()
#         name = "_".join(file[:-5].split("/")[-3:])
#         # split into 4x4 grid
#         coords = utils.get_sub_tiles(int(name.split("_")[1]), int(name.split("_")[2]))

#         images = einops.rearrange(transform(image), "c (g1 h) (g2 w) -> (g1 g2) c h w", h=224, w=224)
#         names = [f"16_{x}_{y}" for x, y in coords]
#         toImg = transforms.ToPILImage()
#         for image, name in zip(images, names):
#             toImg(image).save(f"{output_dir}/{name}.webp", format='WEBP', quality=100)


# Use if already converted to 14_xxxx_yyyy.pbf from 14/xxxx/yyyy.pbf
def run(folder, output_dir, max_images):
    files = [name for name in glob.glob(f"{folder}/*.webp", recursive=True)]
    if max_images:
        files = files[:max_images]
    for file in tqdm(files):
        image = Image.open(file).convert("RGB")
        transform = transforms.ToTensor()
        # name = "_".join(file[:-5].split("/")[-3:])
        name = file.split("/")[-1].split(".")[0]
        # split into 4x4 grid
        coords = utils.get_sub_tiles(int(name.split("_")[1]), int(name.split("_")[2]))

        images = einops.rearrange(
            transform(image), "c (g1 h) (g2 w) -> (g1 g2) c h w", h=224, w=224
        )
        names = [f"16_{x}_{y}" for x, y in coords]
        toImg = transforms.ToPILImage()
        for image, name in zip(images, names):
            toImg(image).save(f"{output_dir}/{name}.webp", format="WEBP", quality=100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="split_sat_tiles", description="Split all satellite images in dir"
    )
    parser.add_argument(
        "-i", "--dir", type=str, help="Directory containing satellite images"
    )
    parser.add_argument("-o", "--out", type=str, help="Output directory")
    parser.add_argument(
        "--max-images", type=int, help="Maximum number of images to process"
    )

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run(args.dir, args.out, args.max_images)
