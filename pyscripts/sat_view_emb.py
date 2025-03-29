import pandas as pd
import umap.umap_ as umap
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torchvision.transforms.functional import to_pil_image

# Prepare the 2D embedding using UMAP
obj = pd.read_pickle(
    "pyscripts/sat/scalemae/embeddings.pkl"
)  # Adjust path if necessary
tiles = obj["tiles"]
cls = obj["cls"]
features = obj["features"]


def plot(features, title, block=False):
    umap_model = umap.UMAP(n_components=2, random_state=42)
    mapped = umap_model.fit_transform(features)

    # Create the plot
    fig, ax = plt.subplots()
    ax.scatter(mapped[:, 0], mapped[:, 1], s=100, c="none", edgecolors="black")

    for i, (x, y) in enumerate(mapped):
        if i >= len(tiles.SAT_imgs):
            break
        img = to_pil_image(tiles.SAT_imgs[i])
        imagebox = OffsetImage(img, zoom=0.20)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)
    plt.title(title)
    plt.show(block=block)


plot(cls, "SCALEMAE TOKEN 0")
plot(features[:, -1, :], "SCALEMAE TOKEN 196")

obj = pd.read_pickle("pyscripts/sat/resnet/embeddings.pkl")
plot(obj["features"], "RESNET")

obj = pd.read_pickle("pyscripts/sat/efficientnet/embeddings.pkl")
plot(obj["features"], "EfficientNet")

obj = pd.read_pickle("pyscripts/sat/vitb16/embeddings.pkl")
plot(obj["features"], "ViTB16", block=True)
