import pandas as pd
import umap.umap_ as umap
from torch.utils.data import DataLoader

from src.data.components.tiles import TileBatch
from src.modules.tag_encoder import TagIndexer
from src.data.components.tiles import collate_tiles
from src.modules.tag_encoder import TagEncoder

# Prepare the 2D embedding using UMAP

encoder = TagEncoder()
indexer = TagIndexer()


def plot(embs, tags, title, block=False):
    umap_model = umap.UMAP(n_components=2)
    mapped = umap_model.fit_transform(embs)

    import plotly.express as px

    tag_strings = []
    for feature in tags:
        tag_string = []
        for idx in feature:
            if idx != 0:
                tag_string.append(f"[{indexer.index_to_tag(idx.item())}]")
        tag_strings.append("   ".join(tag_string))
    data = {
        "X": mapped[:, 0],
        "Y": mapped[:, 1],
        "Label": tag_strings,
        # 'Value': [100, 200, 150, 250, 300]
    }
    df = pd.DataFrame(data)
    df["Color"] = df["Label"].apply(lambda x: "traffic_signals" in x)
    fig = px.scatter(
        df,
        x="X",
        y="Y",
        color="Color",
        # text='Label',  # Adds labels to each point
        hover_data=["Label"],  # Specifies which data to show on hover
        title="Interactive Scatter Plot",
        labels={"X": "X-axis Label", "Y": "Y-axis Label"},  # Axis labels
        template="plotly_white",  # Optional: Choose a template for styling
    )

    # fig.update_traces(
    #     marker=dict(size=12, color='skyblue', line=dict(width=2, color='DarkSlateGrey')),
    #     textposition='top center'
    # )

    fig.show()


org = TileDatasetOnFile(
    "data/tiles/tiny/tasks/traffic_signals", split="test", load_images=False
)
org_loader = DataLoader(org, batch_size=6, collate_fn=collate_tiles, num_workers=0)
cheat = TileDatasetOnFile(
    "data/tiles/tiny/tasks/pretraining", split="train", load_images=False
)
cheat_loader = DataLoader(org, batch_size=10, collate_fn=collate_tiles, num_workers=0)

for tiles in cheat_loader:
    tiles: TileBatch = tiles
    x = encoder(tiles.tags)
    B, F, _ = x.shape
    x = x.view(B * F, -1)
    tags = tiles.tags.view(B * F, -1)
    plot(x, tags, "Original", block=True)
    input("Plot next")
