from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import List
import src.data.components.processed_tile_group_pb2 as osm_schema
import rootutils


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation

from src.data.components.tiles import uncollate_first_tile
from src.data.tiles_datamodule import TilesDataModule
from src.modules.tokenizer import Modality

from src import log
from src.data.components.tiles import Tile, TileBatch
from src.modules.tag_encoder import TagIndexer
from torch.utils.data import Dataset

from src.modules.tokenizer import TileTokenizer


class RawTileDataset(Dataset):
    def __init__(
        self,
        task_dir="data/tiles/tiny/pretraining",
        image_dir="data/tiles/huge/sat_tiles",
        split="train",
        load_images=True,
        tag_embeddings_file="data/tiles/embeddings.pkl",
    ):
        self.img_dir = image_dir
        self.split = split
        self.tile_dir = task_dir
        self.load_images = load_images
        self.tag_indexer = TagIndexer(tag_embeddings_file)

    def __getitem__(
        self, name
    ) -> List[Tile]:  # Replace 'Tile' with your actual Tile class/type
        try:
            tile_group_path = Path(self.tile_dir) / self.split / f"{name}.pbf"
            tile_group = osm_schema.TileGroup()
            with open(tile_group_path, "rb") as f:
                tile_group.ParseFromString(f.read())
            g_coord = torch.tensor(
                (tile_group.zoom, tile_group.x, tile_group.y), dtype=torch.int32
            )
            tiles = tile_group.tiles
            image_dict = {}
            items = []
            for tile in tiles:
                tile_coord = torch.tensor(
                    (tile.zoom, tile.x, tile.y), dtype=torch.int32
                )
                assert not torch.isnan(tile_coord).any(), "nodes contains NaNs"
                # Extract nodes: reshape flat list to [N, 2]
                nodes = torch.tensor(tile.nodes, dtype=torch.float32).reshape(-1, 2)
                assert not torch.isnan(nodes).any(), "nodes contains NaNs"
                bbox_local_coords = torch.tensor(
                    tile.local_coords, dtype=torch.float32
                ).reshape(-1, 2)
                assert not torch.isnan(bbox_local_coords).any(), (
                    "bbox_local_coords contains NaNs"
                )
                # Extract intra_edges: reshape to [2, E_intra]
                edges = torch.tensor(tile.intra_edges, dtype=torch.long)
                assert not torch.isnan(edges).any(), "intra_edges contains NaNs"
                intra_edges = torch.stack((edges[::2], edges[1::2]))

                # Extract inter_edges: reshape to [2, E_inter]
                edges = torch.tensor(tile.inter_edges, dtype=torch.long)
                assert not torch.isnan(edges).any(), "inter_edges contains NaNs"
                inter_edges = torch.stack((edges[::2], edges[1::2]))

                # Extract min_boxes
                min_boxes = torch.zeros((len(tile.features), 4, 2), dtype=torch.float32)
                no_min_box = 0
                for j, feature in enumerate(tile.features):
                    if len(feature.min_box) == 0:
                        no_min_box += 1
                        continue
                    min_boxes[j] = torch.tensor(
                        feature.min_box, dtype=torch.float32
                    ).reshape(4, 2)
                assert not torch.isnan(min_boxes).any(), "min_boxes contains NaNs"
                min_boxes = min_boxes[:, :, [1, 0]]
                # if no_min_box > 0:
                #     log.info(f"{no_min_box} features without min_box")

                node_to_feature = torch.tensor(tile.node_to_feature, dtype=torch.long)
                assert not torch.isnan(node_to_feature).any(), (
                    "node_to_feature index contains NaNs"
                )

                # Extract tags
                tags = []
                for feature in tile.features:
                    temp = []
                    for k, v in zip(feature.tags[::2], feature.tags[1::2]):
                        tag = self.tag_indexer.tag_to_index((k, v))
                        if tag != 0:
                            temp.append(tag)
                    tags.append(torch.tensor(temp, dtype=torch.long))
                if len(tags) == 0:
                    log.info(
                        f"Tile {tile.zoom}_{tile.x}_{tile.y} has no tags, group: {g_coord}"
                    )
                    tags = torch.tensor([[]], dtype=torch.long)
                else:
                    tags = pad_sequence(tags, batch_first=True, padding_value=0)
                # tensor of shape [max_Features, max_Tags]

                items.append(
                    Tile(
                        nbr_features=len(tile.features),
                        group_coord=g_coord,
                        tile_coord=tile_coord,
                        # tile_name=f"{tile.zoom}_{tile.x}_{tile.y}",
                        # -----------
                        nodes=nodes,
                        bbox_local_coords=bbox_local_coords,
                        inter_edges=inter_edges,
                        intra_edges=intra_edges,
                        node_to_feature=node_to_feature,
                        # -----------
                        min_boxes=min_boxes,
                        tags=tags,
                        SAT_img=torch.tensor([]),
                        original_img=torch.tensor([]),
                    )
                )
            return items
        except Exception as e:
            log.error(f"Error in __getitem__ at index : {e}")
            raise e  # Re-raise the exception to let the DataLoader handle it


def animate_tokens(tile_batch, tokens, positions, modalities, indices, sample_idx=0):
    # Get positions and modalities for the sample
    positions_sample = positions[sample_idx]  # Shape: [num_tokens, 8]
    modalities_sample = modalities[sample_idx]  # Shape: [num_tokens]
    indices_sample = indices[sample_idx]

    num_tokens = positions_sample.shape[0]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect("equal")
    # ax.invert_yaxis()  # Invert y-axis to match image coordinate system
    ax.axis("off")

    # Function to initialize the plot
    def init():
        return []

    # Function to update the plot at each frame
    def update(frame):
        # ax.clear()
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)
        ax.set_aspect("equal")
        # ax.invert_yaxis()
        # ax.axis('off')

        idx = frame  # Token index

        # Get the bounding box
        bbox = positions_sample[idx]
        # Reshape bbox to (4, 2)
        corners = bbox.view(4, 2).cpu().numpy()

        # Create a Polygon patch
        color = "red"
        if modalities_sample[idx] == Modality.IMG:
            color = "blue"
            polygon = Polygon(
                corners, closed=True, edgecolor=color, fill=False, linewidth=1
            )
            ax.add_patch(polygon)
            return [polygon]
        else:
            polygon = Polygon(
                corners, closed=True, edgecolor=color, fill=False, linewidth=1
            )
            ax.add_patch(polygon)
            # centroid = corners.mean(axis=0)
            # ax.text(centroid[0], centroid[1], str(idx), color='blue', fontsize=8, ha='center', va='center')
            return [polygon]

    ani = animation.FuncAnimation(
        fig, update, frames=range(num_tokens), init_func=init, interval=30, blit=False
    )

    plt.show()


def log_emb_image(embedding, name):
    matrix = embedding.clone().detach()
    fig, ax = plt.subplots(figsize=(30, 45))
    ax.imshow(matrix.cpu().numpy(), cmap="viridis", interpolation="nearest")
    ax.set_title("Matrix Heatmap using imshow")
    ax.set_xlabel("Hidden dimensions")
    ax.set_ylabel("Tile tokens")
    plt.show()
    # for mod in range(Modality.PAD + 1, Modality.CLS + 1):
    #     run(embedding[mods == mod], Modality(mod).name)


def run():
    # find idx
    # name = '16_18689_24947'
    # data = RawTileDataset("data/tiles/huge/tasks/pretraining", split="train")
    #
    # coord = list(map(int, name.split("_")))
    # parent = get_parent_tile(coord[1], coord[2])
    # print(parent)
    # tiles: List[Tile] = data.__getitem__(f"14_{parent[0]}_{parent[1]}")
    # batch = collate_tiles(tiles)
    # print(batch.names())
    # index = batch.names().index(name)
    # tile = tiles[index]
    # batch = collate_tiles([tiles[index]])

    m = TilesDataModule(
        "data/tiles/huge/tasks/pretraining",
        batch_size=3,
        group_size=2,
        load_images=False,
        num_workers=1,
        shuffle=False,
    )
    m.setup()
    loader = m.train_dataloader()
    iter = loader.__iter__()
    for x in range(10):
        batch: TileBatch = next(iter)
        for i in range(1000):
            if (
                batch.feature_counts.max().item() < 30
                or batch.feature_counts.max().item() > 100
            ):
                batch = next(iter)
            else:
                break
        tile = uncollate_first_tile(batch)

        # dataset = TileDataset("data/tiles/huge/tasks/pretraining", split="val", load_images=False)
        # dataset.__init_worker__()
        # tile = dataset.__getitem__(0)

        tokenizer = TileTokenizer(
            token_dim=32,
            tokenize_images=False,
            tokenize_geometry=True,
            sort_spatially=True,
        )
        tokens, positions, modalities, indices = tokenizer.forward(batch)
        B, T, _ = positions.shape
        log_emb_image(
            modalities.unsqueeze(-1).expand(B, T, 8).permute(1, 0, 2).reshape(T, B * 8),
            "",
        )
        log_emb_image(tokens[0], "")
        log_emb_image(positions[0], "")
        log_emb_image(tokens.permute(1, 0, 2).reshape(T, -1), "")
        log_emb_image(positions.permute(1, 0, 2).reshape(T, B * 8), "")
        # for i in indices[0][:60]:
        #     print(f"\n\nFeature {i}:")
        #     print("pos:", tile.min_boxes[i])
        # for tt in tile.tags[i]:
        #     if tt != 0:
        #         print(tt.item(), tok.index_to_tag(tt.item()))

        animate_tokens(batch, tokens, positions, modalities, indices, sample_idx=0)


if __name__ == "__main__":
    run()
