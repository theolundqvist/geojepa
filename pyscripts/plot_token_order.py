import time
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation

from src.data.components.raw_tile_dataset import RawTileDataset
from src.data.components.tiles import collate_tiles
from src.modules.mock_tokenizer import MockTokenizer
from src.modules.tokenizer import Modality


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


def main():
    # Step 1: Get a TileBatch from your dataset
    data = RawTileDataset("data/tiles/huge/processed", split="")
    tile_batch = data.__getitem__(0)
    idx = 13
    tile_batch = collate_tiles(tile_batch)
    print("plotting:", tile_batch.names()[idx])
    # Repeat next(iter_loader) as necessary to get the desired batch

    # Step 2: Create a MockTokenizer and process the TileBatch
    tokenizer = MockTokenizer(token_dim=1, tokenize_geometry=False, sort_spatially=True)
    start = time.time()
    tokens, positions, modalities = tokenizer(tile_batch)
    print(f"tokenized in {time.time() - start}s")

    # Step 3: Visualize tokens in order
    animate_tokens(
        tile_batch, tokens, positions, modalities, tokenizer.indices, sample_idx=idx
    )


if __name__ == "__main__":
    main()
