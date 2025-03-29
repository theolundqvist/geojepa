import math
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib import cm


def tile_zxy_to_latlon_bbox(z: int, x: int, y: int) -> tuple:
    """
    Converts tile coordinates (z, x, y) to latitude and longitude bounding box.

    Args:
        z (int): Zoom level.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.

    Returns:
        tuple: (lat1, lon1, lat2, lon2) representing the bounding box.
    """
    MIN_ZOOM_LEVEL = 0
    MAX_ZOOM_LEVEL = 22

    if not (MIN_ZOOM_LEVEL <= z <= MAX_ZOOM_LEVEL):
        raise ValueError(
            f"Zoom level {z} is out of range [{MIN_ZOOM_LEVEL}, {MAX_ZOOM_LEVEL}]"
        )

    max_xy = 2**z - 1
    if not (0 <= x <= max_xy):
        raise ValueError(f"Tile x value {x} is out of range [0, {max_xy}]")

    if not (0 <= y <= max_xy):
        raise ValueError(f"Tile y value {y} is out of range [0, {max_xy}]")

    def tile_to_lon(x, z):
        return x / (2**z) * 360.0 - 180.0

    def tile_to_lat(y, z):
        n = math.pi - 2.0 * math.pi * y / (2**z)
        return math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))

    lon1 = tile_to_lon(x, z)
    lat1 = tile_to_lat(y, z)
    lon2 = tile_to_lon(x + 1, z)
    lat2 = tile_to_lat(y + 1, z)

    return (lat1, lon1, lat2, lon2)


def read_tile_file(file_path: str) -> pd.DataFrame:
    """
    Reads the tile file and parses it into a DataFrame.

    Args:
        file_path (str): Path to the input file.

    Returns:
        pd.DataFrame: DataFrame with columns ['z', 'x', 'y', 'number']
    """
    data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                tile, number = line.split(":")
                z, x, y = map(int, tile.split("_"))
                number = float(number)
                data.append({"z": z, "x": x, "y": y, "number": number})
            except ValueError as e:
                print(f"Skipping invalid line: {line}. Error: {e}")
    df = pd.DataFrame(data)
    return df


def normalize_numbers(df: pd.DataFrame) -> tuple:
    """
    Normalize the 'number' column based on mean and standard deviation.
    - 0.0: Fully transparent.
    - 0.1 <= number < threshold: Color intensity scaled between 100 and 255.
    - number >= threshold: Maximum red intensity (255).

    Args:
        df (pd.DataFrame): DataFrame with a 'number' column.

    Returns:
        tuple: (DataFrame with 'color' column, mean, std, threshold)
    """
    mean = df["number"].mean()
    std = df["number"].std()
    threshold = mean + std * 2

    def get_color(x):
        if x == 0.0:
            return 0  # Fully transparent
        elif x >= threshold:
            return 255  # Maximum red
        elif x >= 0.1:
            # Scale between 100 and 255
            scaled = 100 + (x / threshold) * 155
            return min(int(scaled), 255)
        else:
            return 100  # Minimum visible red for numbers < 0.1 but > 0

    df["color"] = df["number"].apply(get_color)

    return df, mean, std, threshold


def plot_tiles(
    df: pd.DataFrame, mean: float, std: float, threshold: float, task_name=""
):
    """
    Plots the tiles on a map with colors based on the 'number' value.

    Args:
        df (pd.DataFrame): DataFrame with columns ['z', 'x', 'y', 'number', 'color']
        mean (float): Mean of the 'number' column.
        std (float): Standard deviation of the 'number' column.
        threshold (float): Calculated as mean + std.
    """
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # Compute overall bounding box
    lat_min = df.apply(
        lambda row: tile_zxy_to_latlon_bbox(row["z"], row["x"], row["y"])[0], axis=1
    ).min()
    lon_min = df.apply(
        lambda row: tile_zxy_to_latlon_bbox(row["z"], row["x"], row["y"])[1], axis=1
    ).min()
    lat_max = df.apply(
        lambda row: tile_zxy_to_latlon_bbox(row["z"], row["x"], row["y"])[2], axis=1
    ).max()
    lon_max = df.apply(
        lambda row: tile_zxy_to_latlon_bbox(row["z"], row["x"], row["y"])[3], axis=1
    ).max()

    ax.set_extent(
        [lon_min - 0.2, lon_max + 0.2, lat_min - 0.2, lat_max + 0.2],
        crs=ccrs.PlateCarree(),
    )

    cmap = cm.get_cmap("Reds")

    for _, row in df.iterrows():
        z, x, y, color, label = (
            row["z"],
            row["x"],
            row["y"],
            row["color"],
            row["number"],
        )
        lat1, lon1, lat2, lon2 = tile_zxy_to_latlon_bbox(z, x, y)
        width = lon2 - lon1
        height = lat2 - lat1

        if color == 0:
            # Fully transparent
            rect = Rectangle(
                (lon1, lat1),
                width,
                height,
                linewidth=0.5,
                edgecolor="black",
                alpha=0.1,
                facecolor=(1, 1, 1),
                transform=ccrs.PlateCarree(),
            )
        elif label < 0:
            rect = Rectangle(
                (lon1, lat1),
                width,
                height,
                linewidth=0.5,
                edgecolor="black",
                alpha=0.5,
                facecolor=(0, 0, 1),
                transform=ccrs.PlateCarree(),
            )

        else:
            # Scale color to [0,1] for cmap
            normalized_color = color / 255
            facecolor = cmap(normalized_color)  # RGBA
            rect = Rectangle(
                (lon1, lat1),
                width,
                height,
                linewidth=0.5,
                edgecolor="black",
                facecolor=facecolor,
                alpha=0.8,
                transform=ccrs.PlateCarree(),
            )

        ax.add_patch(rect)

    # Create a custom normalization for the colorbar
    norm = Normalize(vmin=0, vmax=threshold)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Number Value")

    # Add a horizontal line or marker to indicate the threshold
    # Calculate the position in normalized [0,1] space
    cbar.ax.axhline(
        y=threshold / threshold if threshold != 0 else 0,
        color="black",
        linewidth=1,
        linestyle="--",
    )
    cbar.ax.text(
        1.05,
        threshold / threshold if threshold != 0 else 0,
        f"Mean + 2 Std\n({threshold:.2f})",
        transform=cbar.ax.transAxes,
        verticalalignment="center",
    )

    plt.title(f"Tile Visualization Colored by Number\n{task_name}")
    plt.show()


def main(labels_path: str):
    """
    Main function to read, process, and plot the tile data.

    Args:
        labels_path (str): Path to the input labels.txt file.
    """
    df_tiles = read_tile_file(labels_path)
    if df_tiles.empty:
        print("No valid data to plot.")
        return
    df_tiles, mean, std, threshold = normalize_numbers(df_tiles)
    print(
        f"Mean: {mean:.2f}, Std: {std:.2f}, Threshold (Mean + 2 Std): {threshold:.2f}"
    )
    plot_tiles(df_tiles, mean, std, threshold, labels_path.split("/")[-2])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot map tiles colored by number values."
    )
    parser.add_argument(
        "-i",
        "--task",
        type=str,
        required=True,
        help="Path to the input dir containing (labels.txt).",
    )
    args = parser.parse_args()
    main(args.task + "/labels.txt")
