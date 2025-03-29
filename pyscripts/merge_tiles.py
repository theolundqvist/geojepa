import os
import shutil
import argparse

from tqdm import tqdm  # Import tqdm for progress bars


def clear_directory(directory, extensions=None):
    """Remove all files from a directory with specified extensions."""
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                if extensions is None or file.endswith(extensions):
                    os.remove(file_path)


def match_and_copy_files(sat_dir, osm_dir, combined_dir):
    # Clear the combined_tiles directory before copying .pbf and .webp files
    clear_directory(combined_dir, (".pbf", ".webp"))

    # Ensure combined_tiles directory exists
    os.makedirs(combined_dir, exist_ok=True)

    # Get all zoom directories
    zoom_dirs = [
        d for d in os.listdir(sat_dir) if os.path.isdir(os.path.join(sat_dir, d))
    ]

    # Iterate over each zoom directory with tqdm
    for zoom_dir in zoom_dirs:
        zoom_path = os.path.join(sat_dir, zoom_dir)

        for x_dir in tqdm(
            os.listdir(zoom_path),
            desc="Copying files from sat directories",
            unit="dirs",
        ):
            x_path = os.path.join(zoom_path, x_dir)

            if os.path.isdir(x_path):
                webp_files = [f for f in os.listdir(x_path) if f.endswith(".webp")]
                for webp_file in webp_files:
                    base_y = os.path.splitext(webp_file)[0]  # Extract <y> from <y>.webp
                    y = 8192 + (8192 - int(base_y)) - 1
                    # Build corresponding .pbf file path
                    pbf_file = f"{zoom_dir}_{x_dir}_{y}.pbf"
                    pbf_path = os.path.join(osm_dir, pbf_file)

                    # Check if the .pbf file exists
                    if os.path.exists(pbf_path):
                        # Copy .webp file and rename it to match the .pbf name
                        webp_source = os.path.join(x_path, webp_file)
                        new_webp_name = f"{zoom_dir}_{x_dir}_{y}.webp"
                        webp_dest = os.path.join(combined_dir, new_webp_name)
                        shutil.copy2(webp_source, webp_dest)

                        # Copy .pbf file
                        pbf_dest = os.path.join(combined_dir, pbf_file)
                        shutil.copy2(pbf_path, pbf_dest)

                        # print(f"Copied: {new_webp_name} and {pbf_file}")

    if len(zoom_dirs) == 0:
        webp_files = [f for f in os.listdir(sat_dir) if f.endswith(".webp")]
        for webp_file in webp_files:
            sat_file = os.path.splitext(webp_file)[0]  # Extract <y> from <y>.webp

            # Build corresponding .pbf file path
            pbf_file = f"{sat_file}.pbf"
            pbf_path = os.path.join(osm_dir, pbf_file)

            # Check if the .pbf file exists
            if os.path.exists(pbf_path):
                # Copy .webp file and rename it to match the .pbf name
                webp_source = os.path.join(sat_dir, webp_file)
                webp_dest = os.path.join(combined_dir, webp_file)
                shutil.copy2(webp_source, webp_dest)

                # Copy .pbf file
                pbf_dest = os.path.join(combined_dir, pbf_file)
                shutil.copy2(pbf_path, pbf_dest)

                # print(f"Copied: {new_webp_name} and {pbf_file}")


def run(sat_dir, osm_dir, combined_dir):
    # Call the function to match and copy files
    match_and_copy_files(sat_dir, osm_dir, combined_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="pair_tiles",
        description="Combine satellite images and pbf tiles into training, validation, and test",
    )

    parser.add_argument(
        "-sat", "--sat_dir", type=str, help="Directory containing satellite images"
    )
    parser.add_argument(
        "-osm", "--osm_dir", type=str, help="Directory containing pbf files"
    )
    parser.add_argument(
        "-out", "--base_dir", type=str, help="Directory to put combined tiles"
    )

    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)
    run(args.sat_dir, args.osm_dir, args.base_dir)
