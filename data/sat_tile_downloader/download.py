import argparse
import asyncio
import os
import sys
from urllib.parse import urlparse

import aiohttp
import cartopy.crs as ccrs
import click
import matplotlib.pyplot as plt
import planetary_computer
from pystac_client import Client
from tqdm.asyncio import tqdm_asyncio

# Configuration
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_ID = "naip"  # NAIP collection


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download NAIP imagery with specified parameters."
    )

    parser.add_argument(
        "--download_dir",
        type=str,
        default="NAIP",
        help="Directory to save downloaded data. Default is 'NAIP'.",
    )
    parser.add_argument(
        "--max_concurrent_downloads",
        type=int,
        default=6,
        help="Max concurrent downloads. Adjust based on your bandwidth and system capabilities. Default is 6.",
    )
    parser.add_argument(
        "--min_lon",
        type=float,
        default=-122.60,
        help="Minimum longitude of the bounding box. Default is -122.60.",
    )
    parser.add_argument(
        "--min_lat",
        type=float,
        default=37.224,
        help="Minimum latitude of the bounding box. Default is 37.224.",
    )
    parser.add_argument(
        "--max_lon",
        type=float,
        default=-121.125,
        help="Maximum longitude of the bounding box. Default is -121.125.",
    )
    parser.add_argument(
        "--max_lat",
        type=float,
        default=38.276,
        help="Maximum latitude of the bounding box. Default is 38.276.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2018-01-01",
        help="Start date for the latest images (format YYYY-MM-DD). Default is '2018-01-01'.",
    )
    parser.add_argument(
        "--max_downloads",
        type=int,
        default=100,
        help="Maximum number of downloads. Default is 100.",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--non-interactive", dest="interactive", action="store_false")
    parser.set_defaults(interactive=True)

    return parser.parse_args()


async def query_stac(args):
    """Query the STAC API for NAIP items within the specified bounding box and date range.

    Returns a list of asset URLs, selecting only the latest image per tile.
    """
    try:
        # Initialize STAC client
        client = Client.open(STAC_API_URL)

        # Perform the search with date filter
        search = client.search(
            collections=[COLLECTION_ID],
            bbox=[args.min_lon, args.min_lat, args.max_lon, args.max_lat],
            datetime=f"{args.start_date}T00:00:00Z/..",
            # limit=10000,  # Adjust as needed
        )

        items = list(search.items())

        if not items:
            print("No items found for the specified area and date range.")
            sys.exit(0)

        print(f"Found {len(items)} items in the specified area and date range.")

        # Dictionary to hold the latest asset per tile
        latest_assets = {}

        for item in items:
            # Extract acquisition date
            try:
                acquisition_date = item.datetime
            except AttributeError:
                # Fallback if datetime is not directly available
                acquisition_date = item.properties.get("datetime", None)
                if not acquisition_date:
                    print(
                        f"Item {item.id} does not have a datetime property. Skipping."
                    )
                    continue

            for asset_key, asset in item.assets.items():
                if "image/tiff" in asset.media_type:  # Adjust based on asset type
                    # Assume that the tile name can be extracted from the asset href
                    parsed_url = urlparse(asset.href)
                    path_parts = parsed_url.path.split("/")
                    name = path_parts[-1]
                    quadrangle = name.split("_")[1]
                    quarter_quad = name.split("_")[2]
                    tile_key = f"{quadrangle}_{quarter_quad}"

                    # Check if this tile_key is already in the dictionary
                    if tile_key in latest_assets:
                        existing_date = latest_assets[tile_key]["datetime"]
                        if acquisition_date > existing_date:
                            # Replace with the newer asset
                            latest_assets[tile_key] = {
                                "href": asset.href,
                                "datetime": acquisition_date,
                            }
                    else:
                        latest_assets[tile_key] = {
                            "href": asset.href,
                            "datetime": acquisition_date,
                        }

        # Extract the signed URLs of the latest assets
        asset_urls = [
            planetary_computer.sign(asset_info["href"])
            for asset_info in latest_assets.values()
        ]

        print(f"Total latest assets to download: {len(asset_urls)}")
        return asset_urls

    except Exception as e:
        print(f"Error querying STAC API: {e}")
        sys.exit(1)


async def download_blob(args, session, blob_url, download_path):
    """Download a single blob using aiohttp."""
    try:
        os.makedirs(os.path.dirname(download_path), exist_ok=True)
        async with session.get(blob_url) as response:
            response.raise_for_status()
            with open(download_path, "wb") as f:
                async for chunk in response.content.iter_chunked(
                    1024 * 1024 * 50
                ):  # 50MB chunks
                    f.write(chunk)
    except Exception as e:
        print(f"Error downloading {blob_url}: {e}")


async def download_all_assets(args, asset_urls):
    """Download all assets concurrently with a limit on maximum concurrent downloads.

    Skips downloading files that already exist in the DOWNLOAD_DIR.
    """
    semaphore = asyncio.Semaphore(args.max_concurrent_downloads)
    async with aiohttp.ClientSession() as session:
        tasks = []
        skipped_files = 0  # Counter for skipped files

        for url in asset_urls:
            # Extract blob name from URL
            parsed_url = urlparse(url)
            blob_name = os.path.basename(parsed_url.path)
            download_path = os.path.join(args.download_dir, blob_name)

            if os.path.exists(download_path):
                # print(f"Skipping download; file already exists: {download_path}")
                skipped_files += 1
                continue  # Skip adding a download task for this file

            async def sem_download(url=url, path=download_path):
                async with semaphore:
                    await download_blob(args, session, url, path)

            tasks.append(sem_download())

        total_tasks = len(tasks)
        if total_tasks == 0:
            print("All files already exist. No downloads needed.")
            return

        print(
            f"Starting download of {total_tasks} asset(s) to '{args.download_dir}' directory..."
        )
        print(f"Skipped {skipped_files} existing file(s).")

        # Use tqdm for progress bar
        for f in tqdm_asyncio.as_completed(
            tasks, total=total_tasks, desc="Downloading Assets"
        ):
            await f

        print("Download process completed.")


async def main(args):
    """Main function to orchestrate the download process."""
    print("Querying STAC API for NAIP items in San Francisco area...")
    asset_urls = await query_stac(args)

    if not asset_urls:
        print("No assets to download.")
        return

    print(f"Starting download to '{args.download_dir}' directory...")
    await download_all_assets(args, asset_urls[: args.max_downloads])
    print("Download completed successfully.")


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    if args.interactive:
        # Create a new plot
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Set extent to the bounding box
        ax.set_extent(
            [args.min_lon - 2, args.max_lon + 2, args.min_lat - 2, args.max_lat + 2],
            crs=ccrs.PlateCarree(),
        )

        # Add geographical features
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Plot the bounding box
        plt.plot(
            [args.min_lon, args.max_lon, args.max_lon, args.min_lon, args.min_lon],
            [args.min_lat, args.min_lat, args.max_lat, args.max_lat, args.min_lat],
            color="red",
            linewidth=2,
            linestyle="--",
            label="Bounding Box",
        )

        # Add labels and title
        plt.title("Bounding Box on Map")
        plt.legend()

        # Show the plot
        plt.show(block=False)

        if not click.confirm("Do you want to continue?", default=True):
            plt.close()
            exit(0)

        plt.close()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nDownload interrupted by user. Exiting...")
