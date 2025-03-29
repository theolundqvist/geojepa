This folder contains everything (almost) for sourcing and processing training data.

## Folder Structure

The `data` directory is organized as follows:

C++ Projects:
- **sat_tile_downloader/**: Tools to download and process satellite imagery tiles from NAIP.
- **osm_tile_extractor/**: Utilities to extract tiles from OpenStreetMap data.
- **tile_post_processor/**: Scripts and utilities to post-process of tiles after extraction, geometry -> visibility graph.
- **task_generator/**: Components responsible for generating training tasks from the tile data.

Data folder
- **tiles/**: Contains configs for the different datasets and the associated processed tiles. This folder also contains scripts for generating os specific makefiles and building the CPP projects and running the data processing steps.

Additional files:
- **logfile.txt**: Contains logs related to data processing operations.
- **.gitkeep**: Empty file used to ensure the directory is tracked by git even when empty.


### Ps.
built datasets are available through the links in the main readme.