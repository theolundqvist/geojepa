#!/bin/bash

# =============================================================================
# Script: build_all.sh
# Description:
#   Sets executable permissions and builds multiple components.
#   Exits immediately if any build fails.
#
# Usage:
#   ./build_all.sh
#
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e
cd "$(dirname "$0")"

# -------------------- Configuration --------------------
# Define an array of directories to build
BUILD_DIRS=(
    #"../osm_tile_extractor"
    "../sat_tile_downloader"
    "../task_generator"
    "../tile_post_processor"
    "../osm_tile_extractor"
)
# --------------------------------------------------------

# Function to build a single component
build_component() {
    local dir="$1"
    local name
    name=$(basename "$dir")

    echo "Building $name..."
    (
        cd "$dir" && sudo bash build_linux.sh
    ) || { 
        echo "Failed to build $name" 
        exit 1 
    }

    echo "$name built successfully."
    echo "------------------------------------------"
}

# Iterate over each build directory and build the component
for dir in "${BUILD_DIRS[@]}"; do
    build_component "$dir"
done

echo "All components built successfully."