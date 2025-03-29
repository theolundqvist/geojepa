#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
IMAGE_NAME="build_osm_tile_extractor"
PROJECT_DIR="$(pwd)"
BUILD_DIR_NAME="cmake-build-release-linux"
BUILD_DIR="$PROJECT_DIR/$BUILD_DIR_NAME"

# Create the build directory on host if it doesn't exist
mkdir -p "$BUILD_DIR"

# Build the Docker image
echo "Building Docker image: $IMAGE_NAME"
docker build -t "$IMAGE_NAME" .

# Run the Docker container to perform the build
echo "Running Docker container to build the project..."
sudo docker run --rm \
    -v "$PROJECT_DIR":/project \
    -v "$PROJECT_DIR/builds":/project/builds \
    -w "/project/$BUILD_DIR_NAME" \
    "$IMAGE_NAME" bash -c "
        cmake .. \
            -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
            -DCMAKE_BUILD_TYPE=Release && \
        make -j\$(nproc)
    "

  rm -rf "$BUILD_DIR"

# Notify completion
echo "Build completed. Artifacts are available in the 'builds/linux_x64' directory."
