#!/bin/bash

# Configuration Variables
PIXELS_PER_TILE_16=224
NUM_THREADS=12

ZOOM_LEVEL=14
TILE_SIZE=$((PIXELS_PER_TILE_16 * (16 - ZOOM_LEVEL) * 2))
echo "TILE_SIZE=$TILE_SIZE"

# Function to get file size in bytes, compatible with Linux and macOS
get_file_size() {
    local file="$1"
    local size=0

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        size=$(stat -c%s "$file" 2>/dev/null)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        size=$(stat -f%z "$file" 2>/dev/null)
    else
        echo "Unsupported OS type: $OSTYPE"
        exit 1
    fi

    echo "$size"
}

# Function to format sizes in human-readable form using Bash's built-in printf
human_readable_size() {
    local size=$1
    local units=("B" "KB" "MB" "GB" "TB")
    local unit_index=0

    while (( $(echo "$size >= 1024" | bc -l) )) && [ $unit_index -lt $((${#units[@]} - 1)) ]; do
        size=$(echo "scale=2; $size / 1024" | bc)
        unit_index=$((unit_index + 1))
    done

    printf "%.2f%s" "$size" "${units[$unit_index]}"
}

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_cores> <directory> <output_directory>"
    exit 1
fi

NUM_CORES="$1"
DIRECTORY="$2"
TILES_DIR="$3"

# Verify that the provided argument is a directory
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: '$DIRECTORY' is not a directory."
    exit 1
fi
echo "Starting tile.sh with $NUM_CORES cores, processing '$DIRECTORY' to '$TILES_DIR'"

# Get the list of .tif files
shopt -s nullglob
FILES=("$DIRECTORY"/*.tif)
shopt -u nullglob

# Check if there are any .tif files
if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No .tif files found in the directory."
    exit 1
fi

TOTAL_TIF_FILES=${#FILES[@]}
echo "Found $TOTAL_TIF_FILES .tif file(s) in '$DIRECTORY'."

# Display size of each .tif file and calculate total size
TOTAL_TIF_SIZE=0
echo "Sizes of .tif files:"
for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    FILE_SIZE_BYTES=$(get_file_size "$FILE")

    # Check if stat was successful
    if [ -z "$FILE_SIZE_BYTES" ]; then
        echo "  - $BASENAME: Unable to determine size."
        continue
    fi

    TOTAL_TIF_SIZE=$(echo "$TOTAL_TIF_SIZE + $FILE_SIZE_BYTES" | bc)
    FORMATTED_SIZE=$(human_readable_size "$FILE_SIZE_BYTES")
    echo "  - $BASENAME: $FORMATTED_SIZE"
done

FORMATTED_TOTAL_TIF_SIZE=$(human_readable_size "$TOTAL_TIF_SIZE")
echo "Total size of all .tif files: $FORMATTED_TOTAL_TIF_SIZE"

mkdir -p "$TILES_DIR"
echo "Starting processing of $TOTAL_TIF_FILES .tif file(s) in '$DIRECTORY'..."

# Initialize a log file to keep track of failed files
mkdir -p logs
LOG_DIR="${CLOUD_ARTIFACT_DIR:-logs}"
FAILED_FILES_LOG="$LOG_DIR/failed_files.log"
JOBLOG="$LOG_DIR/parallel_joblog.log"
echo "Failed files will be logged to: $FAILED_FILES_LOG"


# Clean previous logs if they exist
rm -f "$FAILED_FILES_LOG" "$JOBLOG"

# Export necessary variables and functions for GNU Parallel
export ZOOM_LEVEL
export TILE_SIZE
export TILES_DIR
export LOG_DIR
export FAILED_FILES_LOG

# Function to process a single file
process_file() {
    local FILE="$1"
    local BASENAME
    BASENAME=$(basename "$FILE")
    # Extract the base name without the directory path
    local OUTPUT_DIR="$TILES_DIR/$BASENAME"

    # Extract the filename without the extension
    FILENAME="${BASENAME%.*}"

    # Extract the file extension
    EXTENSION="${BASENAME##*.}"

    # Construct the new filename with '_proj' before the extension
    NEW_FILENAME="${FILENAME}_proj.${EXTENSION}"


    echo "Processing: $BASENAME"
    ERROR_LOG_FILE="${LOG_DIR}/${BASENAME}_error.log"
    # Run gdalwarp with the new filename
    gdalwarp -t_srs EPSG:3857 "$FILE" "$NEW_FILENAME" 2>> "$ERROR_LOG_FILE"

    # Remove the original file
    rm "$FILE"
    mv "$NEW_FILENAME" "$FILE"

    if gdal2tiles -z "${ZOOM_LEVEL}-${ZOOM_LEVEL}" -p mercator \
        -r bilinear \
        --tilesize="$TILE_SIZE" --tiledriver=WEBP --webp-lossless \
        "$FILE" "$OUTPUT_DIR" > /dev/null 2> "$ERROR_LOG_FILE"; then
        echo "✓ Successfully processed: $BASENAME"
    else
        echo "✗ Failed to process: $BASENAME"
        # Print the error message
        echo "Error details:"
        cat "$ERROR_LOG_FILE"
        # Append the failed filename to the log in a thread-safe manner
        echo "$BASENAME" >> "$FAILED_FILES_LOG"
    fi
    # -e exists and -s is not empty
    if [ -e "$ERROR_LOG_FILE" ] && [ ! -s "$ERROR_LOG_FILE" ]; then
        rm "$ERROR_LOG_FILE"
    fi
}

export -f process_file

# Start Parallel Processing
# --joblog logs job statuses, which can be used to determine successes/failures
# --eta shows the estimated time of arrival (completion)
# --bar provides a progress bar
printf "%s\n" "${FILES[@]}" | parallel \
    --jobs "$NUM_CORES" \
    --joblog "$JOBLOG" \
    --bar \
    process_file {}

echo "All processing jobs have been initiated. Waiting for completion..."

# Wait for all parallel jobs to finish
wait

echo "All files have been processed."

# Count the total number of files in the tiles directory recursively
TOTAL_TILE_FILES=$(find "$TILES_DIR" -type f | wc -l)
echo "Total number of tile files in '$TILES_DIR': $TOTAL_TILE_FILES"

# Calculate the total size of the tiles directory
# Using du to get disk usage in a human-readable format
if command -v du >/dev/null 2>&1; then
    TILES_SIZE=$(du -sh "$TILES_DIR" | awk '{print $1}')
    echo "Total size of 'tiles' directory: $TILES_SIZE"
else
    echo "Warning: 'du' command not found. Unable to determine the size of 'tiles' directory."
fi

# Report any failed files
if [ -f "$FAILED_FILES_LOG" ]; then
    FAILED_FILES=()
    while IFS= read -r line; do
        FAILED_FILES+=("$line")
    done < "$FAILED_FILES_LOG"

    if [ "${#FAILED_FILES[@]}" -ne 0 ]; then
        echo ""
        echo "The following files failed to process:"
        for FAILED_FILE in "${FAILED_FILES[@]}"; do
            echo "  - $FAILED_FILE"
        done
    else
        echo "All files processed successfully."
    fi

    # Clean up the failed files log
    rm -f "$FAILED_FILES_LOG"
else
    echo "All files processed successfully."
fi
