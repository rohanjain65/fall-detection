#!/bin/bash

BASE_URL="https://falldataset.com/data"
TRAIN_VIDEOS=("489" "569" "581" "722" "731" "758" "807" "1219" "1260" "1301" "1373" "1378" "1392" "1790" "1843" "1954")
VAL_VIDEOS=("1176" "2123")
TEST_VIDEOS=("832" "786" "925")

# Define output directory
OUTPUT_DIR="data"
if [ "$1" ]; then
    OUTPUT_DIR="$1"
fi

echo "Downloading Fall Detection Dataset to $OUTPUT_DIR..."

# Create main output directory and subdirectories
mkdir -p "$OUTPUT_DIR/train" "$OUTPUT_DIR/test" "$OUTPUT_DIR/val"

download_and_extract() {
    local video_id="$1"
    local split="$2"
    local url="$BASE_URL/$video_id/$video_id.tar.gz"
    local tar_path="$OUTPUT_DIR/$video_id.tar.gz"
    local extract_path="$OUTPUT_DIR/$split"

    echo "Downloading $video_id..."
    curl -L -o "$tar_path" "$url"
    
    echo "Extracting $video_id to $extract_path..."
    tar -xzf "$tar_path" -C "$extract_path"
    rm "$tar_path"
}

# Process each dataset split
declare -A SPLITS=(
    [train]="${TRAIN_VIDEOS[@]}"
    [test]="${TEST_VIDEOS[@]}"
    [val]="${VAL_VIDEOS[@]}"
)

for split in "${!SPLITS[@]}"; do
    for video_id in ${SPLITS[$split]}; do
        download_and_extract "$video_id" "$split"
    done
done

echo "Dataset setup complete."