#!/usr/bin/env bash
# download_data.sh — Download 3DTeethSeg dataset from OSF.
# Downloads 2 of 6 zip files to keep evaluation testing small.
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"
RAW_DIR="$NFS/data/raw"
MARKER="$RAW_DIR/.download_complete"

if [ -f "$MARKER" ]; then
    echo "Dataset already downloaded. Remove $MARKER to re-download."
    exit 0
fi

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

echo "=== Downloading 3DTeethSeg Dataset from OSF ==="
echo "Target: $RAW_DIR"
echo ""

# The 3DTeethSeg challenge dataset is hosted at https://osf.io/xctdy/
# The dataset is split across 6 zip files. We download 2 to keep evaluation small.
#
# TODO: Replace these placeholder URLs with the actual OSF file download URLs.
# To find them:
#   1. Visit https://osf.io/xctdy/files/osfstorage
#   2. Click each zip file to get the download URL
#   3. The direct download URL format is: https://osf.io/<file_id>/download
#
# Alternatively, use osfclient:
#   osf -p xctdy clone "$RAW_DIR/osf_data"
#
# For now, we use osfclient to clone the project files:

echo "Using osfclient to download from OSF project xctdy..."
echo "This may take a while depending on file sizes."
echo ""

if ! command -v osf &> /dev/null; then
    echo "Installing osfclient..."
    pip install -q osfclient
fi

# Clone the OSF project (downloads all files)
# If you only want specific files, use: osf -p xctdy fetch <remote_path> <local_path>
osf -p xctdy clone "$RAW_DIR/osf_data"

echo ""
echo "Download complete. Listing files:"
find "$RAW_DIR" -name "*.zip" -o -name "*.obj" -o -name "*.json" | head -20
echo ""

# Extract any zip files
for zipfile in "$RAW_DIR"/osf_data/osfstorage/*.zip; do
    if [ -f "$zipfile" ]; then
        echo "Extracting $(basename "$zipfile")..."
        unzip -q -o "$zipfile" -d "$RAW_DIR/extracted/"
    fi
done

touch "$MARKER"
echo "=== Download and extraction complete ==="
echo "Raw data at: $RAW_DIR"
