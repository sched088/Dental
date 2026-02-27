#!/usr/bin/env bash
# cleanup.sh — Interactive checklist before terminating instance and filesystem.
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"

echo "=== Pre-Cleanup Checklist ==="
echo ""

# Check if results have been offloaded
if [ -d "$NFS/offload" ] && [ "$(ls -A "$NFS/offload/models" 2>/dev/null)" ]; then
    echo "[OK] Results have been offloaded to $NFS/offload/"
else
    echo "[!!] Results have NOT been offloaded. Run offload.sh first!"
fi

# Check if model exists
if ls "$NFS/work_dirs/teethseg/models/"*.t7 1>/dev/null 2>&1; then
    echo "[OK] Trained model found"
else
    echo "[--] No trained model found (may not have trained yet)"
fi

# Check evaluation results
if ls "$NFS/results/"*.json 1>/dev/null 2>&1; then
    echo "[OK] Assessment results found"
else
    echo "[--] No assessment results found"
fi

echo ""
echo "=== Manual Cleanup Steps ==="
echo ""
echo "1. TERMINATE the instance (stops hourly billing):"
echo "   Lambda Console > Instances > Select > Terminate"
echo ""
echo "2. DELETE the filesystem (stops storage billing):"
echo "   Lambda Console > Storage > Filesystems > Select > Delete"
echo ""
echo "   WARNING: Filesystem deletion is permanent and irreversible."
echo "   Make sure you've rsync'd all results to your local machine first."
