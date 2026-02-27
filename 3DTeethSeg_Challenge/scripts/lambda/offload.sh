#!/usr/bin/env bash
# offload.sh — Copy important results to a staging directory for rsync.
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"
OFFLOAD_DIR="$NFS/offload"

echo "=== Offloading Results ==="

mkdir -p "$OFFLOAD_DIR"/{models,results,logs}

# Copy model checkpoints
if [ -d "$NFS/work_dirs/teethseg/models" ]; then
    cp -v "$NFS/work_dirs/teethseg/models/"*.t7 "$OFFLOAD_DIR/models/" 2>/dev/null || echo "No .t7 models found"
fi

# Copy training logs
if [ -f "$NFS/work_dirs/teethseg/run.log" ]; then
    cp -v "$NFS/work_dirs/teethseg/run.log" "$OFFLOAD_DIR/logs/"
fi

# Copy evaluation results
if [ -d "$NFS/results" ]; then
    cp -v "$NFS/results/"*.json "$OFFLOAD_DIR/results/" 2>/dev/null || echo "No result JSONs found"
fi

echo ""
echo "=== Offload complete ==="
echo "Files staged at: $OFFLOAD_DIR"
echo ""
echo "To copy to your local machine:"
echo "  rsync -avz ubuntu@<LAMBDA_IP>:$OFFLOAD_DIR/ ./teethseg_results/"
