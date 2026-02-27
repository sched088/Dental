#!/usr/bin/env bash
# run_all.sh — Run the full pipeline: download → preprocess → train → assess.
# Each phase checks for completion markers and skips if already done.
# Ctrl+C safe — re-run to resume from the last completed phase.
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"
REPO_DIR="$NFS/repos/Dental"
SCRIPTS="$REPO_DIR/3DTeethSeg_Challenge/scripts/lambda"

echo "=== 3DTeethSeg + TSegFormer Full Pipeline ==="
echo "NFS: $NFS"
echo ""

# Phase 1: Download data
echo "--- Phase 1: Download Data ---"
bash "$SCRIPTS/download_data.sh"
echo ""

# Phase 2: Preprocess
PREPROCESS_MARKER="$NFS/data/processed/.preprocess_complete"
if [ -f "$PREPROCESS_MARKER" ]; then
    echo "--- Phase 2: Preprocess (already complete, skipping) ---"
else
    echo "--- Phase 2: Preprocess Data ---"
    python "$SCRIPTS/preprocess_data.py"
    touch "$PREPROCESS_MARKER"
fi
echo ""

# Phase 3: Train
TRAIN_MARKER="$NFS/work_dirs/teethseg/models/best_model.t7"
if [ -f "$TRAIN_MARKER" ]; then
    echo "--- Phase 3: Training (model found, skipping) ---"
    echo "  To retrain, remove: $TRAIN_MARKER"
else
    echo "--- Phase 3: Train TSegFormer ---"
    bash "$SCRIPTS/train.sh"
fi
echo ""

# Phase 4: Assessment
echo "--- Phase 4: Assessment ---"
python "$SCRIPTS/evaluate.py" --tag trained
echo ""

echo "=== Pipeline Complete ==="
echo "Results at: $NFS/results/"
echo "Run offload.sh to stage results for download."
