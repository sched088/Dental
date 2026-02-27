#!/usr/bin/env bash
# train.sh — Train TSegFormer on preprocessed 3DTeethSeg data.
# Usage: bash train.sh [EPOCHS] [NUM_POINTS] [BATCH_SIZE]
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"
REPO_DIR="$NFS/repos/Dental"
PROJECT_DIR="$REPO_DIR/3DTeethSeg_Challenge"
TSEGFORMER_DIR="$PROJECT_DIR/vendor/TSegFormer"
DATA_DIR="$NFS/data/processed"
WORK_DIR="$NFS/work_dirs"

EPOCHS="${1:-200}"
NUM_POINTS="${2:-10000}"
BATCH_SIZE="${3:-6}"

echo "=== TSegFormer Training ==="
echo "Data:       $DATA_DIR"
echo "Work dir:   $WORK_DIR"
echo "Epochs:     $EPOCHS"
echo "Num points: $NUM_POINTS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Verify data exists
SAMPLE_COUNT=$(find "$DATA_DIR" -name "*_aligned.json" | wc -l)
if [ "$SAMPLE_COUNT" -eq 0 ]; then
    echo "ERROR: No preprocessed data found at $DATA_DIR"
    echo "Run preprocess_data.py first."
    exit 1
fi
echo "Found $SAMPLE_COUNT preprocessed samples"

mkdir -p "$WORK_DIR"

# Note: TSegFormer's main.py hardcodes CUDA_VISIBLE_DEVICES='5,6' and device_ids=[0,1].
# We need to override this for Lambda instances. We set CUDA_VISIBLE_DEVICES to use
# the first 2 available GPUs.
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs available: $GPU_COUNT"

# Use first 2 GPUs (TSegFormer uses DataParallel with 2 GPUs)
export CUDA_VISIBLE_DEVICES="0,1"

cd "$TSEGFORMER_DIR"

python main.py \
    --data_path "$DATA_DIR" \
    --save_path "$WORK_DIR" \
    --exp_name teethseg \
    --epochs "$EPOCHS" \
    --num_points "$NUM_POINTS" \
    --batch_size "$BATCH_SIZE" \
    --use_sgd True \
    --lr 0.001 \
    --scheduler cos

echo ""
echo "=== Training complete ==="
echo "Model saved at: $WORK_DIR/teethseg/models/best_model.t7"
