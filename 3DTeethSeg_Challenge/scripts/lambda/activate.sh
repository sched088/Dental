#!/usr/bin/env bash
# activate.sh — Source this on each SSH session to activate the environment.
# Usage: source scripts/lambda/activate.sh

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"
CONDA_DIR="$NFS/env/miniconda3"
ENV_DIR="$NFS/env/teethseg"

if [ ! -d "$CONDA_DIR" ]; then
    echo "ERROR: Miniconda not found at $CONDA_DIR"
    echo "Run setup_env.sh first."
    return 1 2>/dev/null || exit 1
fi

export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

export LAMBDA_NFS="$NFS"
export REPO_DIR="$NFS/repos/Dental"
export PROJECT_DIR="$REPO_DIR/3DTeethSeg_Challenge"
export SCRIPTS="$PROJECT_DIR/scripts/lambda"
export PYTHONPATH="$PROJECT_DIR/vendor/TSegFormer:${PYTHONPATH:-}"

echo "Environment activated. NFS=$NFS, GPUs=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '?')"
