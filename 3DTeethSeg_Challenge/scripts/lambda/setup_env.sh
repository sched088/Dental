#!/usr/bin/env bash
# setup_env.sh — One-time environment setup on Lambda NFS.
# Run this once after launching your first instance.
# The conda environment persists on NFS across instance restarts.
set -euo pipefail

NFS="${LAMBDA_NFS:-/lambda/nfs/teethseg}"

echo "=== 3DTeethSeg + TSegFormer Environment Setup ==="
echo "NFS root: $NFS"

# Create directory structure
for dir in env repos data data/raw data/processed work_dirs results; do
    mkdir -p "$NFS/$dir"
done

# Install miniconda on NFS if not present
CONDA_DIR="$NFS/env/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda on NFS..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
fi
export PATH="$CONDA_DIR/bin:$PATH"

# Create environment if it doesn't exist
ENV_DIR="$NFS/env/teethseg"
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating conda environment..."
    conda create -y -p "$ENV_DIR" python=3.10
fi

# Activate
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "Installing PyTorch..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing TSegFormer dependencies..."
pip install -q scikit-learn tqdm

echo "Installing preprocessing dependencies..."
pip install -q trimesh numpy scipy

echo "Installing evaluation dependencies..."
pip install -q osfclient

# Clone repo if not present
REPO_DIR="$NFS/repos/Dental"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning Dental repo..."
    git clone https://github.com/sched088/Dental.git "$REPO_DIR"
    cd "$REPO_DIR"
    git submodule update --init --recursive
fi

# Verify GPU access
echo ""
echo "=== Verification ==="
python -c "
import torch
n = torch.cuda.device_count()
print(f'GPUs: {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_mem / 1e9:.1f} GB)')
print(f'CUDA: {torch.version.cuda}')
print(f'PyTorch: {torch.__version__}')
"

python -c "import trimesh; print(f'trimesh {trimesh.__version__} imported')"
python -c "import sklearn; print(f'scikit-learn {sklearn.__version__} imported')"

echo ""
echo "=== Setup complete ==="
echo "Activate in future sessions with: source \$SCRIPTS/activate.sh"
