# 3DTeethSeg + TSegFormer Lambda Cloud Pipeline Guide

Step-by-step guide for training TSegFormer on the 3DTeethSeg MICCAI Challenge dataset using Lambda Cloud GPUs. Covers environment setup through evaluation and cleanup.

## Prerequisites

1. **Lambda Cloud account** — Sign up at [lambdalabs.com/cloud](https://lambdalabs.com/cloud). Add payment method and upload your SSH public key.
2. **SSH client** — For connecting to the Lambda instance.
3. **Local disk space** — ~500 MB to rsync back checkpoints and results.

---

## Step 1: Create a Lambda Cloud Persistent Filesystem

1. Go to **Lambda Console > Storage > Filesystems**
2. Click **Create Filesystem**
3. Name it **`teethseg`**
4. Choose a **region** — your instance must be in the same region
5. Set size to **100 GB** (dataset + checkpoints + environment)

> **Cost**: $0.20/GB/month = ~$20/month for 100 GB. Delete when done.

---

## Step 2: Launch a GPU Instance

1. Go to **Lambda Console > Instances > Launch Instance**
2. Select **1x A100 (80 GB)** — sufficient for TSegFormer (batch size 6, DataParallel on 2 GPUs if available)
3. Select the **same region** as your filesystem
4. Under **Attach Filesystems**, select `teethseg`
5. Select your SSH key
6. Launch

> **Larger instances**: 2x A100 gives faster training via DataParallel. TSegFormer defaults to 2 GPUs.

---

## Step 3: SSH Into the Instance

```bash
ssh ubuntu@<LAMBDA_IP>
```

Verify the filesystem:
```bash
ls /lambda/nfs/teethseg/
```

---

## Step 4: Clone the Repository

```bash
cd /lambda/nfs/teethseg/
mkdir -p repos
git clone --recurse-submodules https://github.com/sched088/Dental.git repos/Dental
```

Set up convenience variables:
```bash
SCRIPTS="/lambda/nfs/teethseg/repos/Dental/3DTeethSeg_Challenge/scripts/lambda"
```

---

## Step 5: Run Environment Setup (One-Time)

```bash
bash $SCRIPTS/setup_env.sh
```

**What it installs:**
- Python 3.10 via Miniconda
- PyTorch (latest) + CUDA 12.1
- scikit-learn, tqdm (TSegFormer dependencies)
- trimesh, numpy, scipy (preprocessing)
- osfclient (dataset download)

**Verification**: The script prints detected GPUs and confirms imports.

---

## Step 6: Activate the Environment

**Do this every time you SSH in:**

```bash
source $SCRIPTS/activate.sh
```

---

## Step 7: Download the Dataset

```bash
bash $SCRIPTS/download_data.sh
```

Downloads the 3DTeethSeg dataset from OSF (https://osf.io/xctdy/). Uses `osfclient` to fetch files, then extracts zip archives.

**Idempotent**: Uses a completion marker. Re-running skips if already downloaded.

---

## Step 8: Preprocess Data

```bash
python $SCRIPTS/preprocess_data.py
```

Converts 3DTeethSeg format (OBJ meshes + JSON annotations) to TSegFormer format (JSON with 8D point cloud features).

**What it does:**
1. Loads each OBJ mesh with trimesh
2. Computes 8D features per vertex: XYZ + normals + mean curvature + avg dihedral angle curvature
3. Maps FDI tooth labels to TSegFormer's 0-32 label scheme
4. Determines jaw type (mandible/maxillary) from FDI numbers
5. Outputs `{symbol}_aligned.json` files in TSegFormer's expected directory structure

**Options:**
- `--raw-dir /path/to/raw` — Custom raw data location
- `--output-dir /path/to/output` — Custom output location

---

## Step 9: Train TSegFormer

```bash
bash $SCRIPTS/train.sh
```

**Default settings**: 200 epochs, 10000 points per sample, batch size 6, SGD + cosine annealing.

**Customize:**
```bash
bash $SCRIPTS/train.sh 100 10000 8   # 100 epochs, 10k points, batch 8
```

Saves best model to `$NFS/work_dirs/teethseg/models/best_model.t7`.

---

## Step 10: Run Assessment

```bash
python $SCRIPTS/evaluate.py --tag trained
```

Computes: accuracy, balanced accuracy, mean IoU, F1, PPV, NPV, sensitivity, specificity.

Saves results to `$NFS/results/assessment_trained.json`.

---

## Step 11: Offload Results

Before terminating the instance:

```bash
bash $SCRIPTS/offload.sh
```

Then **from your local machine**:

```bash
rsync -avz ubuntu@<LAMBDA_IP>:/lambda/nfs/teethseg/offload/ ./teethseg_results/
```

---

## Step 12: Cleanup

```bash
bash $SCRIPTS/cleanup.sh
```

Interactive checklist that verifies results are offloaded before giving manual steps to terminate instance and delete filesystem.

---

## Running Everything at Once

```bash
bash $SCRIPTS/run_all.sh
```

Each phase checks for completion markers and skips if already done. Ctrl+C safe — re-run to resume.

---

## Known Issues

### TSegFormer Conv1d Input Channels

TSegFormer's `model.py` defines `conv1 = nn.Conv1d(7, 64, ...)` but the data has 8 features. If training crashes with a dimension mismatch, edit `vendor/TSegFormer/model.py` line 85:

```python
# Change from:
self.conv1 = nn.Conv1d(7, 64, kernel_size=1, bias=False)
# To:
self.conv1 = nn.Conv1d(8, 64, kernel_size=1, bias=False)
```

### TSegFormer Hardcoded GPU IDs

`main.py` line 20 hardcodes `CUDA_VISIBLE_DEVICES = '5,6'` and line 51 uses `device_ids=[0, 1]`. The `train.sh` script overrides `CUDA_VISIBLE_DEVICES` to `0,1`, but you may need to edit `main.py` if your instance has a different GPU topology.

---

## Cost Summary

| Phase | Duration | Cost (1x A100) |
|-------|----------|-----------------|
| Setup + download + preprocess | ~30 min | ~$1 |
| Training (200 epochs) | 2-4 hr | ~$4-8 |
| Assessment | ~10 min | ~$0.30 |
| Storage (100 GB, 1 week) | — | ~$5 |
| **Total** | **3-5 hr** | **$10-15** |
