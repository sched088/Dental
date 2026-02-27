# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D teeth segmentation project: train TSegFormer on the 3DTeethSeg MICCAI Challenge dataset, then evaluate. Training runs on Lambda Cloud GPUs.

## External Dependencies (git submodules)

- **TSegFormer**: `vendor/TSegFormer/` — Transformer-based 3D tooth segmentation model (MICCAI 2023)
- **3DTeethSeg_MICCAI_Challenges**: `vendor/3DTeethSeg_MICCAI_Challenges/` — Dataset repo with evaluation script
- **Dataset**: 3DTeethSeg from OSF (https://osf.io/xctdy/) — using 2 of 6 zip files

## Data Format Gap

3DTeethSeg provides OBJ meshes + JSON annotations (FDI labels per vertex). TSegFormer expects preprocessed JSON files with 8D point cloud features. `scripts/lambda/preprocess_data.py` bridges this gap:
- 8D features: XYZ (3) + normals (3) + mean curvature (1) + avg dihedral angle curvature (1)
- FDI labels (11-48) mapped to TSegFormer labels (0-32)
- Output files: `{L|U}_aligned.json` per sample (L=mandible, U=maxillary)

## Lambda Cloud Pipeline

All scripts in `scripts/lambda/`. See `LAMBDA_PIPELINE_GUIDE.md` for step-by-step instructions.

```bash
SCRIPTS="$NFS/repos/Dental/3DTeethSeg_Challenge/scripts/lambda"
bash $SCRIPTS/setup_env.sh       # one-time environment setup
source $SCRIPTS/activate.sh      # activate on each SSH session
bash $SCRIPTS/download_data.sh   # download dataset from OSF
python $SCRIPTS/preprocess_data.py  # OBJ+JSON → TSegFormer JSON
bash $SCRIPTS/train.sh           # train TSegFormer (200 epochs default)
python $SCRIPTS/evaluate.py      # run assessment metrics
bash $SCRIPTS/run_all.sh         # or run everything at once
```

## Known TSegFormer Issues

- `model.py:85` has `Conv1d(7, 64, ...)` but data has 8 features — may need changing to `Conv1d(8, 64, ...)`
- `main.py:20` hardcodes `CUDA_VISIBLE_DEVICES = '5,6'` — `train.sh` overrides this to `0,1`
- `main.py:51` hardcodes `device_ids=[0, 1]` — needs editing for single-GPU instances

## Data Locations

- **Local dev**: `~/Documents/Data/3DTeethSeg_Challenge/*`
- **Lambda NFS**: `/lambda/nfs/teethseg/data/` (raw and processed)
