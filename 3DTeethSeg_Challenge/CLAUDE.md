# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D teeth segmentation evaluation project using the **3DTeethSeg MICCAI Challenge** dataset with the **TSegFormer** model. The goal is to run inference and evaluate segmentation quality — not to train from scratch.

## Key External Dependencies

- **Dataset**: 3DTeethSeg MICCAI Challenge — https://osf.io/xctdy/overview (using 2 of 6 zip files)
- **Model**: TSegFormer — https://github.com/huiminxiong/TSegFormer/
- **Evaluation script**: from https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges

## Data Location

Datasets are stored locally at `~/Documents/Data/3DTeethSeg_Challenge/*` (not in this repo). Split files may need verification to confirm correct data mapping.
