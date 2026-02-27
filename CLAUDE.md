# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is the top-level Dental research repo. Each subdirectory is a self-contained project.

- **3DTeethSeg_Challenge/**: 3D teeth segmentation using TSegFormer on the 3DTeethSeg MICCAI dataset

## Submodules

External repos are included as git submodules under each project's `vendor/` directory. After cloning:

```bash
git submodule update --init --recursive
```

## Conventions

- Lambda Cloud GPU pipeline scripts live in `<project>/scripts/lambda/`
- Each project has its own `CLAUDE.md` with project-specific details
- Each project has a `LAMBDA_PIPELINE_GUIDE.md` with step-by-step cloud training instructions
