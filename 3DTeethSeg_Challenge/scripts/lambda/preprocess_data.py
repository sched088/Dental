#!/usr/bin/env python3
"""
preprocess_data.py — Convert 3DTeethSeg OBJ meshes + JSON annotations
to TSegFormer's expected JSON format.

Input format (3DTeethSeg):
  - OBJ files: 3D mesh with vertices and faces
  - JSON files: {"labels": [...], "instances": [...]}
    - labels: FDI tooth number per vertex (0 = gingiva)
    - instances: instance ID per vertex

Output format (TSegFormer):
  - JSON files: {"feature": [[...], ...], "label": [...], "category": [...]}
    - feature: Nx8 array (XYZ + normals + mean_curvature + avg_angle_curvature)
    - label: N integers (0-32, where 0 = gingiva, 1-16 per jaw type)
    - category: [1,0] for mandible / [0,1] for maxillary

Directory structure expected by TSegFormer:
  ROOT_PATH/
    sample_001/
      L_aligned.json   (mandible, symbol = "L")
    sample_002/
      U_aligned.json   (maxillary, symbol = "U")
"""

import argparse
import json
import os
import glob
import numpy as np
import trimesh
from pathlib import Path


# FDI numbering: 11-18, 21-28 = maxillary (upper jaw)
#                 31-38, 41-48 = mandible (lower jaw)
# TSegFormer uses labels 1-16 per jaw (+ 0 for gingiva), total 33 classes.
# For mandible predictions, main.py adds +16 offset: seg_pred[seg_pred > 0] += 16

FDI_TO_TSEG = {}
# Upper right quadrant: FDI 11-18 -> TSegFormer 1-8
for i in range(8):
    FDI_TO_TSEG[11 + i] = 1 + i
# Upper left quadrant: FDI 21-28 -> TSegFormer 9-16
for i in range(8):
    FDI_TO_TSEG[21 + i] = 9 + i
# Lower left quadrant: FDI 31-38 -> TSegFormer 1-8 (mandible uses same 1-16 range)
for i in range(8):
    FDI_TO_TSEG[31 + i] = 1 + i
# Lower right quadrant: FDI 41-48 -> TSegFormer 9-16
for i in range(8):
    FDI_TO_TSEG[41 + i] = 9 + i


def is_mandible(fdi_labels):
    """Determine jaw type from FDI labels. Mandible = 31-48, maxillary = 11-28."""
    non_gingiva = fdi_labels[fdi_labels > 0]
    if len(non_gingiva) == 0:
        return None  # skip — no teeth labeled
    median_label = np.median(non_gingiva)
    return median_label >= 30


def compute_vertex_curvatures(mesh):
    """
    Compute per-vertex curvature estimates from a trimesh mesh.

    Returns:
        mean_curvature: (N,) mean curvature per vertex
        avg_angle_curvature: (N,) average dihedral angle deviation per vertex
            (the feature TSegFormer uses for geometry-guided loss)
    """
    n_verts = len(mesh.vertices)

    # Mean curvature via discrete Laplace-Beltrami
    # trimesh doesn't have a direct mean curvature function,
    # so we approximate using the face adjacency angles.
    mean_curvature = np.zeros(n_verts, dtype=np.float32)
    avg_angle_curvature = np.zeros(n_verts, dtype=np.float32)

    if hasattr(mesh, 'face_adjacency_angles') and len(mesh.face_adjacency_angles) > 0:
        # face_adjacency_angles: angle between adjacent face normals (dihedral angle)
        # face_adjacency: pairs of adjacent face indices
        angles = mesh.face_adjacency_angles  # (M,) in radians
        adj_faces = mesh.face_adjacency  # (M, 2) face index pairs

        # For each edge between adjacent faces, attribute the dihedral angle
        # to the shared vertices
        vertex_angle_sum = np.zeros(n_verts, dtype=np.float64)
        vertex_angle_count = np.zeros(n_verts, dtype=np.int32)

        for idx, (f1, f2) in enumerate(adj_faces):
            angle = angles[idx]
            # Find shared vertices between the two faces
            shared = np.intersect1d(mesh.faces[f1], mesh.faces[f2])
            for v in shared:
                vertex_angle_sum[v] += angle
                vertex_angle_count[v] += 1

        mask = vertex_angle_count > 0
        avg_angle_curvature[mask] = (
            vertex_angle_sum[mask] / vertex_angle_count[mask]
        ).astype(np.float32)

        # Mean curvature approximation: deviation from pi (flat surface)
        mean_curvature[mask] = (
            np.abs(vertex_angle_sum[mask] / vertex_angle_count[mask] - np.pi)
        ).astype(np.float32)

    return mean_curvature, avg_angle_curvature


def process_scan(obj_path, json_path, output_dir, sample_name):
    """
    Convert a single 3DTeethSeg scan (OBJ + JSON) to TSegFormer format.

    Args:
        obj_path: Path to OBJ mesh file
        json_path: Path to JSON annotation file
        output_dir: Root output directory for TSegFormer data
        sample_name: Name for the output subdirectory
    """
    # Load mesh
    mesh = trimesh.load(obj_path, process=False)

    # Load annotations
    with open(json_path, 'r') as f:
        annotations = json.load(f)

    fdi_labels = np.array(annotations['labels'], dtype=np.int32)
    vertices = np.array(mesh.vertices, dtype=np.float32)
    n_verts = len(vertices)

    assert n_verts == len(fdi_labels), (
        f"Vertex count mismatch: mesh has {n_verts}, labels has {len(fdi_labels)}"
    )

    # Determine jaw type
    mandible = is_mandible(fdi_labels)
    if mandible is None:
        print(f"  Skipping {sample_name}: no teeth labeled")
        return False

    category = [1, 0] if mandible else [0, 1]
    symbol = "L" if mandible else "U"  # L = lower (mandible), U = upper (maxillary)

    # Map FDI labels to TSegFormer labels (0-16 per jaw, 0 = gingiva)
    tseg_labels = np.zeros(n_verts, dtype=np.int64)
    for i, fdi in enumerate(fdi_labels):
        if fdi == 0:
            tseg_labels[i] = 0  # gingiva
        elif fdi in FDI_TO_TSEG:
            tseg_labels[i] = FDI_TO_TSEG[fdi]
        else:
            print(f"  Warning: Unknown FDI label {fdi} in {sample_name}, treating as gingiva")
            tseg_labels[i] = 0

    # Compute vertex normals
    if mesh.vertex_normals is not None and len(mesh.vertex_normals) == n_verts:
        normals = np.array(mesh.vertex_normals, dtype=np.float32)
    else:
        # Fallback: estimate normals from faces
        normals = np.zeros((n_verts, 3), dtype=np.float32)

    # Compute curvatures
    mean_curvature, avg_angle_curvature = compute_vertex_curvatures(mesh)

    # Build 8D feature vector: [x, y, z, nx, ny, nz, mean_curv, avg_angle_curv]
    features = np.column_stack([
        vertices,           # (N, 3) XYZ
        normals,            # (N, 3) vertex normals
        mean_curvature,     # (N, 1) mean curvature
        avg_angle_curvature # (N, 1) avg dihedral angle curvature
    ])

    # Write output in TSegFormer format
    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)
    output_path = os.path.join(sample_dir, f"{symbol}_aligned.json")

    output_data = {
        "feature": features.tolist(),
        "label": tseg_labels.tolist(),
        "category": category
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    return True


def find_scan_pairs(raw_dir):
    """
    Find OBJ + JSON file pairs in the raw data directory.
    Handles multiple possible directory structures from 3DTeethSeg.
    """
    pairs = []

    # Pattern 1: OBJ and JSON in same directory, matching names
    for obj_file in sorted(glob.glob(os.path.join(raw_dir, "**/*.obj"), recursive=True)):
        obj_path = Path(obj_file)
        # Look for corresponding JSON annotation
        json_candidates = [
            obj_path.with_suffix('.json'),
            obj_path.parent / (obj_path.stem + '_labels.json'),
            obj_path.parent / 'labels' / (obj_path.stem + '.json'),
        ]
        for json_path in json_candidates:
            if json_path.exists():
                pairs.append((str(obj_path), str(json_path)))
                break

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Preprocess 3DTeethSeg data for TSegFormer')
    parser.add_argument('--raw-dir', type=str,
                        default=os.path.join(os.environ.get('LAMBDA_NFS', '/lambda/nfs/teethseg'), 'data/raw'),
                        help='Directory containing raw OBJ + JSON files')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.environ.get('LAMBDA_NFS', '/lambda/nfs/teethseg'), 'data/processed'),
                        help='Output directory for TSegFormer-format JSON files')
    args = parser.parse_args()

    print(f"=== Preprocessing 3DTeethSeg → TSegFormer format ===")
    print(f"Raw data: {args.raw_dir}")
    print(f"Output:   {args.output_dir}")

    # Find all scan pairs
    pairs = find_scan_pairs(args.raw_dir)
    if not pairs:
        print(f"ERROR: No OBJ + JSON pairs found in {args.raw_dir}")
        print("Expected: .obj files with matching .json annotation files")
        return

    print(f"Found {len(pairs)} scan pairs")
    os.makedirs(args.output_dir, exist_ok=True)

    success_count = 0
    for i, (obj_path, json_path) in enumerate(pairs):
        sample_name = f"sample_{i:04d}"
        print(f"  [{i+1}/{len(pairs)}] {Path(obj_path).name} → {sample_name}")
        try:
            if process_scan(obj_path, json_path, args.output_dir, sample_name):
                success_count += 1
        except Exception as e:
            print(f"  ERROR processing {obj_path}: {e}")

    print(f"\n=== Preprocessing complete ===")
    print(f"Successfully converted: {success_count}/{len(pairs)}")
    print(f"Output directory: {args.output_dir}")

    # Summary stats
    mandible_count = len(glob.glob(os.path.join(args.output_dir, "*/L_aligned.json")))
    maxillary_count = len(glob.glob(os.path.join(args.output_dir, "*/U_aligned.json")))
    print(f"Mandible (lower jaw): {mandible_count}")
    print(f"Maxillary (upper jaw): {maxillary_count}")
    print(f"\nTSegFormer splits (based on directory order):")
    print(f"  val:   first 2000 samples")
    print(f"  test:  last 2000 samples")
    print(f"  train: first 4000 samples")


if __name__ == '__main__':
    main()
