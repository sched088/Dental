#!/usr/bin/env python3
"""
run_evaluation.py — Run TSegFormer inference and compute metrics.

This script:
1. Loads a trained TSegFormer model
2. Runs inference on the test set
3. Computes TSegFormer's internal metrics (accuracy, IoU, F1)
4. Saves results summary
"""

import argparse
import json
import os
import sys
import numpy as np

# Add TSegFormer to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TSEGFORMER_DIR = os.path.join(PROJECT_DIR, 'vendor', 'TSegFormer')
sys.path.insert(0, TSEGFORMER_DIR)


def run_tsegformer_test(data_path, model_path, num_points, work_dir):
    """Run TSegFormer's built-in test assessment."""
    import torch
    import torch.nn as nn
    from model import TSegFormer
    from data import Teeth
    from torch.utils.data import DataLoader
    from util import calculate_shape_IoU, calculate_metrics
    import sklearn.metrics as metrics
    import tqdm

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = Teeth(num_points=num_points, ROOT_PATH=data_path, partition="test")
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    print(f"Test samples: {len(test_dataset)}")

    # Load model — create a minimal args namespace
    class Args:
        pass
    args = Args()

    seg_num_all = 33
    model = TSegFormer(args, seg_num_all).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.set_to_inference_mode = lambda: None  # no-op

    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []

    with torch.no_grad():
        for data, label, seg in tqdm.tqdm(test_loader, desc="Running assessment"):
            data = data.to(device).permute(0, 2, 1)
            label = label.to(device)
            seg = seg.to(device)

            seg_pred, ging_pred = model(data, label)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.max(dim=2)[1]

            seg_np = seg.cpu().numpy()
            pred_np = pred.cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label[:, 0].type(torch.int32).reshape(-1))

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)

    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg)
    all_f1, all_ppv, all_npv, all_sensitivity, all_specificity = calculate_metrics(
        test_pred_seg, test_true_seg, test_label_seg
    )

    results = {
        "accuracy": float(test_acc),
        "balanced_accuracy": float(avg_per_class_acc),
        "mean_iou": float(np.mean(test_ious)),
        "mean_f1": float(np.mean(all_f1)),
        "mean_ppv": float(np.mean(all_ppv)),
        "mean_npv": float(np.mean(all_npv)),
        "mean_sensitivity": float(np.mean(all_sensitivity)),
        "mean_specificity": float(np.mean(all_specificity)),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Assess TSegFormer on 3DTeethSeg')
    nfs = os.environ.get('LAMBDA_NFS', '/lambda/nfs/teethseg')
    parser.add_argument('--data-path', type=str,
                        default=os.path.join(nfs, 'data/processed'),
                        help='Path to preprocessed TSegFormer data')
    parser.add_argument('--model-path', type=str,
                        default=os.path.join(nfs, 'work_dirs/teethseg/models/best_model.t7'),
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-points', type=int, default=10000,
                        help='Number of points per sample')
    parser.add_argument('--tag', type=str, default='default',
                        help='Tag for naming the results file')
    args = parser.parse_args()

    work_dir = os.path.join(nfs, 'results')
    os.makedirs(work_dir, exist_ok=True)

    print("=== TSegFormer Assessment ===")
    print(f"Data:  {args.data_path}")
    print(f"Model: {args.model_path}")
    print(f"Points: {args.num_points}")
    print()

    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        print("Train the model first with train.sh")
        sys.exit(1)

    results = run_tsegformer_test(args.data_path, args.model_path, args.num_points, work_dir)

    # Print results
    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    output_path = os.path.join(work_dir, f"assessment_{args.tag}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
