#!/usr/bin/env python3
"""
模型评估脚本 — 在测试集上评估完整 Pipeline。

Usage:
    python server/evaluate.py \
        --model data/models/best.pt \
        --data /path/to/test_videos/ \
        --ground-truth /path/to/annotations.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--ground-truth", type=str, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def compute_iou(boxA: Tuple, boxB: Tuple) -> float:
    """计算两个边界框的 IoU。"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = inter_area / float(boxA_area + boxB_area - inter_area)
    return iou


def calculate_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> float:
    """
    计算 Average Precision (简化版)。

    Args:
        predictions: [{"bbox": [x1,y1,x2,y2], "class": int, "score": float}, ...]
        ground_truths: [{"bbox": [x1,y1,x2,y2], "class": int}, ...]

    Returns:
        Average Precision
    """
    # 按置信度排序
    predictions.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))
    gt_matched = set()

    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt = -1

        for j, gt in enumerate(ground_truths):
            if j in gt_matched:
                continue
            if pred["class"] != gt["class"]:
                continue
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if best_iou >= iou_threshold:
            tp[i] = 1
            gt_matched.add(best_gt)
        else:
            fp[i] = 1

    # 累积
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / len(ground_truths) if ground_truths else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

    # 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        if np.any(recalls >= t):
            ap += np.max(precisions[recalls >= t]) / 11

    return ap


def main():
    args = parse_args()

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    video_dir = Path(args.data)
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))

    results = {
        "model": args.model,
        "num_videos": len(video_files),
        "videos": [],
    }

    all_aps = []

    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")

        # TODO: 运行完整 Pipeline 并收集预测结果
        # predictions = run_pipeline(video_path, args.model, args.conf)
        # ground_truths = load_ground_truth(video_path, args.ground_truth)
        # ap = calculate_ap(predictions, ground_truths)

        # For now placeholder
        predictions = []
        ground_truths = []
        ap = 0.0

        all_aps.append(ap)
        results["videos"].append({
            "name": video_path.name,
            "predictions": len(predictions),
            "ground_truths": len(ground_truths),
            "ap": ap,
        })

    results["mAP"] = float(np.mean(all_aps)) if all_aps else 0.0

    # 保存
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Evaluation Complete — mAP: {results['mAP']:.4f}")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
