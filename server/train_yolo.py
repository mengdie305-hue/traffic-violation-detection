#!/usr/bin/env python3
"""
YOLOv8 模型训练脚本 — 在 GPU 服务器上运行。

支持：
- 从零训练 / 微调预训练模型
- 自定义数据集（COCO 格式 / YOLO 格式）
- 数据增强、学习率调度、早停

Usage:
    python server/train_yolo.py \
        --data /path/to/dataset.yaml \
        --model yolov8n.pt \
        --epochs 100 \
        --batch 16 \
        --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Traffic Detection")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset YAML (YOLO format)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base model: yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device: 0 / 0,1 / cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--output", type=str, default="runs/traffic_yolo",
                        help="Output directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8 Training — Traffic Detection")
    print("=" * 60)
    print(f"  Dataset:  {args.data}")
    print(f"  Model:    {args.model}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch}")
    print(f"  Device:   {args.device}")
    print("=" * 60)

    # 加载模型
    model = YOLO(args.model)

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        lr0=args.lr,
        patience=args.patience,
        project=args.output,
        name="train",
        resume=args.resume is not None,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    # 验证
    print("\nEvaluating best model...")
    metrics = model.val()

    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"  Best model: {results.save_dir}/weights/best.pt")
    print(f"  mAP@50:     {metrics.box.map50:.4f}")
    print(f"  mAP@50-95:  {metrics.box.map:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
