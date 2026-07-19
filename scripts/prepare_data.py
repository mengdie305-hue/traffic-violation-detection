#!/usr/bin/env python3
"""
数据预处理脚本。

功能：
1. 从视频抽帧（每 N 帧保存一张图片）
2. 生成 YOLO 格式标注文件模板
3. 数据集划分（train/val/test）

Usage:
    python scripts/prepare_data.py --video data/videos/sample.mp4 --output data/images/
"""

import argparse
import shutil
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/images/")
    parser.add_argument("--frame-interval", type=int, default=30,
                        help="Extract 1 frame every N frames")
    parser.add_argument("--max-images", type=int, default=500)
    return parser.parse_args()


def extract_frames(
    video_path: str, output_dir: str, interval: int, max_images: int
):
    """从视频中抽取关键帧。"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(output_dir)
    video_name = Path(video_path).stem
    extract_dir = output_dir / video_name
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f}, Total: {total_frames} frames")
    print(f"Extracting every {interval} frames → {extract_dir}/")

    saved = 0
    frame_idx = 0

    while saved < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            out_path = extract_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved % 50 == 0:
                print(f"  Saved {saved} images...")

        frame_idx += 1

    cap.release()
    print(f"Done! Saved {saved} images to {extract_dir}/")
    return extract_dir


def create_yolo_dataset_structure(base_dir: str):
    """
    创建 YOLO 格式数据集目录结构。

    dataset/
      ├── images/
      │   ├── train/
      │   ├── val/
      │   └── test/
      └── labels/
          ├── train/
          ├── val/
          └── test/
    """
    base = Path(base_dir)
    for split in ["train", "val", "test"]:
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Created YOLO dataset structure at {base}/")


def main():
    args = parse_args()

    print("=" * 60)
    print("  Data Preparation")
    print("=" * 60)

    # 1. 抽帧
    extract_frames(
        args.video, args.output,
        args.frame_interval, args.max_images,
    )

    # 2. 创建数据集结构
    create_yolo_dataset_structure("data/dataset/")


if __name__ == "__main__":
    main()
