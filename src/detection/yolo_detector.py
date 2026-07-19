"""
YOLO 目标检测器 — 负责检测车辆、交通灯、行人等目标。

使用 Ultralytics YOLOv8 作为底层模型。
"""

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from ultralytics import YOLO

# COCO 数据集中与交通相关的类别
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0
TRAFFIC_LIGHT_CLASS = 9  # COCO 中交通灯类别


class YOLODetector:
    """YOLOv8 检测器封装，支持 GPU/CPU 推理。"""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        """
        初始化 YOLO 检测器。

        Args:
            model_path: 模型权重路径（本地文件或 Ultralytics 模型名）
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            device: 推理设备 ("auto"/"cpu"/"cuda"/"mps")
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 设备选择：优先 MPS (Apple Silicon) > CUDA > CPU
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[YOLODetector] Loading model on device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(
        self, frame: np.ndarray, classes: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对单帧图像执行目标检测。

        Args:
            frame: BGR 图像 (H, W, 3)
            classes: 要检测的类别 ID 列表，None 表示检测全部

        Returns:
            (boxes, scores, class_ids):
                boxes:  [N, 4] (x1, y1, x2, y2)
                scores: [N]
                class_ids: [N]
        """
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False,
        )

        if results[0].boxes is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(np.int32)

        return boxes, scores, class_ids

    def detect_vehicles(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """检测所有机动车。"""
        return self.detect(frame, classes=list(VEHICLE_CLASSES.keys()))

    def detect_traffic_lights(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """检测交通灯。"""
        return self.detect(frame, classes=[TRAFFIC_LIGHT_CLASS])
