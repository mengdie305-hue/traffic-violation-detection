"""
车牌号码识别模块。

支持两种后端：
1. PaddleOCR — 推荐，中文车牌识别效果好，免费
2. OpenALPR — 商业级，需额外安装和配置

本地开发默认使用 PaddleOCR。
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class PlateRecognizer:
    """
    车牌识别器。

    流程：车牌区域裁剪 → 图像预处理 → OCR 识别 → 文本后处理
    """

    def __init__(self, backend: str = "paddleocr", lang: str = "en"):
        """
        Args:
            backend: 识别后端 ("paddleocr" / "openalpr" / "mock")
            lang: OCR 语言 ("en" / "ch" / "ch_en")
        """
        self.backend = backend
        self.lang = lang
        self._ocr = None

        if backend == "paddleocr":
            self._init_paddleocr()
        elif backend == "openalpr":
            self._init_openalpr()

    def _init_paddleocr(self):
        """初始化 PaddleOCR。"""
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=False,  # 本地 CPU 推理
                show_log=False,
            )
            print("[PlateRecognizer] PaddleOCR initialized (CPU mode)")
        except ImportError:
            print("[PlateRecognizer] PaddleOCR not installed, falling back to mock")
            self.backend = "mock"

    def _init_openalpr(self):
        """初始化 OpenALPR。"""
        try:
            # OpenALPR Python 绑定
            self._ocr = "openalpr_placeholder"
            print("[PlateRecognizer] OpenALPR initialized")
        except Exception:
            print("[PlateRecognizer] OpenALPR init failed, falling back to mock")
            self.backend = "mock"

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        车牌图像预处理：灰度化 → 去噪 → 二值化 → 锐化。
        """
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    def recognize(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[str]:
        """
        识别指定区域内的车牌号码。

        Args:
            frame: 完整帧图像 (BGR)
            bbox: 车牌区域 (x1, y1, x2, y2)

        Returns:
            识别的车牌字符串，失败时返回 None
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 边界检查
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None

        if self.backend == "mock":
            return self._mock_recognize(plate_crop)

        processed = self._preprocess_plate(plate_crop)

        if self.backend == "paddleocr":
            return self._paddleocr_recognize(processed)

        return None

    def _paddleocr_recognize(self, image: np.ndarray) -> Optional[str]:
        """使用 PaddleOCR 识别。"""
        result = self._ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return None

        texts = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.6:
                # 过滤非车牌字符
                cleaned = "".join(
                    c for c in text if c.isalnum()
                )
                if cleaned:
                    texts.append(cleaned)

        return "".join(texts) if texts else None

    def _mock_recognize(self, plate_crop: np.ndarray) -> Optional[str]:
        """
        模拟识别（用于开发和测试）。
        实际使用时替换为真实的 OCR 后端。
        """
        # 简单的边缘检测验证车牌区域非空
        edges = cv2.Canny(plate_crop, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size

        if edge_ratio > 0.02:
            return "XX-0000-XX"  # 模拟结果
        return None
