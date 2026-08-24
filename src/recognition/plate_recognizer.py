"""
车牌号码识别模块。

支持两种后端：
1. PaddleOCR — 推荐，中文车牌识别效果好，免费
2. OpenALPR — 商业级，需额外安装和配置

本地开发默认使用 PaddleOCR。
"""

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np


class PlateRecognizer:
    """
    车牌识别器。

    流程：车牌区域裁剪 → 图像预处理 → OCR 识别 → 文本后处理
    """

    def __init__(
        self,
        backend: str = "paddleocr",
        lang: str = "en",
        fallback_to_mock: bool = False,
        sample_interval: int = 5,
        vote_window: int = 5,
        plate_region_start: float = 0.5,
    ):
        """
        Args:
            backend: 识别后端 ("paddleocr" / "openalpr" / "mock")
            lang: OCR 语言 ("en" / "ch" / "ch_en")
        """
        self.backend = backend
        self.lang = lang
        self.fallback_to_mock = fallback_to_mock
        self.sample_interval = max(1, int(sample_interval))
        self.vote_window = max(1, int(vote_window))
        self.plate_region_start = min(0.9, max(0.0, plate_region_start))
        self._ocr = None
        self._paddleocr_major = 0
        self.initialization_error: Optional[str] = None
        self._observations: Dict[int, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.vote_window)
        )
        self._observation_counts: Dict[int, int] = defaultdict(int)

        if backend == "paddleocr":
            self._init_paddleocr()
        elif backend == "openalpr":
            self._init_openalpr()
        elif backend not in {"mock", "disabled"}:
            raise ValueError(f"Unsupported plate recognition backend: {backend}")

    @property
    def available(self) -> bool:
        return self.backend == "mock" or self._ocr is not None

    def _handle_init_failure(self, backend: str, exc: Exception) -> None:
        self.initialization_error = f"{type(exc).__name__}: {exc}"
        if self.fallback_to_mock:
            print(
                f"[PlateRecognizer] {backend} unavailable, falling back to mock: "
                f"{self.initialization_error}"
            )
            self.backend = "mock"
        else:
            print(
                f"[PlateRecognizer] {backend} unavailable; plate results will be "
                f"UNKNOWN: {self.initialization_error}"
            )
            self.backend = "disabled"

    def _init_paddleocr(self):
        """初始化 PaddleOCR。"""
        try:
            import paddleocr
            from paddleocr import PaddleOCR
            self._paddleocr_major = int(paddleocr.__version__.split(".")[0])
            if self._paddleocr_major >= 3:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            else:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                )
            print("[PlateRecognizer] PaddleOCR initialized (CPU mode)")
        except Exception as exc:
            self._handle_init_failure("PaddleOCR", exc)

    def _init_openalpr(self):
        """初始化 OpenALPR。"""
        try:
            from openalpr import Alpr

            self._ocr = Alpr("us", "", "")
            if not self._ocr.is_loaded():
                raise RuntimeError("OpenALPR failed to load its runtime data")
            print("[PlateRecognizer] OpenALPR initialized")
        except Exception as exc:
            self._handle_init_failure("OpenALPR", exc)

    def _extract_plate_region(
        self, frame: np.ndarray, vehicle_bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """裁剪车辆框下部的候选车牌区域。"""
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        height, width = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        region_y1 = y1 + int((y2 - y1) * self.plate_region_start)
        crop = frame[region_y1:y2, x1:x2]
        return crop if crop.size else None

    def _preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        车牌图像预处理：灰度化 → 去噪 → 二值化 → 锐化。
        """
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[1] < 160:
            scale = 160 / max(1, gray.shape[1])
            gray = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        return enhanced

    @staticmethod
    def _normalize_plate(text: str) -> Optional[str]:
        """清理 OCR 文本并做通用长度校验。"""
        cleaned = "".join(character for character in text.upper() if character.isalnum())
        return cleaned if 4 <= len(cleaned) <= 12 else None

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
        plate_crop = self._extract_plate_region(frame, bbox)
        if plate_crop is None:
            return None

        if self.backend == "mock":
            return self._mock_recognize(plate_crop)

        processed = self._preprocess_plate(plate_crop)

        if self.backend == "paddleocr":
            return self._paddleocr_recognize(processed)

        if self.backend == "openalpr":
            result = self._ocr.recognize_ndarray(plate_crop)
            candidates = result.get("results", []) if result else []
            return (
                self._normalize_plate(candidates[0].get("plate", ""))
                if candidates
                else None
            )

        return None

    def observe(
        self,
        track_id: int,
        frame: np.ndarray,
        vehicle_bbox: Tuple[int, int, int, int],
    ) -> Optional[str]:
        """周期性识别车牌，并返回该轨迹当前的多帧投票结果。"""
        self._observation_counts[track_id] += 1
        should_run = (
            self._observation_counts[track_id] == 1
            or self._observation_counts[track_id] % self.sample_interval == 0
        )
        if should_run:
            plate = self.recognize(frame, vehicle_bbox)
            if plate:
                self._observations[track_id].append(plate)
        return self.get_consensus(track_id)

    def get_consensus(self, track_id: int) -> Optional[str]:
        """返回轨迹 OCR 历史中票数最高的车牌；并列时取最近结果。"""
        observations = self._observations.get(track_id)
        if not observations:
            return None
        counts = Counter(observations)
        max_votes = max(counts.values())
        for plate in reversed(observations):
            if counts[plate] == max_votes:
                return plate
        return None

    def _paddleocr_recognize(self, image: np.ndarray) -> Optional[str]:
        """使用 PaddleOCR 识别。"""
        if self._paddleocr_major >= 3:
            texts = []
            for result in self._ocr.predict(image):
                payload = getattr(result, "json", result)
                if callable(payload):
                    payload = payload()
                if isinstance(payload, dict) and "res" in payload:
                    payload = payload["res"]
                recognized = payload.get("rec_texts", []) if isinstance(payload, dict) else []
                scores = payload.get("rec_scores", []) if isinstance(payload, dict) else []
                for text, confidence in zip(recognized, scores):
                    if float(confidence) > 0.6:
                        cleaned = self._normalize_plate(text)
                        if cleaned:
                            texts.append(cleaned)
            return self._normalize_plate("".join(texts)) if texts else None

        result = self._ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return None

        texts = []
        for line in result[0]:
            text = line[1][0]
            confidence = line[1][1]
            if confidence > 0.6:
                # 过滤非车牌字符
                cleaned = self._normalize_plate(text)
                if cleaned:
                    texts.append(cleaned)

        return self._normalize_plate("".join(texts)) if texts else None

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
