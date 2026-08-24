"""
视频处理工具。

提供视频读写、ROI 区域绘制、帧图像保存等功能。
"""

from pathlib import Path
from typing import Optional, Tuple, Generator

import cv2
import numpy as np


class VideoProcessor:
    """视频读写处理器。"""

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        resize: Optional[Tuple[int, int]] = None,
        frame_skip: int = 1,
    ):
        """
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径（可选，用于保存带标注的视频）
            resize: 输出帧尺寸 (width, height)，None 保持原尺寸
            frame_skip: 跳帧数，1=逐帧处理，N=每 N 帧处理一次
        """
        self.input_path = Path(input_path)
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.frame_skip = frame_skip

        if not self.input_path.exists():
            raise FileNotFoundError(f"Video not found: {input_path}")

        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video: {input_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            raise ValueError(f"Unable to read a valid FPS from: {input_path}")
        self.processing_fps = self.fps / self.frame_skip
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.resize = resize
        if resize:
            self.width, self.height = resize

        self._writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                output_path, fourcc, self.processing_fps,
                (self.width, self.height)
            )
            if not self._writer.isOpened():
                raise ValueError(f"Unable to create output video: {output_path}")

        print(
            f"[VideoProcessor] {self.input_path.name}: "
            f"{self.width}x{self.height}, {self.fps:.1f} fps, "
            f"{self.frame_count} frames, skip={frame_skip}"
        )

    def frames(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        逐帧迭代器。

        Yields:
            (frame_idx, timestamp_seconds, frame_bgr)
        """
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if idx % self.frame_skip == 0:
                if self.resize:
                    frame = cv2.resize(frame, self.resize)
                timestamp = idx / self.fps
                yield idx, timestamp, frame
            idx += 1

        self.cap.release()
        if self._writer:
            self._writer.release()

    def write_frame(self, frame: np.ndarray):
        """写入一帧到输出视频。"""
        if self._writer:
            if self.resize and (
                frame.shape[1] != self.width or frame.shape[0] != self.height
            ):
                frame = cv2.resize(frame, self.resize)
            self._writer.write(frame)

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """获取指定帧。"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret and self.resize:
            frame = cv2.resize(frame, self.resize)
        return frame if ret else None

    def get_fps(self) -> float:
        """返回源视频 FPS，用于基于原始帧号计算时间戳。"""
        return self.fps

    def get_processing_fps(self) -> float:
        """返回处理帧流 FPS，用于仅写出处理帧的输出视频。"""
        return self.processing_fps

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cap.release()
        if self._writer:
            self._writer.release()


def draw_roi(frame: np.ndarray, roi: dict) -> np.ndarray:
    """
    在帧上绘制 ROI 区域（调试用）。

    Args:
        frame: BGR 图像
        roi: ROI 配置字典 {"stop_line": [...], "traffic_light_bbox": [...], ...}

    Returns:
        带标注的图像
    """
    overlay = frame.copy()

    # 停止线
    if "stop_line" in roi:
        x1, y1, x2, y2 = roi["stop_line"]
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            overlay, "STOP LINE", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )

    # 信号灯区域
    if "traffic_light_bbox" in roi:
        x1, y1, x2, y2 = roi["traffic_light_bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            overlay, "TRAFFIC LIGHT", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return overlay
