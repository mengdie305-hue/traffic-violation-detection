"""
超速检测模块。

基于车辆追踪轨迹计算行驶速度。
需要相机标定参数（像素→真实世界的映射比例）。
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


@dataclass
class SpeedingViolation:
    """超速违章记录。"""
    track_id: int
    speed_kmh: float             # 计算速度
    speed_limit_kmh: float       # 限速
    timestamp: float
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    plate_number: Optional[str] = None


class SpeedingDetector:
    """超速检测器。"""

    def __init__(
        self,
        max_speed_kmh: float = 60.0,
        pixel_to_meter_ratio: float = 0.05,
        fps: float = 30.0,
        smoothing_window: int = 5,
    ):
        """
        Args:
            max_speed_kmh: 最高限速 (km/h)
            pixel_to_meter_ratio: 像素→米的转换比例（需相机标定）
            fps: 视频帧率
            smoothing_window: 速度平滑窗口大小（帧数）
        """
        self.max_speed_kmh = max_speed_kmh
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.fps = fps
        self.smoothing_window = smoothing_window

        self.violations: List[SpeedingViolation] = []
        self._speed_history: dict = {}  # track_id → List[float]

    def calculate_speed(
        self,
        track_id: int,
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
    ) -> float:
        """
        根据相邻帧位移计算瞬时速度。

        Args:
            prev_center: 上一帧车辆中心点 (cx, cy)
            curr_center: 当前帧车辆中心点 (cx, cy)

        Returns:
            瞬时速度 (km/h)
        """
        # 像素位移
        dp = np.sqrt(
            (curr_center[0] - prev_center[0]) ** 2
            + (curr_center[1] - prev_center[1]) ** 2
        )

        # 转为真实世界位移（米）
        dm = dp * self.pixel_to_meter_ratio

        # 速度 = 位移 / 时间 → km/h
        speed_ms = dm * self.fps           # 米/秒
        speed_kmh = speed_ms * 3.6         # 公里/小时

        return speed_kmh

    def update(
        self,
        track_id: int,
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
    ) -> float:
        """
        更新速度估计（带平滑）。

        Returns:
            平滑后的速度 (km/h)
        """
        raw_speed = self.calculate_speed(track_id, prev_center, curr_center)

        if track_id not in self._speed_history:
            self._speed_history[track_id] = []

        history = self._speed_history[track_id]
        history.append(raw_speed)

        # 保持窗口大小
        if len(history) > self.smoothing_window:
            history.pop(0)

        # 中值滤波抑制噪声
        return float(np.median(history))

    def check_violation(
        self,
        track_id: int,
        prev_center: Tuple[float, float],
        curr_center: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        timestamp: float,
        frame_idx: int,
    ) -> Optional[SpeedingViolation]:
        """
        检查是否超速。

        Returns:
            超速时返回违章记录，否则返回 None
        """
        speed = self.update(track_id, prev_center, curr_center)

        if speed > self.max_speed_kmh:
            violation = SpeedingViolation(
                track_id=track_id,
                speed_kmh=round(speed, 1),
                speed_limit_kmh=self.max_speed_kmh,
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=tuple(int(v) for v in bbox),
            )
            self.violations.append(violation)
            return violation

        return None
