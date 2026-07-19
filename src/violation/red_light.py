"""
闯红灯检测模块。

逻辑：
1. 检测交通灯状态为 RED
2. 车辆越过停止线进入路口
3. 记录违章时间戳和车辆信息
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from collections import defaultdict

import numpy as np

from src.detection.traffic_light import TrafficLightState


@dataclass
class RedLightViolation:
    """闯红灯违章记录。"""
    track_id: int
    timestamp: float          # 视频时间（秒）
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    plate_number: Optional[str] = None


class RedLightDetector:
    """闯红灯检测器。"""

    def __init__(
        self,
        stop_line: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        overlap_threshold: float = 0.3,
        red_light_duration: float = 1.0,
    ):
        """
        Args:
            stop_line: 停止线像素坐标
            overlap_threshold: 车辆与停止线区域的碰撞阈值
            red_light_duration: 红灯期间进入路口的最小持续时间（秒）才判定违章
        """
        self.stop_line = stop_line
        self.overlap_threshold = overlap_threshold
        self.red_light_duration = red_light_duration

        # 追踪每辆车的违章状态
        self._violation_candidates: dict = defaultdict(lambda: {
            "start_frame": None,
            "frames_in_violation": 0,
        })
        self.violations: List[RedLightViolation] = []

    def _is_behind_stop_line(
        self, bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        判断车辆是否在停止线内（路口侧）。

        简化逻辑：检测框底边中点 y 坐标是否超过停止线 y 坐标。
        实际应用需根据摄像头视角和道路几何调整。
        """
        x1, y1, x2, y2 = bbox
        vehicle_bottom = y2
        stop_line_y = (self.stop_line[1] + self.stop_line[3]) / 2

        # 车辆底边低于停止线 = 已越过停止线进入路口
        return vehicle_bottom > stop_line_y

    def _bbox_overlap_stop_line(
        self, bbox: Tuple[float, float, float, float]
    ) -> float:
        """计算车辆检测框与停止线的重叠比例。"""
        bx1, by1, bx2, by2 = bbox
        sx1, sy1, sx2, sy2 = self.stop_line

        # 计算交集
        ix1 = max(bx1, sx1)
        iy1 = max(by1, sy1 - 10)  # 垂直方向扩展容差
        ix2 = min(bx2, sx2)
        iy2 = min(by2, sy2 + 10)

        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = (bx2 - bx1) * (by2 - by1)

        return inter_area / bbox_area if bbox_area > 0 else 0.0

    def update(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        light_state: TrafficLightState,
        timestamp: float,
        frame_idx: int,
    ) -> Optional[RedLightViolation]:
        """
        每帧更新：检测指定追踪车辆是否闯红灯。

        Returns:
            如果检测到闯红灯，返回违章记录；否则返回 None
        """
        # 只在红灯时检查
        if light_state != TrafficLightState.RED:
            # 重置绿灯/黄灯期间的候选状态
            if track_id in self._violation_candidates:
                del self._violation_candidates[track_id]
            return None

        # 检查车辆是否越过停止线
        overlap = self._bbox_overlap_stop_line(bbox)
        behind_line = self._is_behind_stop_line(bbox)

        if not behind_line and overlap < self.overlap_threshold:
            # 车辆未越线
            if track_id in self._violation_candidates:
                del self._violation_candidates[track_id]
            return None

        # 车辆在红灯状态下越线
        candidate = self._violation_candidates[track_id]
        if candidate["start_frame"] is None:
            candidate["start_frame"] = frame_idx
        candidate["frames_in_violation"] += 1

        return None

    def check_violation(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        light_state: TrafficLightState,
        timestamp: float,
        frame_idx: int,
        fps: float = 30.0,
    ) -> Optional[RedLightViolation]:
        """
        综合检查：先 update，再判断是否构成完整违章。

        Args:
            fps: 视频帧率，用于将帧数转换为时长
        """
        self.update(track_id, bbox, light_state, timestamp, frame_idx)

        if track_id not in self._violation_candidates:
            return None

        candidate = self._violation_candidates[track_id]
        duration = candidate["frames_in_violation"] / fps

        if duration >= self.red_light_duration:
            violation = RedLightViolation(
                track_id=track_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=tuple(int(v) for v in bbox),
            )
            self.violations.append(violation)
            # 清除已记录的违章，避免重复
            del self._violation_candidates[track_id]
            return violation

        return None
