"""
逆行 / 违规转弯检测模块。

通过分析车辆运动方向与车道规定方向的偏差来判断。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


class LaneDirection(Enum):
    STRAIGHT = "straight"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class WrongWayViolation:
    """逆行违章记录。"""
    track_id: int
    violation_type: str        # "wrong_way" / "illegal_turn"
    direction_angle: float      # 实际行驶角度
    expected_direction: str     # 车道规定方向
    timestamp: float
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    plate_number: Optional[str] = None


class WrongWayDetector:
    """逆行检测器。"""

    def __init__(
        self,
        lane_directions: dict,          # lane_id → LaneDirection
        angle_threshold: float = 30.0,   # 方向偏差阈值（度）
        min_history: int = 10,           # 最小轨迹点数才判断方向
    ):
        """
        Args:
            lane_directions: {lane_id: LaneDirection} 车道方向映射
            angle_threshold: 超过此偏差角判定为逆行
            min_history: 最少轨迹点才进行方向判断
        """
        self.lane_directions = lane_directions
        self.angle_threshold = angle_threshold
        self.min_history = min_history

        self.violations: List[WrongWayViolation] = []

    def _compute_movement_angle(
        self, history: List[Tuple[float, float]]
    ) -> float:
        """
        从轨迹历史计算整体运动方向角度。

        Returns:
            角度（度），0° = 向右，90° = 向下（图像坐标系）
        """
        if len(history) < 2:
            return 0.0

        # 使用首尾点计算总体方向
        start = history[0]
        end = history[-1]

        dx = end[0] - start[0]
        dy = end[1] - start[1]

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _get_expected_angle(self, lane_id: int) -> float:
        """
        获取车道规定方向的期望角度。

        图像坐标系中：
        - 直行（远离摄像头/向下）：~90°
        - 直行（靠近摄像头/向上）：~-90°
        - 左转：~0° 到 90°
        - 右转：~90° 到 180°
        """
        direction = self.lane_directions.get(lane_id, LaneDirection.STRAIGHT)

        # 默认假设：摄像头朝向路口，车辆远离摄像头行驶 = 向下 = 90°
        expected_angles = {
            LaneDirection.STRAIGHT: 90.0,    # 向下（远离）
            LaneDirection.LEFT: 0.0,          # 向左
            LaneDirection.RIGHT: 180.0,       # 向右
        }
        return expected_angles[direction]

    def _get_lane_for_position(
        self, center: Tuple[float, float]
    ) -> Optional[int]:
        """
        根据车辆位置判断所在车道。

        简化实现：基于 x 坐标与各车道边界的比较。
        实际应用需要完整的车道线检测和地图。
        """
        # TODO: 接入车道线检测结果
        # 当前返回 None 表示未实现车道判定
        return None

    def check_violation(
        self,
        track_id: int,
        history: List[Tuple[float, float]],
        bbox: Tuple[float, float, float, float],
        timestamp: float,
        frame_idx: int,
    ) -> Optional[WrongWayViolation]:
        """
        检查是否逆行。

        Args:
            history: 车辆轨迹点列表 [(cx, cy), ...]

        Returns:
            逆行时返回违章记录，否则返回 None
        """
        if len(history) < self.min_history:
            return None

        # 计算实际运动方向
        actual_angle = self._compute_movement_angle(history)

        # 判断车辆所在车道
        center = history[-1]
        lane_id = self._get_lane_for_position(center)

        if lane_id is None:
            # 无法判定车道时，使用通用规则：
            # 正常情况下车辆应向下行驶（远离摄像头）
            # 向上行驶 → 疑似逆行
            expected_angle = 90.0  # 默认向下
        else:
            expected_angle = self._get_expected_angle(lane_id)

        # 计算角度偏差
        angle_diff = abs(actual_angle - expected_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)  # 归一化

        if angle_diff > self.angle_threshold:
            violation = WrongWayViolation(
                track_id=track_id,
                violation_type="wrong_way",
                direction_angle=round(actual_angle, 1),
                expected_direction=f"{expected_angle}°",
                timestamp=timestamp,
                frame_idx=frame_idx,
                bbox=tuple(int(v) for v in bbox),
            )
            self.violations.append(violation)
            return violation

        return None
