"""
闯红灯检测模块。

逻辑：
1. 检测交通灯状态为 RED
2. 车辆越过停止线进入路口
3. 记录违章时间戳和车辆信息
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional

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


class CrossingState(Enum):
    """车辆相对停止线的位置状态。"""

    BEFORE_LINE = "before_line"
    CROSSING_LINE = "crossing_line"
    AFTER_LINE = "after_line"


@dataclass
class TrackCrossingState:
    """单辆车的跨线状态。"""

    phase: CrossingState
    previous_side: float
    seen_before_line: bool
    violation_recorded: bool = False


class RedLightDetector:
    """闯红灯检测器。"""

    def __init__(
        self,
        stop_line: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        overlap_threshold: float = 0.3,
        red_light_duration: float = 1.0,
        crossing_direction: str = "downward",
    ):
        """
        Args:
            stop_line: 停止线像素坐标
            overlap_threshold: 车辆与停止线区域的碰撞阈值
            red_light_duration: 红灯期间进入路口的最小持续时间（秒）才判定违章
            crossing_direction: 有效越线方向，``downward``/``rightward`` 表示
                有符号距离从负到正，``upward``/``leftward`` 表示从正到负
        """
        self.stop_line = stop_line
        self.overlap_threshold = overlap_threshold
        self.red_light_duration = red_light_duration
        direction = crossing_direction.lower()
        if direction in {"downward", "rightward", "positive"}:
            self._crossing_sign = 1.0
        elif direction in {"upward", "leftward", "negative"}:
            self._crossing_sign = -1.0
        else:
            raise ValueError(
                "crossing_direction must be downward/rightward/positive "
                "or upward/leftward/negative"
            )
        self.crossing_direction = direction

        self._track_states: Dict[int, TrackCrossingState] = {}
        self._last_light_state = TrafficLightState.UNKNOWN
        self._red_since: Optional[float] = None
        self.violations: List[RedLightViolation] = []

    def update_light_state(
        self, light_state: TrafficLightState, timestamp: float
    ) -> None:
        """更新信号灯状态，并记录连续红灯的开始时间。"""
        if light_state == TrafficLightState.RED:
            if self._last_light_state != TrafficLightState.RED:
                self._red_since = timestamp
        else:
            self._red_since = None
        self._last_light_state = light_state

    def _signed_side(self, bbox: Tuple[float, float, float, float]) -> float:
        """返回车辆底边中点相对停止线的有符号距离近似值。"""
        x1, _, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = self.stop_line
        vehicle_x = (x1 + x2) / 2

        if sx1 == sx2:
            # 垂直停止线：默认从左向右穿越为正方向。
            return vehicle_x - sx1

        ratio = (vehicle_x - sx1) / (sx2 - sx1)
        line_y = sy1 + ratio * (sy2 - sy1)
        return y2 - line_y

    def _within_stop_line_span(
        self, bbox: Tuple[float, float, float, float]
    ) -> bool:
        """车辆中心投影是否位于停止线覆盖范围内。"""
        x1, y1, x2, y2 = bbox
        sx1, sy1, sx2, sy2 = self.stop_line
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if abs(sx2 - sx1) >= abs(sy2 - sy1):
            return min(sx1, sx2) <= center_x <= max(sx1, sx2)
        return min(sy1, sy2) <= center_y <= max(sy1, sy2)

    def _is_behind_stop_line(
        self, bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        判断车辆是否在停止线内（路口侧）。

        简化逻辑：检测框底边中点 y 坐标是否超过停止线 y 坐标。
        实际应用需根据摄像头视角和道路几何调整。
        """
        return self._signed_side(bbox) > 0

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

    def check_violation(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        light_state: TrafficLightState,
        timestamp: float,
        frame_idx: int,
        fps: float = 30.0,
    ) -> Optional[RedLightViolation]:
        """检测车辆是否在连续红灯期间从停止线前方跨到后方。

        ``fps`` 为兼容旧调用保留；判定使用视频时间戳，不受跳帧影响。
        """
        self.update_light_state(light_state, timestamp)
        # 将目标方向统一映射为“负侧 → 正侧”，使状态机同时支持
        # 向上/向左和向下/向右的车辆越线。
        side = self._crossing_sign * self._signed_side(bbox)
        state = self._track_states.get(track_id)

        if state is None:
            self._track_states[track_id] = TrackCrossingState(
                phase=(
                    CrossingState.BEFORE_LINE
                    if side <= 0
                    else CrossingState.AFTER_LINE
                ),
                previous_side=side,
                seen_before_line=side <= 0,
            )
            return None

        crossed_forward = (
            state.previous_side <= 0
            and side > 0
            and self._within_stop_line_span(bbox)
        )

        if side <= 0:
            state.phase = CrossingState.BEFORE_LINE
            state.seen_before_line = True
        elif crossed_forward:
            state.phase = CrossingState.CROSSING_LINE
        else:
            state.phase = CrossingState.AFTER_LINE

        red_duration = (
            timestamp - self._red_since
            if self._red_since is not None
            else 0.0
        )
        is_violation = (
            crossed_forward
            and state.seen_before_line
            and light_state == TrafficLightState.RED
            and red_duration >= self.red_light_duration
            and not state.violation_recorded
        )
        state.previous_side = side

        if not is_violation:
            return None

        state.violation_recorded = True
        violation = RedLightViolation(
            track_id=track_id,
            timestamp=timestamp,
            frame_idx=frame_idx,
            bbox=tuple(int(v) for v in bbox),
        )
        self.violations.append(violation)
        return violation
