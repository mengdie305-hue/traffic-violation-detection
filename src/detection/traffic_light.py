"""
交通灯颜色识别模块。

基于 HSV 颜色空间分割，判断信号灯状态（红/黄/绿）。
"""

from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


# HSV 颜色范围（可根据实际场景调整）
# 红色有两个范围（HSV 色相环绕）
COLOR_RANGES = {
    TrafficLightState.RED: [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),     # 红色下限
        (np.array([160, 100, 100]), np.array([180, 255, 255])),  # 红色上限
    ],
    TrafficLightState.YELLOW: [
        (np.array([15, 100, 100]), np.array([35, 255, 255])),
    ],
    TrafficLightState.GREEN: [
        (np.array([36, 100, 100]), np.array([85, 255, 255])),
    ],
}


class TrafficLightDetector:
    """交通灯颜色识别器。"""

    def __init__(
        self,
        min_area_ratio: float = 0.05,
        brightness_threshold: int = 100,
    ):
        """
        Args:
            min_area_ratio: 最小颜色占比阈值（ROI 区域内）
            brightness_threshold: 最小亮度阈值，过滤暗光环境误判
        """
        self.min_area_ratio = min_area_ratio
        self.brightness_threshold = brightness_threshold

    def recognize(
        self, roi_image: np.ndarray
    ) -> TrafficLightState:
        """
        识别 ROI 区域内的交通灯颜色。

        Args:
            roi_image: 信号灯区域 BGR 图像

        Returns:
            信号灯状态枚举
        """
        if roi_image.size == 0:
            return TrafficLightState.UNKNOWN

        # 转 HSV
        hsv = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)
        roi_area = roi_image.shape[0] * roi_image.shape[1]

        # 亮度检查
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < self.brightness_threshold:
            return TrafficLightState.UNKNOWN

        # 对每种颜色计算掩码面积
        max_area = 0
        best_state = TrafficLightState.UNKNOWN

        for state, ranges in COLOR_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, lower, upper)

            # 形态学操作去噪
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)

            area = cv2.countNonZero(mask)
            if area > max_area:
                max_area = area
                best_state = state

        # 验证颜色占比达到阈值
        if max_area / roi_area < self.min_area_ratio:
            return TrafficLightState.UNKNOWN

        return best_state
