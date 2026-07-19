"""检测模块单元测试。"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.traffic_light import TrafficLightDetector, TrafficLightState


class TestTrafficLightDetector:
    """交通灯颜色识别测试。"""

    def setup_method(self):
        self.detector = TrafficLightDetector()

    def test_empty_image(self):
        """空图像返回 UNKNOWN。"""
        result = self.detector.recognize(np.array([]))
        assert result == TrafficLightState.UNKNOWN

    def test_dark_image(self):
        """暗光图像返回 UNKNOWN。"""
        dark = np.zeros((50, 50, 3), dtype=np.uint8)
        result = self.detector.recognize(dark)
        assert result == TrafficLightState.UNKNOWN

    def test_red_color(self):
        """识别红色信号灯。"""
        # 创建明亮红色块 (BGR，白色底+红色通道最大确保亮度)
        red_bgr = np.ones((50, 50, 3), dtype=np.uint8) * 100
        red_bgr[:, :, 2] = 255  # Red channel max → bright red
        result = self.detector.recognize(red_bgr)
        assert result == TrafficLightState.RED

    def test_green_color(self):
        """识别绿色信号灯。"""
        green_bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        green_bgr[:, :] = (0, 255, 0)  # BGR green
        result = self.detector.recognize(green_bgr)
        assert result == TrafficLightState.GREEN

    def test_yellow_color(self):
        """识别黄色信号灯。"""
        yellow_bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        yellow_bgr[:, :] = (0, 255, 255)  # BGR yellow
        result = self.detector.recognize(yellow_bgr)
        assert result == TrafficLightState.YELLOW


class TestRedLightDetector:
    """闯红灯检测测试。"""

    def test_vehicle_behind_line(self):
        """车辆在停止线后 → 不触发违章。"""
        from src.violation.red_light import RedLightDetector
        detector = RedLightDetector(
            stop_line=(100, 300, 500, 300),
            red_light_duration=0.5,
        )

        # 车辆在停止线上方（未越线）
        bbox = (200, 100, 300, 280)  # 底边在停止线之上
        assert not detector._is_behind_stop_line(bbox)

    def test_vehicle_crossed_line(self):
        """车辆越过停止线 → 触发位置判定。"""
        from src.violation.red_light import RedLightDetector
        detector = RedLightDetector(
            stop_line=(100, 300, 500, 300),
            red_light_duration=0.5,
        )

        # 车辆底边在停止线之下（越线）
        bbox = (200, 200, 300, 350)
        assert detector._is_behind_stop_line(bbox)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
