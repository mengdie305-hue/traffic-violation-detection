"""
可视化绘制工具。

在帧图像上绘制检测框、追踪 ID、违章标记等。
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np

from src.tracking.deepsort_tracker import Track


# 类别颜色映射 (BGR)
CLASS_COLORS = {
    "car": (0, 255, 0),        # 绿色
    "motorcycle": (0, 255, 255),
    "bus": (255, 0, 0),        # 蓝色
    "truck": (255, 0, 255),
    "person": (255, 255, 0),
    "traffic_light": (0, 255, 255),
}

VIOLATION_COLOR = (0, 0, 255)   # 红色 — 违章车辆


class Visualizer:
    """帧可视化器。"""

    def __init__(
        self,
        show_tracks: bool = True,
        show_labels: bool = True,
        show_trails: bool = True,
        trail_length: int = 30,
    ):
        self.show_tracks = show_tracks
        self.show_labels = show_labels
        self.show_trails = show_trails
        self.trail_length = trail_length

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        violation_ids: Optional[set] = None,
    ) -> np.ndarray:
        """
        绘制所有追踪目标。

        Args:
            frame: BGR 图像
            tracks: 追踪列表
            violation_ids: 违章车辆的 track_id 集合（高亮显示）

        Returns:
            带标注的图像
        """
        if violation_ids is None:
            violation_ids = set()

        for track in tracks:
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            is_violator = track.track_id in violation_ids

            color = (
                VIOLATION_COLOR if is_violator
                else CLASS_COLORS.get(track.class_name, (128, 128, 128))
            )
            thickness = 3 if is_violator else 2

            # 检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # 标签
            if self.show_labels:
                label = f"ID:{track.track_id} {track.class_name}"
                if is_violator:
                    label = f"⚠ {label}"

                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
                )

            # 运动轨迹
            if self.show_trails and len(track.history) > 1:
                trail = track.history[-self.trail_length:]
                for i in range(1, len(trail)):
                    pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    cv2.line(frame, pt1, pt2, color, 1)

        return frame

    def draw_traffic_light_status(
        self, frame: np.ndarray, status: str, roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """在信号灯区域旁绘制状态文字。"""
        x1, y1, x2, y2 = roi

        color_map = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
            "unknown": (128, 128, 128),
        }
        color = color_map.get(status, (128, 128, 128))

        cv2.putText(
            frame, f"LIGHT: {status.upper()}", (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

        return frame

    def draw_violation_alert(
        self, frame: np.ndarray, violation_type: str, plate: Optional[str] = None
    ) -> np.ndarray:
        """在帧顶部绘制违章警示横幅。"""
        h, w = frame.shape[:2]

        # 半透明横幅
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        # 文字
        alert_text = f"⚠ VIOLATION: {violation_type.upper()}"
        if plate:
            alert_text += f" | Plate: {plate}"

        cv2.putText(
            frame, alert_text, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        return frame
