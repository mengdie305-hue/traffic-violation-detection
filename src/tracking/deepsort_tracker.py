"""
DeepSORT 多目标追踪模块。

基于 DeepSORT 算法实现车辆跨帧追踪，为每辆车分配唯一的 track_id。
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """单条追踪记录。"""

    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    center: Tuple[float, float]               # (cx, cy)
    velocity: Tuple[float, float] = (0.0, 0.0)
    history: List[Tuple[float, float]] = field(default_factory=list)
    frames_since_update: int = 0
    total_frames: int = 0
    was_updated: bool = False
    confidence: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.frames_since_update < 30


class DeepSORTTracker:
    """
    DeepSORT 追踪器（简化实现）。

    完整版本需要：
    1. 卡尔曼滤波 — 状态预测
    2. 匈牙利算法 — 检测框与追踪匹配
    3. ReID 特征提取 — 外观特征匹配

    本简化版使用 IOU + 中心点距离作为关联度量，
    适合本地开发和快速验证。服务器训练时替换为完整 DeepSORT。
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 50.0,
    ):
        """
        Args:
            max_age: 追踪丢失前的最大帧数
            min_hits: 确认追踪所需的最小命中数
            iou_threshold: 检测框匹配的 IOU 阈值
            max_distance: 中心点匹配的最大欧氏距离（像素）
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

        self.tracks: Dict[int, Track] = {}
        self._next_id = 0

    def update(
        self,
        detections: np.ndarray,    # [N, 4] (x1,y1,x2,y2)
        class_ids: np.ndarray,     # [N]
        class_names: List[str],    # [N]
        scores: Optional[np.ndarray] = None,  # [N]
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        用新检测框更新追踪状态。

        Returns:
            当前所有活跃的追踪列表
        """
        # 预测：更新已有追踪的位置（简化：直接用上一帧位置预测）
        for track in self.tracks.values():
            track.frames_since_update += 1
            track.was_updated = False

        if len(detections) == 0:
            return self._get_active_tracks()

        # 已存在的追踪中心点
        tracked_ids = [tid for tid, t in self.tracks.items() if t.is_active]
        tracked_centers = np.array([
            self.tracks[tid].center for tid in tracked_ids
        ])

        # 检测框中心点
        det_centers = np.column_stack([
            (detections[:, 0] + detections[:, 2]) / 2,
            (detections[:, 1] + detections[:, 3]) / 2,
        ])

        # 简单的全局贪心匹配。显式保存 (track_index, detection_index)，
        # 避免分别存入 set 后丢失二者的配对关系。
        matches = []
        matched_track_indices = set()
        matched_det_indices = set()

        if len(tracked_centers) > 0 and len(det_centers) > 0:
            candidates = []
            for ti, tc in enumerate(tracked_centers):
                for di, dc in enumerate(det_centers):
                    track = self.tracks[tracked_ids[ti]]
                    if int(track.class_id) != int(class_ids[di]):
                        continue
                    dist = np.linalg.norm(tc - dc)
                    if dist < self.max_distance:
                        candidates.append((float(dist), ti, di))

            # 优先选择距离最小的组合，每条轨迹和每个检测最多使用一次。
            for _, ti, di in sorted(candidates):
                if ti in matched_track_indices or di in matched_det_indices:
                    continue
                matches.append((ti, di))
                matched_track_indices.add(ti)
                matched_det_indices.add(di)

        # 更新已匹配的追踪
        for ti, det_i in matches:
            tid = tracked_ids[ti]
            track = self.tracks[tid]
            track.bbox = tuple(detections[det_i].tolist())
            new_center = tuple(det_centers[det_i].tolist())
            track.velocity = (
                new_center[0] - track.center[0],
                new_center[1] - track.center[1],
            )
            track.center = new_center
            track.history.append(new_center)
            track.frames_since_update = 0
            track.total_frames += 1
            track.was_updated = True
            track.confidence = (
                float(scores[det_i]) if scores is not None else None
            )
            # 限制历史长度
            if len(track.history) > 100:
                track.history = track.history[-100:]

        # 未匹配的检测 → 创建新追踪
        for di in range(len(detections)):
            if di not in matched_det_indices:
                tid = self._next_id
                self._next_id += 1
                center = tuple(det_centers[di].tolist())
                self.tracks[tid] = Track(
                    track_id=tid,
                    class_id=class_ids[di],
                    class_name=class_names[di],
                    bbox=tuple(detections[di].tolist()),
                    center=center,
                    history=[center],
                    total_frames=1,
                    was_updated=True,
                    confidence=(
                        float(scores[di]) if scores is not None else None
                    ),
                )

        # 清理过期追踪
        expired = [
            tid for tid, t in self.tracks.items()
            if t.frames_since_update > self.max_age
        ]
        for tid in expired:
            del self.tracks[tid]

        return self._get_active_tracks()

    def _get_active_tracks(self) -> List[Track]:
        """获取所有确认的活跃追踪。"""
        return [
            t for t in self.tracks.values()
            if t.frames_since_update <= self.max_age
            and t.total_frames >= self.min_hits
        ]
