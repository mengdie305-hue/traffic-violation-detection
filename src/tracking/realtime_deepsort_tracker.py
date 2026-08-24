"""基于 deep-sort-realtime 的完整 DeepSORT 适配器。"""

from typing import Dict, List, Optional

import numpy as np

from .deepsort_tracker import Track


class RealtimeDeepSORTTracker:
    """将 deep-sort-realtime 输出转换为项目统一的 ``Track`` 对象。"""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.2,
        embedder: str = "mobilenet",
        embedder_gpu: bool = False,
    ):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            raise RuntimeError(
                "realtime_deepsort requires 'deep-sort-realtime'. "
                "Install project requirements before using the real pipeline."
            ) from exc

        self.max_age = max_age
        self.min_hits = min_hits
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=min_hits,
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
            bgr=True,
        )
        self.tracks: Dict[int, Track] = {}

    def update(
        self,
        detections: np.ndarray,
        class_ids: np.ndarray,
        class_names: List[str],
        scores: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        if frame is None:
            raise ValueError("RealtimeDeepSORTTracker requires the current video frame")
        if scores is None:
            scores = np.ones(len(detections), dtype=np.float32)

        raw_detections = []
        supplementary = []
        for bbox, score, class_id, class_name in zip(
            detections, scores, class_ids, class_names
        ):
            x1, y1, x2, y2 = [float(value) for value in bbox]
            raw_detections.append(
                ([x1, y1, x2 - x1, y2 - y1], float(score), class_name)
            )
            supplementary.append(int(class_id))

        external_tracks = self._tracker.update_tracks(
            raw_detections, frame=frame, others=supplementary
        )
        active_ids = set()

        for external in external_tracks:
            if not external.is_confirmed() or external.time_since_update > self.max_age:
                continue

            track_id = int(external.track_id)
            active_ids.add(track_id)
            bbox = tuple(float(value) for value in external.to_ltrb())
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            was_updated = external.time_since_update == 0
            class_name = external.get_det_class() or "unknown"
            class_id = external.get_det_supplementary()
            class_id = int(class_id) if class_id is not None else -1
            confidence = external.get_det_conf()

            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=str(class_name),
                    bbox=bbox,
                    center=center,
                    history=[center],
                    total_frames=self.min_hits,
                    was_updated=was_updated,
                    confidence=(float(confidence) if confidence is not None else None),
                )
                continue

            track = self.tracks[track_id]
            previous_center = track.center
            track.bbox = bbox
            track.center = center
            track.frames_since_update = int(external.time_since_update)
            track.was_updated = was_updated
            if class_id != -1:
                track.class_id = class_id
            if class_name != "unknown":
                track.class_name = str(class_name)
            if confidence is not None:
                track.confidence = float(confidence)
            if was_updated:
                track.velocity = (
                    center[0] - previous_center[0],
                    center[1] - previous_center[1],
                )
                track.history.append(center)
                track.history = track.history[-100:]
                track.total_frames += 1

        expired_ids = set(self.tracks) - active_ids
        for track_id in expired_ids:
            del self.tracks[track_id]

        return list(self.tracks.values())
