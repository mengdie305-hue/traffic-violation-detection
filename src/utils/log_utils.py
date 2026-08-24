"""
违章事件日志工具。

记录每次检测到的违章事件，支持 CSV 和 JSON 格式输出。
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


class ViolationLogger:
    """违章事件日志记录器。"""

    def __init__(
        self,
        output_dir: str = "data/logs",
        format: str = "csv",
        save_frames: bool = False,
    ):
        """
        Args:
            output_dir: 日志输出目录
            format: 输出格式 ("csv" / "json")
            save_frames: 是否保存违章帧截图
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self.save_frames = save_frames

        self.violations: List[Dict[str, Any]] = []
        self._event_keys = set()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(
        self,
        violation_type: str,
        track_id: int,
        timestamp: float,
        frame_idx: int,
        bbox: tuple,
        plate_number: Optional[str] = None,
        traffic_light_state: Optional[str] = None,
        detection_confidence: Optional[float] = None,
        vehicle_class_id: Optional[int] = None,
        vehicle_class_name: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        记录一条违章事件。

        Args:
            violation_type: 违章类型 ("red_light" / "speeding" / "wrong_way")
            track_id: 车辆追踪 ID
            timestamp: 视频时间戳（秒）
            frame_idx: 帧序号
            bbox: 车辆检测框
            plate_number: 车牌号码
            extra: 额外字段（如速度、方向角度等）
        """
        event_key = (self.session_id, int(track_id), violation_type)
        if event_key in self._event_keys:
            return None

        record = {
            "id": len(self.violations) + 1,
            "session_id": self.session_id,
            "violation_type": violation_type,
            "track_id": track_id,
            "timestamp": round(timestamp, 2),
            "formatted_time": self._format_timestamp(timestamp),
            "frame_idx": frame_idx,
            "bbox_x1": bbox[0],
            "bbox_y1": bbox[1],
            "bbox_x2": bbox[2],
            "bbox_y2": bbox[3],
            "plate_number": plate_number or "UNKNOWN",
            "traffic_light_state": traffic_light_state or "UNKNOWN",
            "detection_confidence": (
                round(float(detection_confidence), 4)
                if detection_confidence is not None
                else None
            ),
            "vehicle_class_id": vehicle_class_id,
            "vehicle_class_name": vehicle_class_name or "UNKNOWN",
            "evidence_path": "",
        }

        if extra:
            record.update(extra)

        if self.save_frames and frame is not None and frame.size > 0:
            evidence_dir = self.output_dir / "evidence" / self.session_id
            evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_dir / (
                f"{violation_type}_track-{track_id}_frame-{frame_idx}.jpg"
            )
            evidence_frame = frame.copy()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(
                evidence_frame, (x1, y1), (x2, y2), (0, 0, 255), 3
            )
            if cv2.imwrite(str(evidence_path), evidence_frame):
                record["evidence_path"] = str(evidence_path)

        self._event_keys.add(event_key)
        self.violations.append(record)
        return record

    def _format_timestamp(self, seconds: float) -> str:
        """秒数转 时:分:秒 格式。"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def save(self) -> str:
        """将日志写入文件。"""
        if not self.violations:
            print("[ViolationLogger] No violations to save.")
            return ""

        if self.format == "csv":
            return self._save_csv()
        elif self.format == "json":
            return self._save_json()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

    def _save_csv(self) -> str:
        """保存为 CSV。"""
        filepath = self.output_dir / f"violations_{self.session_id}.csv"

        if not self.violations:
            return str(filepath)

        # 不同违章类型可能携带不同 extra 字段，CSV 列取所有字段并集。
        fieldnames = list(dict.fromkeys(
            key for record in self.violations for key in record.keys()
        ))
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(self.violations)

        print(f"[ViolationLogger] Saved {len(self.violations)} violations → {filepath}")
        return str(filepath)

    def _save_json(self) -> str:
        """保存为 JSON Lines。"""
        filepath = self.output_dir / f"violations_{self.session_id}.jsonl"

        with open(filepath, "w", encoding="utf-8") as f:
            for record in self.violations:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[ViolationLogger] Saved {len(self.violations)} violations → {filepath}")
        return str(filepath)

    def get_summary(self) -> Dict[str, int]:
        """获取违章统计摘要。"""
        summary = {}
        for v in self.violations:
            vtype = v["violation_type"]
            summary[vtype] = summary.get(vtype, 0) + 1
        return summary
