"""
违章事件日志工具。

记录每次检测到的违章事件，支持 CSV 和 JSON 格式输出。
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


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
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(
        self,
        violation_type: str,
        track_id: int,
        timestamp: float,
        frame_idx: int,
        bbox: tuple,
        plate_number: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
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
        }

        if extra:
            record.update(extra)

        self.violations.append(record)

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

        fieldnames = self.violations[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
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
