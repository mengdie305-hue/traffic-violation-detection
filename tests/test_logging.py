"""违章日志与证据保存测试。"""

import csv
import json
from pathlib import Path

import numpy as np

from src.utils.log_utils import ViolationLogger


def test_duplicate_event_is_ignored_and_evidence_is_saved(tmp_path):
    logger = ViolationLogger(
        output_dir=str(tmp_path), format="csv", save_frames=True
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)

    first = logger.log(
        violation_type="red_light",
        track_id=7,
        timestamp=1.25,
        frame_idx=25,
        bbox=(10, 20, 50, 60),
        plate_number=None,
        traffic_light_state="red",
        detection_confidence=0.87654,
        vehicle_class_id=2,
        vehicle_class_name="car",
        frame=frame,
    )
    duplicate = logger.log(
        violation_type="red_light",
        track_id=7,
        timestamp=1.5,
        frame_idx=30,
        bbox=(12, 22, 52, 62),
        frame=frame,
    )

    assert first is not None
    assert duplicate is None
    assert len(logger.violations) == 1
    assert first["plate_number"] == "UNKNOWN"
    assert first["traffic_light_state"] == "red"
    assert first["detection_confidence"] == 0.8765
    assert first["vehicle_class_id"] == 2
    assert first["vehicle_class_name"] == "car"
    assert Path(first["evidence_path"]).is_file()


def test_csv_supports_different_extra_fields(tmp_path):
    logger = ViolationLogger(
        output_dir=str(tmp_path), format="csv", save_frames=False
    )
    logger.log(
        "red_light", 1, 1.0, 10, (0, 0, 10, 10),
        traffic_light_state="red",
    )
    logger.log(
        "speeding", 2, 2.0, 20, (0, 0, 10, 10),
        extra={"speed_kmh": 72.5},
    )

    output_path = logger.save()
    with open(output_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert "speed_kmh" in rows[0]
    assert rows[1]["speed_kmh"] == "72.5"


def test_same_track_can_have_different_violation_types(tmp_path):
    logger = ViolationLogger(output_dir=str(tmp_path), save_frames=False)

    assert logger.log(
        "red_light", 3, 1.0, 10, (0, 0, 10, 10)
    ) is not None
    assert logger.log(
        "speeding", 3, 1.0, 10, (0, 0, 10, 10)
    ) is not None
    assert len(logger.violations) == 2


def test_json_lines_output(tmp_path):
    logger = ViolationLogger(
        output_dir=str(tmp_path), format="json", save_frames=False
    )
    logger.log("red_light", 9, 3.0, 90, (1, 2, 3, 4))

    output_path = logger.save()
    lines = Path(output_path).read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1
    assert json.loads(lines[0])["track_id"] == 9
