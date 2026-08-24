"""跳帧时间基准、速度和输出视频测试。"""

import cv2
import numpy as np
import pytest

from src.utils.video_utils import VideoProcessor
from src.violation.speeding import SpeedingDetector


def test_speed_uses_real_delta_time():
    detector = SpeedingDetector(
        pixel_to_meter_ratio=0.1,
        fps=30,
    )

    # 10 px = 1 m，在 0.2 秒内移动，速度应为 18 km/h。
    speed = detector.calculate_speed(
        1, (0, 0), (10, 0), delta_time=0.2
    )
    assert speed == pytest.approx(18.0)


def test_zero_delta_time_does_not_produce_infinite_speed():
    detector = SpeedingDetector(pixel_to_meter_ratio=0.1, fps=30)
    assert detector.calculate_speed(
        1, (0, 0), (10, 0), delta_time=0
    ) == 0


def test_speed_is_consistent_across_frame_skip_values():
    detector = SpeedingDetector(pixel_to_meter_ratio=0.1, fps=30)

    every_frame = detector.calculate_speed(
        1, (0, 0), (5, 0), delta_time=1 / 30
    )
    every_second_frame = detector.calculate_speed(
        1, (0, 0), (10, 0), delta_time=2 / 30
    )

    assert every_frame == pytest.approx(every_second_frame)


def test_continuous_speeding_is_logged_once_until_speed_drops():
    detector = SpeedingDetector(
        max_speed_kmh=10,
        pixel_to_meter_ratio=0.1,
        fps=30,
        smoothing_window=1,
    )
    bbox = (0, 0, 10, 10)

    first = detector.check_violation(
        1, (0, 0), (10, 0), bbox, 0.2, 6, delta_time=0.2
    )
    duplicate = detector.check_violation(
        1, (10, 0), (20, 0), bbox, 0.4, 12, delta_time=0.2
    )
    normal = detector.check_violation(
        1, (20, 0), (20, 0), bbox, 0.6, 18, delta_time=0.2
    )
    second_event = detector.check_violation(
        1, (20, 0), (30, 0), bbox, 0.8, 24, delta_time=0.2
    )

    assert first is not None
    assert duplicate is None
    assert normal is None
    assert second_event is not None
    assert len(detector.violations) == 2


def test_frame_skip_preserves_timestamps_and_output_duration(tmp_path):
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    fps = 20.0
    size = (64, 48)
    writer = cv2.VideoWriter(
        str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for value in range(20):
        writer.write(np.full((size[1], size[0], 3), value, dtype=np.uint8))
    writer.release()

    processor = VideoProcessor(
        str(input_path), str(output_path), frame_skip=2
    )
    yielded = []
    for frame_idx, timestamp, frame in processor.frames():
        yielded.append((frame_idx, timestamp))
        processor.write_frame(frame)

    assert yielded[0] == (0, 0.0)
    assert yielded[-1] == (18, 0.9)
    assert processor.get_processing_fps() == pytest.approx(10.0)

    output = cv2.VideoCapture(str(output_path))
    output_fps = output.get(cv2.CAP_PROP_FPS)
    output_frames = int(output.get(cv2.CAP_PROP_FRAME_COUNT))
    output.release()

    assert output_fps == pytest.approx(10.0, abs=0.1)
    assert output_frames == 10
    assert output_frames / output_fps == pytest.approx(1.0, abs=0.05)


def test_invalid_frame_skip_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="frame_skip"):
        VideoProcessor(str(tmp_path / "missing.mp4"), frame_skip=0)


def test_corrupted_video_is_rejected(tmp_path):
    video_path = tmp_path / "corrupted.mp4"
    video_path.write_text("not a video", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to open video"):
        VideoProcessor(str(video_path))
