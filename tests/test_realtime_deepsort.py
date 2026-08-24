"""完整 DeepSORT 后端集成测试。"""

import numpy as np

from src.tracking import RealtimeDeepSORTTracker, create_tracker


def test_realtime_deepsort_tracks_a_vehicle_across_frames():
    tracker = create_tracker(
        "realtime_deepsort",
        max_age=5,
        min_hits=1,
        embedder="mobilenet",
        embedder_gpu=False,
    )
    assert isinstance(tracker, RealtimeDeepSORTTracker)

    frame = np.random.default_rng(0).integers(
        0, 256, (120, 160, 3), dtype=np.uint8
    )
    class_ids = np.asarray([2], dtype=np.int32)
    scores = np.asarray([0.9], dtype=np.float32)
    observed_ids = []

    for offset in (0, 2, 4):
        boxes = np.asarray(
            [[20 + offset, 20, 80 + offset, 80]], dtype=np.float32
        )
        tracks = tracker.update(
            boxes, class_ids, ["car"], scores, frame=frame
        )
        if tracks:
            observed_ids.append(tracks[0].track_id)
            assert tracks[0].class_name == "car"
            assert tracks[0].class_id == 2

    assert len(observed_ids) >= 2
    assert len(set(observed_ids)) == 1


def test_unknown_tracking_backend_is_rejected():
    try:
        create_tracker("unknown")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unsupported backend should raise ValueError")


def test_two_vehicles_keep_ids_while_crossing_and_detection_order_changes():
    tracker = create_tracker(
        "realtime_deepsort",
        max_age=5,
        min_hits=1,
        max_iou_distance=0.9,
        max_cosine_distance=0.4,
        embedder="mobilenet",
        embedder_gpu=False,
    )
    positions = [
        (10, 120), (20, 110), (30, 100), (40, 90), (50, 80),
        (60, 70), (70, 60), (80, 50), (90, 40), (100, 30), (110, 20),
    ]
    center_history = {}

    for frame_index, (first_x, second_x) in enumerate(positions):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        frame[30:80, first_x:first_x + 40] = (0, 0, 255)
        frame[30:80, second_x:second_x + 40] = (255, 0, 0)
        boxes = np.asarray([
            [first_x, 30, first_x + 40, 80],
            [second_x, 30, second_x + 40, 80],
        ], dtype=np.float32)
        if frame_index % 2:
            boxes = boxes[::-1]

        tracks = tracker.update(
            boxes,
            np.asarray([2, 2], dtype=np.int32),
            ["car", "car"],
            np.asarray([0.95, 0.95], dtype=np.float32),
            frame=frame,
        )
        for track in tracks:
            center_history.setdefault(track.track_id, []).append(track.center[0])

    assert len(center_history) == 2
    directions = sorted(
        history[-1] - history[0] for history in center_history.values()
    )
    assert directions[0] < -80
    assert directions[1] > 80
