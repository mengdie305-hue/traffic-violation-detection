"""车辆追踪模块测试。"""

import numpy as np

from src.tracking import DeepSORTTracker


def _detections(*boxes):
    return np.asarray(boxes, dtype=np.float32)


def _classes(count, class_id=2):
    return np.full(count, class_id, dtype=np.int32), ["car"] * count


def test_track_is_confirmed_after_min_hits():
    tracker = DeepSORTTracker(min_hits=3, max_distance=30)
    class_ids, class_names = _classes(1)

    assert tracker.update(_detections([0, 0, 10, 10]), class_ids, class_names) == []
    assert tracker.update(_detections([2, 0, 12, 10]), class_ids, class_names) == []
    tracks = tracker.update(_detections([4, 0, 14, 10]), class_ids, class_names)

    assert len(tracks) == 1
    assert tracks[0].track_id == 0
    assert tracks[0].total_frames == 3


def test_detection_order_does_not_swap_track_ids():
    tracker = DeepSORTTracker(min_hits=1, max_distance=50)
    class_ids, class_names = _classes(2)

    first = tracker.update(
        _detections([0, 0, 10, 10], [100, 0, 110, 10]),
        class_ids,
        class_names,
    )
    assert {track.track_id: track.center for track in first} == {
        0: (5.0, 5.0),
        1: (105.0, 5.0),
    }

    # 第二帧反转检测框顺序，ID 仍应跟随空间位置而不是数组索引。
    second = tracker.update(
        _detections([98, 0, 108, 10], [2, 0, 12, 10]),
        class_ids,
        class_names,
    )

    assert {track.track_id: track.center for track in second} == {
        0: (7.0, 5.0),
        1: (103.0, 5.0),
    }


def test_tracks_only_match_the_same_vehicle_class():
    tracker = DeepSORTTracker(min_hits=1, max_distance=50)
    tracker.update(
        _detections([0, 0, 10, 10]),
        np.asarray([2], dtype=np.int32),
        ["car"],
    )

    tracks = tracker.update(
        _detections([1, 0, 11, 10]),
        np.asarray([3], dtype=np.int32),
        ["motorcycle"],
    )

    assert {track.track_id for track in tracks} == {0, 1}
    assert tracker.tracks[0].frames_since_update == 1
    assert tracker.tracks[1].class_name == "motorcycle"


def test_track_survives_short_gap_and_expires_after_max_age():
    tracker = DeepSORTTracker(min_hits=1, max_age=2, max_distance=30)
    class_ids, class_names = _classes(1)
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_classes = np.empty((0,), dtype=np.int32)

    tracker.update(_detections([0, 0, 10, 10]), class_ids, class_names)
    assert len(tracker.update(empty_boxes, empty_classes, [])) == 1
    assert len(tracker.update(empty_boxes, empty_classes, [])) == 1
    assert tracker.update(empty_boxes, empty_classes, []) == []


def test_short_gap_recovers_original_id():
    tracker = DeepSORTTracker(min_hits=1, max_age=2, max_distance=30)
    class_ids, class_names = _classes(1)
    empty_boxes = np.empty((0, 4), dtype=np.float32)
    empty_classes = np.empty((0,), dtype=np.int32)

    tracker.update(_detections([0, 0, 10, 10]), class_ids, class_names)
    tracker.update(empty_boxes, empty_classes, [])
    tracks = tracker.update(_detections([4, 0, 14, 10]), class_ids, class_names)

    assert len(tracks) == 1
    assert tracks[0].track_id == 0
