"""闯红灯跨线状态机测试。"""

from src.detection.traffic_light import TrafficLightState
from src.violation import CrossingState, RedLightDetector


def _detector(red_light_duration=0.0):
    return RedLightDetector(
        stop_line=(100, 300, 500, 300),
        red_light_duration=red_light_duration,
    )


def _check(detector, track_id, bbox, light, timestamp, frame_idx):
    return detector.check_violation(
        track_id,
        bbox,
        light,
        timestamp,
        frame_idx,
        fps=30,
    )


def test_red_light_forward_crossing_creates_one_event():
    detector = _detector()

    assert _check(
        detector, 1, (200, 220, 300, 290),
        TrafficLightState.RED, 1.0, 30,
    ) is None
    violation = _check(
        detector, 1, (200, 250, 300, 320),
        TrafficLightState.RED, 1.1, 33,
    )

    assert violation is not None
    assert violation.track_id == 1
    assert detector._track_states[1].phase == CrossingState.CROSSING_LINE

    # 后续仍在停止线后方时不能重复产生事件。
    assert _check(
        detector, 1, (200, 280, 300, 350),
        TrafficLightState.RED, 1.2, 36,
    ) is None
    assert len(detector.violations) == 1


def test_vehicle_that_crossed_on_green_is_not_flagged_when_light_turns_red():
    detector = _detector()

    _check(
        detector, 2, (200, 220, 300, 290),
        TrafficLightState.GREEN, 1.0, 30,
    )
    assert _check(
        detector, 2, (200, 250, 300, 320),
        TrafficLightState.GREEN, 1.1, 33,
    ) is None
    assert _check(
        detector, 2, (200, 280, 300, 350),
        TrafficLightState.RED, 2.0, 60,
    ) is None


def test_first_seen_after_line_is_not_a_violation():
    detector = _detector()

    assert _check(
        detector, 3, (200, 280, 300, 350),
        TrafficLightState.RED, 1.0, 30,
    ) is None
    assert _check(
        detector, 3, (200, 300, 300, 370),
        TrafficLightState.RED, 1.1, 33,
    ) is None


def test_red_light_must_be_stable_before_crossing():
    detector = _detector(red_light_duration=0.5)
    detector.update_light_state(TrafficLightState.RED, 1.0)

    _check(
        detector, 4, (200, 220, 300, 290),
        TrafficLightState.RED, 1.1, 33,
    )
    assert _check(
        detector, 4, (200, 250, 300, 320),
        TrafficLightState.RED, 1.2, 36,
    ) is None


def test_crossing_outside_stop_line_span_is_ignored():
    detector = _detector()

    _check(
        detector, 5, (600, 220, 700, 290),
        TrafficLightState.RED, 1.0, 30,
    )
    assert _check(
        detector, 5, (600, 250, 700, 320),
        TrafficLightState.RED, 1.1, 33,
    ) is None


def test_upward_crossing_creates_one_event():
    detector = RedLightDetector(
        stop_line=(100, 300, 500, 300),
        red_light_duration=0.0,
        crossing_direction="upward",
    )

    assert _check(
        detector, 6, (200, 280, 300, 330),
        TrafficLightState.RED, 1.0, 30,
    ) is None
    violation = _check(
        detector, 6, (200, 230, 300, 290),
        TrafficLightState.RED, 1.1, 33,
    )

    assert violation is not None
    assert violation.track_id == 6
    assert detector._track_states[6].phase == CrossingState.CROSSING_LINE


def test_invalid_crossing_direction_is_rejected():
    try:
        RedLightDetector(
            stop_line=(100, 300, 500, 300),
            crossing_direction="diagonal",
        )
    except ValueError as exc:
        assert "crossing_direction" in str(exc)
    else:
        raise AssertionError("invalid crossing direction should raise ValueError")
