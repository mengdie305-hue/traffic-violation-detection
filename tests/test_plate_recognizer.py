"""车牌候选区域、文本校验和多帧投票测试。"""

import numpy as np

from src.recognition import PlateRecognizer


def test_extracts_lower_part_of_vehicle_bbox():
    recognizer = PlateRecognizer(
        backend="disabled", plate_region_start=0.5
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = recognizer._extract_plate_region(frame, (20, 10, 120, 90))

    assert crop is not None
    assert crop.shape == (40, 100, 3)


def test_plate_text_normalization_and_validation():
    assert PlateRecognizer._normalize_plate(" ab-1234 ") == "AB1234"
    assert PlateRecognizer._normalize_plate("A-1") is None
    assert PlateRecognizer._normalize_plate("ABCDEFGHIJKLM") is None


def test_disabled_real_backend_returns_none_not_mock_number():
    recognizer = PlateRecognizer(backend="disabled")
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    assert recognizer.recognize(frame, (0, 0, 100, 50)) is None


def test_multiframe_vote_prefers_most_frequent_result(monkeypatch):
    recognizer = PlateRecognizer(
        backend="disabled", sample_interval=1, vote_window=5
    )
    results = iter(["ABC123", "ABC128", "ABC123"])
    monkeypatch.setattr(recognizer, "recognize", lambda frame, bbox: next(results))
    frame = np.zeros((50, 100, 3), dtype=np.uint8)

    recognizer.observe(1, frame, (0, 0, 100, 50))
    recognizer.observe(1, frame, (0, 0, 100, 50))
    consensus = recognizer.observe(1, frame, (0, 0, 100, 50))

    assert consensus == "ABC123"


def test_unsupported_backend_is_rejected():
    try:
        PlateRecognizer(backend="unknown")
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("unsupported backend should raise ValueError")
