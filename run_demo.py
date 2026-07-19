#!/usr/bin/env python3
"""
Demo 脚本 — 在模拟视频上运行完整 Pipeline。

使用 OpenCV 颜色检测替代 YOLO（因为模拟车辆是简单方块），
其余模块（DeepSORT 追踪、违章判定、日志）全部使用真实代码。
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.tracking import DeepSORTTracker
from src.detection.traffic_light import TrafficLightDetector, TrafficLightState
from src.violation import RedLightDetector, SpeedingDetector, WrongWayDetector
from src.utils.draw_utils import Visualizer
from src.utils.log_utils import ViolationLogger

# ============================================================
# 模拟视频参数（需与 generate_test_video.py 一致）
# ============================================================
STOP_LINE_Y = 320
LIGHT_POS = (640, 60)
LIGHT_RADIUS = 25
W, H = 1280, 720


def detect_vehicles_mock(frame):
    """
    模拟车辆检测：用颜色阈值 + 轮廓检测找到彩色方块车。

    Returns:
        boxes: [N, 4] (x1, y1, x2, y2)
        class_ids: [N]
        class_names: [N]
    """
    # 车辆颜色范围 (BGR)
    car_colors = [
        (np.array([200, 0, 0]), np.array([255, 50, 50]), "car"),      # 蓝
        (np.array([0, 200, 0]), np.array([50, 255, 50]), "car"),      # 绿
        (np.array([0, 0, 200]), np.array([50, 50, 255]), "car"),      # 红
        (np.array([0, 200, 200]), np.array([50, 255, 255]), "car"),   # 黄
        (np.array([200, 0, 200]), np.array([255, 50, 255]), "car"),    # 紫
    ]

    boxes, class_ids, class_names = [], [], []

    for class_id, (lower, upper, name) in enumerate(car_colors):
        mask = cv2.inRange(frame, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            class_ids.append(class_id)
            class_names.append(name)

    if boxes:
        return (
            np.array(boxes, dtype=np.float32),
            np.array(class_ids, dtype=np.int32),
            class_names,
        )
    return np.empty((0, 4)), np.empty((0,), dtype=np.int32), []


def detect_traffic_light_mock(frame):
    """
    模拟交通灯检测：在红绿灯区域检测当前亮灯颜色。
    """
    light_detector = TrafficLightDetector(
        brightness_threshold=30,   # 模拟灯的灰度值较低
        min_area_ratio=0.02,
    )
    x, y = LIGHT_POS
    r = LIGHT_RADIUS

    # 检查三个灯位（需与 generate_test_video.py 的 dict 顺序一致）
    for i, state_name in enumerate(["red", "yellow", "green"]):
        ly = y - 55 + i * 55
        # 边界裁剪
        y1, y2 = max(0, ly - r), min(H, ly + r)
        x1, x2 = max(0, x - r), min(W, x + r)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        result = light_detector.recognize(roi)
        if result != TrafficLightState.UNKNOWN:
            return result

    return TrafficLightState.UNKNOWN


def main():
    print("=" * 60)
    print("  Traffic Violation Detection — DEMO")
    print("  Mock detection + Real pipeline")
    print("=" * 60)

    # 输入输出
    input_path = "data/videos/test_intersection.mp4"
    output_path = "data/videos/test_intersection_output.mp4"

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Pipeline 模块
    tracker = DeepSORTTracker(min_hits=3)
    # 降低亮度阈值适配模拟灯（真实场景用默认值 100）
    light_detector = TrafficLightDetector(
        brightness_threshold=30,  # 模拟灯的灰度值较低
        min_area_ratio=0.02,
    )
    visualizer = Visualizer()
    logger = ViolationLogger(output_dir="data/logs", format="csv", save_frames=True)

    red_light_detector = RedLightDetector(
        stop_line=(280, STOP_LINE_Y, 1000, STOP_LINE_Y),
        overlap_threshold=0.05,
        red_light_duration=0.5,    # 红灯越线持续 0.5 秒判定违章
    )
    speeding_detector = SpeedingDetector(
        max_speed_kmh=200,          # 模拟速度偏高
        pixel_to_meter_ratio=0.03,
        fps=fps,
    )
    wrong_way_detector = WrongWayDetector(
        lane_directions={},
        angle_threshold=150,        # 模拟车全向下，大幅放宽
        min_history=30,             # 需要较长轨迹
    )

    violation_ids = set()

    print(f"\nProcessing: {input_path}")
    print(f"Frames: {frame_count}, FPS: {fps:.1f}")
    print("-" * 60)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # 1. 检测
        vehicle_boxes, vehicle_classes, vehicle_names = detect_vehicles_mock(frame)
        light_state = detect_traffic_light_mock(frame)

        # 2. 追踪
        tracks = tracker.update(vehicle_boxes, vehicle_classes, vehicle_names)

        # 3. 违章判定
        for track in tracks:
            tid = track.track_id

            # 闯红灯
            violation = red_light_detector.check_violation(
                tid, track.bbox, light_state, timestamp, frame_idx, fps
            )
            if violation:
                violation_ids.add(tid)
                logger.log(
                    violation_type="red_light",
                    track_id=tid,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    bbox=track.bbox,
                    plate_number=f"XX-{tid:04d}",
                )

            # 超速
            if len(track.history) >= 2:
                prev = track.history[-2]
                curr = track.center
                violating = speeding_detector.check_violation(
                    tid, prev, curr, track.bbox, timestamp, frame_idx
                )
                if violating:
                    violation_ids.add(tid)
                    logger.log(
                        violation_type="speeding",
                        track_id=tid,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        bbox=track.bbox,
                        plate_number=f"XX-{tid:04d}",
                        extra={"speed_kmh": violating.speed_kmh},
                    )

            # 逆行（模拟视频全向下，跳过）
            pass
            # if len(track.history) >= wrong_way_detector.min_history:
            #     violating = wrong_way_detector.check_violation(
            #         tid, track.history, track.bbox, timestamp, frame_idx
            #     )
            #     if violating:
            #         violation_ids.add(tid)
            #         logger.log(
            #             violation_type="wrong_way",
            #             track_id=tid,
            #             timestamp=timestamp,
            #             frame_idx=frame_idx,
            #             bbox=track.bbox,
            #             plate_number=f"XX-{tid:04d}",
            #             extra={"angle": violating.direction_angle},
            #         )

        # 4. 可视化
        annotated = frame.copy()
        annotated = visualizer.draw_tracks(annotated, tracks, violation_ids)
        annotated = visualizer.draw_traffic_light_status(
            annotated, light_state.value, (LIGHT_POS[0] - 35, LIGHT_POS[1] - 90,
                                             LIGHT_POS[0] + 35, LIGHT_POS[1] + 90)
        )

        if len(violation_ids) > 0:
            annotated = visualizer.draw_violation_alert(annotated, "DETECTED")

        cv2.putText(annotated, f"Tracks: {len(tracks)} | Violations: {len(logger.violations)}",
                    (W - 450, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(annotated)

        # 实时显示
        cv2.imshow("Traffic Demo", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_idx % 60 == 0:
            print(f"  Frame {frame_idx:4d} | Tracks: {len(tracks):2d} | "
                  f"Light: {light_state.value:7s} | Violations: {len(logger.violations)}")

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # 保存日志
    log_path = logger.save()
    summary = logger.get_summary()

    print("\n" + "=" * 60)
    print("  DEMO Complete!")
    print("=" * 60)
    print(f"  Frames:            {frame_idx}")
    print(f"  Total violations:  {len(logger.violations)}")
    for vtype, count in summary.items():
        print(f"    - {vtype}: {count}")
    print(f"  Log:               {log_path}")
    print(f"  Output video:      {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
