#!/usr/bin/env python3
"""
Traffic Violation Detection — 主流程入口。

编排完整的检测流水线：
    视频读取 → 检测（YOLO / Mock）→ DeepSORT 追踪 → 交通灯识别
    → 违章判定（闯红灯 / 超速 / 逆行）→ 车牌识别 → 日志保存

Usage:
    # 真实 YOLO 推理
    python src/main.py --video data/videos/sample.mp4 --config configs/config.yaml

    # Mock 模式（在模拟视频上跑完整 Pipeline，无需 GPU）
    python src/main.py --mock --video data/videos/test_intersection.mp4 --show
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np
import yaml

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection import YOLODetector, TrafficLightDetector
from src.detection.traffic_light import TrafficLightState
from src.tracking import create_tracker
from src.recognition import PlateRecognizer
from src.violation import RedLightDetector, SpeedingDetector, WrongWayDetector
from src.utils.video_utils import VideoProcessor, draw_roi
from src.utils.draw_utils import Visualizer
from src.utils.log_utils import ViolationLogger


# COCO 类别名映射
COCO_NAMES = {
    0: "person", 2: "car", 3: "motorcycle",
    5: "bus", 7: "truck", 9: "traffic_light",
}

# ============================================================
# Mock 模式常量（需与 scripts/generate_test_video.py 一致）
# ============================================================
MOCK_STOP_LINE_Y = 320
MOCK_LIGHT_POS = (640, 60)
MOCK_LIGHT_RADIUS = 25
MOCK_W, MOCK_H = 1280, 720


# ============================================================
# Mock 检测函数（模拟车辆检测 + 交通灯识别）
# ============================================================

def _detect_vehicles_mock(frame):
    """
    模拟车辆检测：用颜色阈值 + 轮廓检测找到彩色方块车。

    Returns:
        boxes:      np.ndarray [N, 4] (x1, y1, x2, y2)
        scores:     np.ndarray [N]   (置信度，mock 填 1.0)
        class_ids:  np.ndarray [N]
        class_names: list[str]
    """
    car_colors = [
        (np.array([200, 0, 0]),   np.array([255, 50, 50]),  "car"),   # 蓝
        (np.array([0, 200, 0]),   np.array([50, 255, 50]),  "car"),   # 绿
        (np.array([0, 0, 200]),   np.array([50, 50, 255]),  "car"),   # 红
        (np.array([0, 200, 200]), np.array([50, 255, 255]), "car"),   # 黄
        (np.array([200, 0, 200]), np.array([255, 50, 255]), "car"),   # 紫
    ]

    boxes_list, scores_list, class_ids_list, class_names_list = [], [], [], []

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
            boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
            scores_list.append(1.0)
            class_ids_list.append(class_id)
            class_names_list.append(name)

    if boxes_list:
        return (
            np.array(boxes_list, dtype=np.float32),
            np.array(scores_list, dtype=np.float32),
            np.array(class_ids_list, dtype=np.int32),
            class_names_list,
        )
    return (
        np.empty((0, 4), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int32),
        [],
    )


def _detect_traffic_light_mock(frame):
    """
    模拟交通灯检测：在红绿灯区域检测当前亮灯颜色。

    Returns:
        TrafficLightState
    """
    light_detector = TrafficLightDetector(
        brightness_threshold=30,   # 模拟灯的灰度值较低
        min_area_ratio=0.02,
    )
    x, y = MOCK_LIGHT_POS
    r = MOCK_LIGHT_RADIUS

    for i, state_name in enumerate(["red", "yellow", "green"]):
        ly = y - 55 + i * 55
        y1, y2 = max(0, ly - r), min(MOCK_H, ly + r)
        x1, x2 = max(0, x - r), min(MOCK_W, x + r)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        result = light_detector.recognize(roi)
        if result != TrafficLightState.UNKNOWN:
            return result

    return TrafficLightState.UNKNOWN


# ============================================================
# 主流程
# ============================================================

def load_config(config_path: str) -> Dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection System"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default="data/videos/sample.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/config.yaml",
        help="Configuration YAML path",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock detection (color-based) for the test intersection video",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show real-time visualization",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path with annotations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Max frames to process (for debugging)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # ==========================================
    # 1. 初始化各模块
    # ==========================================
    print("=" * 60)
    mode_str = "MOCK (color-based)" if args.mock else "YOLO (real)"
    print(f"Traffic Violation Detection System — {mode_str}")
    print("=" * 60)

    # --- 视频处理器 ---
    video_cfg = config["video"]
    video_cfg["input_path"] = args.video or video_cfg.get("input_path")
    video = VideoProcessor(
        input_path=video_cfg["input_path"],
        output_path=args.output or video_cfg.get("output_path"),
        resize=(
            video_cfg.get("resize_width", 1280),
            video_cfg.get("resize_height", 720),
        ),
        frame_skip=video_cfg.get("frame_skip", 2),
    )

    # --- YOLO 检测器（非 mock 模式才加载）---
    if not args.mock:
        model_cfg = config["models"]
        detector = YOLODetector(
            model_path=model_cfg.get("yolo", "yolov8n.pt"),
            device=args.device,
        )
    else:
        detector = None

    # --- 追踪器：真实模式使用完整 DeepSORT，Mock 使用轻量确定性后端 ---
    tracking_cfg = config.get("tracking", {})
    tracking_backend = (
        "simple" if args.mock
        else tracking_cfg.get("backend", "realtime_deepsort")
    )
    tracker = create_tracker(
        backend=tracking_backend,
        max_age=tracking_cfg.get("max_age", 30),
        min_hits=tracking_cfg.get("min_hits", 3),
        max_distance=tracking_cfg.get("max_distance", 50.0),
        max_iou_distance=tracking_cfg.get("max_iou_distance", 0.7),
        max_cosine_distance=tracking_cfg.get("max_cosine_distance", 0.2),
        embedder=tracking_cfg.get("embedder", "mobilenet"),
        embedder_gpu=(args.device == "cuda"),
    )

    # --- 交通灯识别（mock 模式下内联处理，不用此实例）---
    tl_detector = TrafficLightDetector(
        brightness_threshold=30 if args.mock else 100,
        min_area_ratio=0.02 if args.mock else 0.05,
    )

    # --- 车牌识别 ---
    recognition_cfg = config.get("recognition", {})
    plate_recognizer = PlateRecognizer(
        backend=("mock" if args.mock else recognition_cfg.get("backend", "paddleocr")),
        lang=recognition_cfg.get("lang", "en"),
        fallback_to_mock=(
            True if args.mock else recognition_cfg.get("fallback_to_mock", False)
        ),
        sample_interval=recognition_cfg.get("sample_interval", 5),
        vote_window=recognition_cfg.get("vote_window", 5),
        plate_region_start=recognition_cfg.get("plate_region_start", 0.5),
    )

    # --- 违章检测器 ---
    roi_cfg = config["roi"]
    v_cfg = config["violation"]

    # Mock 模式下用模拟视频专用的停止线坐标
    if args.mock:
        stop_line = (280, MOCK_STOP_LINE_Y, 1000, MOCK_STOP_LINE_Y)
    else:
        stop_line = tuple(roi_cfg["stop_line"])

    red_light_detector = RedLightDetector(
        stop_line=stop_line,
        overlap_threshold=0.05 if args.mock else v_cfg["red_light"]["overlap_threshold"],
        red_light_duration=0.5 if args.mock else v_cfg["red_light"]["red_light_duration"],
    )
    speeding_detector = SpeedingDetector(
        max_speed_kmh=200 if args.mock else v_cfg["speeding"]["max_speed_kmh"],
        pixel_to_meter_ratio=0.03 if args.mock else v_cfg["speeding"]["pixel_to_meter_ratio"],
        fps=video.get_fps(),
    )
    wrong_way_detector = WrongWayDetector(
        lane_directions={},
        angle_threshold=150 if args.mock else v_cfg["wrong_way"]["angle_threshold"],
        min_history=30 if args.mock else 10,
    )

    # --- 可视化 & 日志 ---
    visualizer = Visualizer()
    log_cfg = config["logging"]
    logger = ViolationLogger(
        output_dir=log_cfg["output_dir"],
        format=log_cfg["format"],
        save_frames=log_cfg["save_frames"],
    )

    tl_roi = tuple(roi_cfg["traffic_light_bbox"])

    # ==========================================
    # 2. 逐帧处理
    # ==========================================
    print("\nProcessing video...")
    violation_ids = set()
    previous_observations = {}
    processed_count = 0
    last_frame_idx = -1

    for frame_idx, timestamp, frame in video.frames():
        if args.max_frames is not None and processed_count >= args.max_frames:
            break
        processed_count += 1
        last_frame_idx = frame_idx

        # --- 2a. 车辆检测 ---
        if args.mock:
            vehicle_boxes, vehicle_scores, vehicle_classes, vehicle_names = \
                _detect_vehicles_mock(frame)
        else:
            vehicle_boxes, vehicle_scores, vehicle_classes = detector.detect_vehicles(frame)
            vehicle_names = [
                COCO_NAMES.get(int(cid), "unknown") for cid in vehicle_classes
            ]

        # --- 2b. 交通灯检测 ---
        if args.mock:
            # Mock 模式：直接从固定位置识别颜色
            light_state = _detect_traffic_light_mock(frame)
        else:
            tl_boxes, tl_scores, tl_classes = detector.detect_traffic_lights(frame)
            light_state = TrafficLightState.UNKNOWN
            if len(tl_boxes) > 0:
                best_idx = int(np.argmax(tl_scores))
                x1, y1, x2, y2 = [int(v) for v in tl_boxes[best_idx]]
                tl_roi = (x1, y1, x2, y2)
                light_state = tl_detector.recognize(frame[y1:y2, x1:x2])

            else:
                x1, y1, x2, y2 = tl_roi
                light_state = tl_detector.recognize(frame[y1:y2, x1:x2])

        # 即使当前帧没有车辆，也要维护连续红灯的真实开始时间。
        red_light_detector.update_light_state(light_state, timestamp)

        # --- 2c. 车辆追踪 ---
        tracks = tracker.update(
            vehicle_boxes, vehicle_classes, vehicle_names,
            vehicle_scores, frame=frame,
        )

        # --- 2d. 违章检测 ---
        for track in tracks:
            tid = track.track_id
            if track.was_updated:
                plate_recognizer.observe(tid, frame, track.bbox)

            # 闯红灯检测
            violation = red_light_detector.check_violation(
                tid, track.bbox, light_state,
                timestamp, frame_idx, video.get_fps(),
            )
            if violation:
                violation_ids.add(tid)
                plate = plate_recognizer.get_consensus(tid)
                logger.log(
                    violation_type="red_light",
                    track_id=tid,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    bbox=track.bbox,
                    plate_number=plate,
                    traffic_light_state=light_state.value,
                    detection_confidence=track.confidence,
                    vehicle_class_id=int(track.class_id),
                    vehicle_class_name=track.class_name,
                    frame=frame,
                )

            # 超速检测
            if track.was_updated and tid in previous_observations:
                prev_center, prev_timestamp = previous_observations[tid]
                curr_center = track.center
                violation = speeding_detector.check_violation(
                    tid, prev_center, curr_center,
                    track.bbox, timestamp, frame_idx,
                    delta_time=timestamp - prev_timestamp,
                )
                if violation:
                    violation_ids.add(tid)
                    plate = plate_recognizer.get_consensus(tid)
                    logger.log(
                        violation_type="speeding",
                        track_id=tid,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        bbox=track.bbox,
                        plate_number=plate,
                        traffic_light_state=light_state.value,
                        detection_confidence=track.confidence,
                        vehicle_class_id=int(track.class_id),
                        vehicle_class_name=track.class_name,
                        frame=frame,
                        extra={"speed_kmh": violation.speed_kmh},
                    )

            if track.was_updated:
                previous_observations[tid] = (track.center, timestamp)

            # 逆行检测（Mock 模式下跳过：所有车辆同向行驶，无逆行场景）
            if not args.mock and len(track.history) >= wrong_way_detector.min_history:
                violation = wrong_way_detector.check_violation(
                    tid, track.history,
                    track.bbox, timestamp, frame_idx,
                )
                if violation:
                    violation_ids.add(tid)
                    plate = plate_recognizer.get_consensus(tid)
                    logger.log(
                        violation_type="wrong_way",
                        track_id=tid,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        bbox=track.bbox,
                        plate_number=plate,
                        traffic_light_state=light_state.value,
                        detection_confidence=track.confidence,
                        vehicle_class_id=int(track.class_id),
                        vehicle_class_name=track.class_name,
                        frame=frame,
                        extra={"angle": violation.direction_angle},
                    )

        # --- 2e. 可视化 ---
        annotated_frame = frame.copy()

        # ROI 标注
        annotated_frame = draw_roi(annotated_frame, roi_cfg)

        # 追踪框 & 轨迹
        annotated_frame = visualizer.draw_tracks(
            annotated_frame, tracks, violation_ids
        )

        # 信号灯状态
        if args.mock:
            tl_display_roi = (
                MOCK_LIGHT_POS[0] - 35, MOCK_LIGHT_POS[1] - 90,
                MOCK_LIGHT_POS[0] + 35, MOCK_LIGHT_POS[1] + 90,
            )
        else:
            tl_display_roi = tl_roi
        annotated_frame = visualizer.draw_traffic_light_status(
            annotated_frame, light_state.value, tl_display_roi
        )

        # 违章横幅
        if violation_ids:
            annotated_frame = visualizer.draw_violation_alert(
                annotated_frame, "violation detected"
            )

        # 写入输出视频
        video.write_frame(annotated_frame)

        # 实时显示
        if args.show:
            cv2.imshow("Traffic Violation Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # 进度
        if frame_idx % 60 == 0:
            print(
                f"  Frame {frame_idx:4d} | "
                f"Tracks: {len(tracks):2d} | "
                f"Light: {light_state.value:7s} | "
                f"Violations: {len(logger.violations)}"
            )

    # ==========================================
    # 3. 保存结果
    # ==========================================
    cv2.destroyAllWindows()

    log_path = logger.save()
    summary = logger.get_summary()

    print("\n" + "=" * 60)
    print("  Processing Complete")
    print("=" * 60)
    print(f"  Frames processed:  {processed_count}")
    print(f"  Last source frame:  {last_frame_idx}")
    print(f"  Total violations:  {len(logger.violations)}")
    for vtype, count in summary.items():
        print(f"    - {vtype}: {count}")
    if log_path:
        print(f"  Log saved to:      {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
