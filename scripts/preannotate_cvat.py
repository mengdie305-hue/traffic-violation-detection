"""Generate review-first CVAT track XML with YOLO and DeepSORT.

The output is a machine suggestion, not ground truth.  Create a CVAT video task
with labels car, motorcycle, bus and truck, then import the generated XML and
review every track before exporting final annotations.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection.yolo_detector import VEHICLE_CLASSES, YOLODetector  # noqa: E402
from src.tracking import create_tracker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "data/models/yolov8n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-step", type=int, default=6, help="Run inference every N frames")
    parser.add_argument("--confidence", type=float, default=0.25)
    return parser.parse_args()


def write_cvat_xml(output: Path, video_name: str, frame_count: int, tracks: dict) -> None:
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = video_name
    ET.SubElement(task, "size").text = str(frame_count)

    for output_id, ((track_id, label), boxes) in enumerate(sorted(tracks.items())):
        track_node = ET.SubElement(
            root,
            "track",
            id=str(output_id),
            label=label,
            source="auto",
        )
        for item in boxes:
            ET.SubElement(
                track_node,
                "box",
                frame=str(item[0]),
                outside="0",
                occluded="0",
                keyframe="1",
                xtl=f"{item[1]:.2f}",
                ytl=f"{item[2]:.2f}",
                xbr=f"{item[3]:.2f}",
                ybr=f"{item[4]:.2f}",
                z_order="0",
            )

    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise ValueError("frame-step must be at least 1")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = YOLODetector(
        model_path=str(args.model),
        conf_threshold=args.confidence,
        device=args.device,
    )
    tracker = create_tracker(
        backend="realtime_deepsort",
        max_age=15,
        min_hits=2,
        max_iou_distance=0.7,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        embedder_gpu=args.device == "cuda",
    )

    suggestions: dict[tuple[int, str], list[tuple]] = defaultdict(list)
    frame_index = 0
    processed = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % args.frame_step:
            frame_index += 1
            continue

        boxes, scores, class_ids = detector.detect_vehicles(frame)
        names = [VEHICLE_CLASSES[int(class_id)] for class_id in class_ids]
        active_tracks = tracker.update(boxes, class_ids, names, scores, frame=frame)
        for track in active_tracks:
            if not track.was_updated:
                continue
            confidence = track.confidence if track.confidence is not None else 0.0
            x1, y1, x2, y2 = track.bbox
            suggestions[(track.track_id, track.class_name)].append(
                (frame_index, x1, y1, x2, y2, confidence)
            )

        processed += 1
        if processed % 100 == 0:
            print(f"processed {processed} sampled frames / source frame {frame_index}", flush=True)
        frame_index += 1

    capture.release()
    write_cvat_xml(args.output, args.video.name, frame_count, suggestions)
    box_count = sum(len(items) for items in suggestions.values())
    print(f"wrote {len(suggestions)} tracks and {box_count} keyframes to {args.output}")


if __name__ == "__main__":
    main()
