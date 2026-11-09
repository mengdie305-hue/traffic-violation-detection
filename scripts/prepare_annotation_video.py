"""Create a trimmed/cropped annotation copy without modifying source video.

The command intentionally performs only deterministic spatial and temporal
cleaning.  It does not enhance images or change playback speed, so frame-based
annotations can be mapped back to the original recording.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="Exclusive end time in seconds")
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--crop-bottom", type=int, default=0)
    return parser.parse_args()


def prepare_video(args: argparse.Namespace) -> dict[str, float | int | str]:
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise ValueError(f"Cannot open input video: {args.input}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video metadata is invalid")

    duration = frame_count / fps
    end_seconds = duration if args.end is None else args.end
    if not 0 <= args.start < end_seconds <= duration + (1 / fps):
        capture.release()
        raise ValueError(f"Invalid time range: {args.start}..{end_seconds} for {duration:.3f}s video")
    if args.crop_top < 0 or args.crop_bottom < 0 or args.crop_top + args.crop_bottom >= height:
        capture.release()
        raise ValueError("Invalid vertical crop")

    start_frame = round(args.start * fps)
    end_frame = min(round(end_seconds * fps), frame_count)
    output_height = height - args.crop_top - args.crop_bottom
    if width % 2 or output_height % 2:
        capture.release()
        raise ValueError("Output dimensions must be even for MP4 encoding")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, output_height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Cannot create output video: {args.output}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    try:
        for _ in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            bottom = height - args.crop_bottom if args.crop_bottom else height
            writer.write(frame[args.crop_top:bottom, :])
            written += 1
    finally:
        capture.release()
        writer.release()

    if written != end_frame - start_frame:
        args.output.unlink(missing_ok=True)
        raise ValueError(f"Decoded {written} frames, expected {end_frame - start_frame}")

    return {
        "input": str(args.input),
        "output": str(args.output),
        "source_fps": fps,
        "source_frames": frame_count,
        "source_width": width,
        "source_height": height,
        "start_frame": start_frame,
        "end_frame_exclusive": end_frame,
        "output_frames": written,
        "output_width": width,
        "output_height": output_height,
        "output_duration_seconds": written / fps,
    }


def main() -> None:
    result = prepare_video(parse_args())
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
