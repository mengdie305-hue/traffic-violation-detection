"""Mock Pipeline 端到端回归测试。"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_pipeline(video_path: Path, config_path: Path, output_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "src/main.py"),
            "--mock",
            "--video", str(video_path),
            "--config", str(config_path),
            "--output", str(output_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result


def _read_single_log(log_dir: Path):
    logs = list(log_dir.glob("violations_*.csv"))
    assert len(logs) == 1
    with logs[0].open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_mock_pipeline_matches_manifest_and_is_repeatable(tmp_path):
    video_path = tmp_path / "test_intersection.mp4"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generate_test_video.py"),
            "--output", str(video_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    base_config = yaml.safe_load(
        (PROJECT_ROOT / "configs/config.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (PROJECT_ROOT / "data/annotations/mock_events.json").read_text(
            encoding="utf-8"
        )
    )

    runs = []
    for run_number in (1, 2):
        run_dir = tmp_path / f"run-{run_number}"
        log_dir = run_dir / "logs"
        output_path = run_dir / "output.mp4"
        config = dict(base_config)
        config["logging"] = {
            "output_dir": str(log_dir),
            "format": "csv",
            "save_frames": True,
        }
        config_path = run_dir / "config.yaml"
        run_dir.mkdir(parents=True)
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
        )

        result = _run_pipeline(video_path, config_path, output_path)
        assert result.returncode == 0
        rows = _read_single_log(log_dir)
        runs.append([(row["violation_type"], int(row["frame_idx"])) for row in rows])

        assert len(rows) == len(manifest["expected_events"])
        for row, expected in zip(rows, manifest["expected_events"]):
            assert row["violation_type"] == expected["violation_type"]
            assert expected["expected_frame_range"][0] <= int(row["frame_idx"])
            assert int(row["frame_idx"]) <= expected["expected_frame_range"][1]
            assert Path(row["evidence_path"]).is_file()

        output = cv2.VideoCapture(str(output_path))
        fps = output.get(cv2.CAP_PROP_FPS)
        frame_count = int(output.get(cv2.CAP_PROP_FRAME_COUNT))
        output.release()
        assert frame_count == 300
        assert frame_count / fps == 20.0

    assert runs[0] == runs[1]
