#!/usr/bin/env python3
"""
生成模拟十字路口测试视频。

包含：道路、停止线、红绿灯、移动车辆、闯红灯场景。
用于验证 Pipeline 全流程，无需真实监控视频。
"""

import argparse

import cv2
import numpy as np

# ============================================================
# 画布参数
# ============================================================
W, H = 1280, 720
FPS = 30
DURATION = 20  # 秒
TOTAL_FRAMES = FPS * DURATION
DEFAULT_OUTPUT = "data/videos/test_intersection.mp4"

# ============================================================
# 路口参数
# ============================================================
STOP_LINE_Y = 320       # 停止线 y 坐标
LIGHT_POS = (640, 60)   # 红绿灯位置
LIGHT_RADIUS = 25

# 车辆参数
CAR_W, CAR_H = 80, 50
VEHICLE_COLORS = [
    (255, 0, 0),     # 蓝车
    (0, 255, 0),     # 绿车
    (0, 0, 255),     # 红车
    (0, 255, 255),   # 黄车
    (255, 0, 255),   # 紫车
    (128, 128, 128), # 灰车
]

# ============================================================
# 场景定义：(start_frame, lane_x, speed, color_index, runs_red_light, direction)
# ============================================================
SCENARIOS = [
    # 正常停车等待 (lane=400, 速度正常，会在停止线前停车)
    {"start": 0,   "lane": 300, "speed": 2.0, "color": 0, "red_run": False, "dir": "down"},
    {"start": 15,  "lane": 500, "speed": 2.0, "color": 2, "red_run": False, "dir": "down"},
    {"start": 30,  "lane": 700, "speed": 2.5, "color": 1, "red_run": False, "dir": "down"},
    {"start": 60,  "lane": 400, "speed": 2.0, "color": 3, "red_run": False, "dir": "down"},
    # ⚠ 闯红灯车 (红灯时越过停止线)
    {"start": 60,  "lane": 550, "speed": 2.5, "color": 4, "red_run": True,  "dir": "down"},
    {"start": 100, "lane": 600, "speed": 2.8, "color": 3, "red_run": True,  "dir": "down"},
    # 正常车
    {"start": 200, "lane": 350, "speed": 2.0, "color": 0, "red_run": False, "dir": "down"},
    {"start": 230, "lane": 450, "speed": 2.2, "color": 2, "red_run": False, "dir": "down"},
    # ⚠ 再一辆闯红灯
    {"start": 365, "lane": 500, "speed": 2.6, "color": 4, "red_run": True,  "dir": "down"},
    {"start": 300, "lane": 650, "speed": 2.0, "color": 1, "red_run": False, "dir": "down"},
]


def draw_road(frame):
    """绘制路面、车道线、停止线。"""
    # 路面
    cv2.rectangle(frame, (0, 0), (W, H), (80, 80, 80), -1)

    # 垂直道路
    cv2.rectangle(frame, (0, 0), (W, H), (60, 60, 60), -1)
    cv2.fillPoly(frame, [np.array([
        (280, 0), (1000, 0), (1000, H), (280, H)
    ])], (50, 50, 50))

    # 水平道路（横向）
    cv2.rectangle(frame, (0, 260), (W, 380), (50, 50, 50), -1)

    # 车道虚线
    for lx in [400, 640, 880]:
        for dy in range(0, H, 40):
            cv2.line(frame, (lx, dy), (lx, dy + 20), (200, 200, 200), 2)

    # 停止线
    cv2.line(frame, (280, STOP_LINE_Y), (1000, STOP_LINE_Y), (255, 255, 255), 3)

    # 人行横道
    for dx in range(280, 1000, 40):
        cv2.rectangle(frame, (dx, STOP_LINE_Y - 20), (dx + 20, STOP_LINE_Y), (200, 200, 200), -1)

    # 路口区域框
    cv2.rectangle(frame, (280, 260), (1000, 380), (200, 200, 200), 1)

    # 文字标注
    cv2.putText(frame, "STOP LINE", (290, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_traffic_light(frame, color):
    """绘制交通灯。"""
    colors = {
        "red":     (0, 0, 255),
        "yellow":  (0, 255, 255),
        "green":   (0, 255, 0),
    }
    x, y = LIGHT_POS
    r = LIGHT_RADIUS

    # 灯框
    cv2.rectangle(frame, (x - 35, y - 90), (x + 35, y + 90), (40, 40, 40), -1)
    cv2.rectangle(frame, (x - 35, y - 90), (x + 35, y + 90), (100, 100, 100), 2)

    # 三个灯
    for i, (state, c) in enumerate(colors.items()):
        ly = y - 55 + i * 55
        if state == color:
            cv2.circle(frame, (x, ly), r, c, -1)
            cv2.circle(frame, (x, ly), r, (255, 255, 255), 2)  # 高亮
        else:
            cv2.circle(frame, (x, ly), r, (80, 80, 80), -1)
            cv2.circle(frame, (x, ly), r, (60, 60, 60), 1)

    # 文字
    cv2.putText(frame, color.upper(), (x + 40, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(color, (255, 255, 255)), 2)


def draw_vehicle(frame, cx, cy, color, label=""):
    """绘制一辆车（俯视角俯视简化为矩形）。"""
    x1 = int(cx - CAR_W // 2)
    y1 = int(cy - CAR_H // 2)
    x2 = int(cx + CAR_W // 2)
    y2 = int(cy + CAR_H // 2)

    # 车身
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 车窗
    wx1, wy1 = x1 + 15, y1 + 5
    wx2, wy2 = x2 - 15, y1 + 20
    cv2.rectangle(frame, (wx1, wy1), (wx2, wy2), (200, 230, 255), -1)

    # 标签
    if label:
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def get_light_state(frame_idx):
    """根据帧号返回红绿灯状态。"""
    cycle = int(frame_idx // (FPS * 4))  # 每 4 秒切换
    states = ["green", "yellow", "red", "red"]  # 绿→黄→红→红→绿
    # 改一下周期让红灯更多
    mod = frame_idx % (FPS * 10)
    if mod < FPS * 4:
        return "green"
    elif mod < FPS * 5:
        return "yellow"
    elif mod < FPS * 9:
        return "red"
    else:
        return "green"


def generate_video(output_path: str = DEFAULT_OUTPUT):
    """生成确定性的 Mock 路口视频。"""
    from pathlib import Path

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, FPS, (W, H))

    vehicles = SCENARIOS

    for frame_idx in range(TOTAL_FRAMES):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_road(frame)

        light = get_light_state(frame_idx)
        draw_traffic_light(frame, light)

        for v in vehicles:
            elapsed = frame_idx - v["start"]
            if elapsed < 0:
                continue

            # 计算位置
            lane_x = v["lane"]
            cy = elapsed * v["speed"]
            cx = lane_x

            if v["dir"] == "down":
                pass  # cy 递增向下

            # 红灯逻辑：如果不应闯红灯，且当前是红灯，车在停止线前停下
            if not v["red_run"] and light == "red" and cy > STOP_LINE_Y - 40:
                cy = STOP_LINE_Y - 40  # 停在停止线前

            if not v["red_run"] and light == "yellow" and cy > STOP_LINE_Y - 20:
                cy = STOP_LINE_Y - 20  # 黄灯也停

            if cy > H + 100:
                continue  # 驶出画面

            color = VEHICLE_COLORS[v["color"]]
            label = f"ID:{v['color']}" + (" RUN!" if v["red_run"] and light == "red" and cy > STOP_LINE_Y else "")
            draw_vehicle(frame, cx, cy, color, label)

        # 帧号 & 时间戳
        timestamp = frame_idx / FPS
        cv2.putText(frame, f"Frame: {frame_idx}  Time: {timestamp:.1f}s",
                    (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"FPS: {FPS}", (W - 120, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(frame)

        if frame_idx % 100 == 0:
            print(f"  Generating frame {frame_idx}/{TOTAL_FRAMES}...")

    writer.release()
    print(f"\n✓ Test video saved to: {output}")
    print(f"  Duration: {DURATION}s, Frames: {TOTAL_FRAMES}, Size: {W}x{H}")
    print(f"  Scenarios: {len(SCENARIOS)} vehicles")
    print(f"  Red-light runners: {sum(1 for v in SCENARIOS if v['red_run'])}")


def main():
    parser = argparse.ArgumentParser(description="Generate the mock intersection video")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate_video(args.output)


if __name__ == "__main__":
    main()
