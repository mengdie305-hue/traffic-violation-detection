# Traffic Violation Detection System

基于视频监控和深度学习的交通违章检测系统，支持闯红灯、超速、逆行等违章行为的自动识别。

## 当前状态

> **Pipeline 已完成闭环验证** — Mock 模式全链路跑通（检测 → 追踪 → 违章判定 → 日志 → 可视化输出）。

| 模块 | 状态 | 说明 |
|------|------|------|
| YOLO 检测器 | ✅ 封装完成 | 支持 MPS/CUDA/CPU，`src/detection/yolo_detector.py` |
| DeepSORT 追踪 | ✅ 已接入 | 真实模式使用 `deep-sort-realtime`；Mock 使用轻量匹配器 |
| 交通灯识别 | ✅ 可用 | HSV 颜色空间分割，红/黄/绿 |
| 闯红灯检测 | ✅ 可用 | 停止线越界 + 红灯持续时长判定 |
| 超速检测 | ✅ 可用 | 轨迹位移 → 真实世界速度换算 |
| 逆行检测 | ✅ 可用 | 运动方向 vs 车道规定方向偏差 |
| 车牌识别 | ⚠️ 已接入待实测 | 真实模式可配置 PaddleOCR；失败明确记录 `UNKNOWN` |
| 日志系统 | ✅ 可用 | CSV / JSON 双格式 |
| Mock 模式 | ✅ 可用 | `--mock` 颜色检测，无需 GPU |
| 训练脚本 | ✅ 代码完成 | `server/train_yolo.py`，需标注数据 |
| 真实数据集 | ❌ 待准备 | 需下载或自行录制 |

## 项目结构

```
traffic/
├── src/                         # 源代码
│   ├── detection/               # 目标检测模块
│   │   ├── yolo_detector.py     # YOLOv8 检测器封装
│   │   └── traffic_light.py     # 交通灯颜色识别 (HSV)
│   ├── tracking/                # 目标追踪模块
│   │   ├── deepsort_tracker.py  # Mock 使用的轻量追踪器
│   │   └── realtime_deepsort_tracker.py # 完整 DeepSORT 适配器
│   ├── recognition/             # 车牌识别模块
│   │   └── plate_recognizer.py  # PaddleOCR / Mock 后端
│   ├── violation/               # 违章判定模块
│   │   ├── red_light.py         # 闯红灯检测
│   │   ├── speeding.py          # 超速检测
│   │   └── wrong_way.py         # 逆行检测
│   ├── utils/                   # 工具
│   │   ├── video_utils.py       # 视频读写 & ROI
│   │   ├── draw_utils.py        # 可视化绘制
│   │   └── log_utils.py         # 违章日志 (CSV/JSON)
│   └── main.py                  # 主 Pipeline 入口
├── configs/
│   └── config.yaml              # 配置文件
├── server/                      # 服务器端
│   ├── train_yolo.py            # YOLO 训练脚本
│   └── evaluate.py              # 模型评估
├── scripts/
│   ├── generate_test_video.py   # 生成模拟路口测试视频
│   └── prepare_data.py          # 视频抽帧 & 数据集准备
├── tests/
│   └── test_detection.py        # 检测模块单元测试
└── data/
    ├── videos/                  # 测试视频
    ├── models/                  # 模型权重
    └── logs/                    # 违章日志输出
```

## 核心流程

```
视频输入 → 帧提取 → 检测(车辆) → DeepSORT追踪 → 交通灯识别
                                                    ↓
                       日志保存 ← 车牌识别 ← 违章判定(闯红灯/超速/逆行)
                                                    ↓
                                               可视化输出
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成并运行 Mock 演示（无需 GUI 或 GPU）
python scripts/generate_test_video.py
python src/main.py --mock --video data/videos/test_intersection.mp4 \
  --output data/videos/test_intersection_output.mp4

# 3. 真实 YOLO 推理 — 需要真实路口视频
python src/main.py --video data/videos/real_intersection.mp4 --config configs/config.yaml

# 4. 可选：加 --show 实时显示，按 q 退出
python src/main.py --mock --video data/videos/test_intersection.mp4 --show
```

### Google Colab（免费 T4）

无需本地安装 CUDA。打开
[`notebooks/Traffic_Violation_Colab.ipynb`](notebooks/Traffic_Violation_Colab.ipynb)，
在 Colab 中选择 `T4 GPU` 后按顺序运行。Notebook 会完成环境检查、依赖安装、
自动化测试、Mock 闭环验证、真实视频上传、ROI 配置、YOLO + DeepSORT 推理和
结果下载。

Colab 当前 Python 3.13 环境先使用 `requirements-colab.txt`，不安装
PaddlePaddle/PaddleOCR；第一轮真实视频验证中的车牌结果会明确记录为
`UNKNOWN`。OCR 将在兼容的独立环境中验证，不影响车辆检测、追踪和闯红灯
状态机测试。

## 两种运行模式

| 模式 | 命令 | 检测方式 | 适用场景 |
|------|------|---------|---------|
| Mock | `--mock` | 颜色阈值 + 轮廓 | 开发调试、无 GPU、模拟视频 |
| YOLO | 默认 | YOLOv8 推理 | 真实路口视频、GPU/MPS 加速 |

真实模式默认配置为 YOLOv8n + `deep-sort-realtime` + PaddleOCR。Mock
模式会使用轻量追踪器和模拟 OCR，以保证测试结果确定且不下载模型。

## 新增违章条例

模块化设计，加新条例只需 3 步，训练和规则互不影响：

```python
# 1. 新建 src/violation/illegal_parking.py
class IllegalParkingDetector:
    def check_violation(self, track_id, bbox, ...):
        ...

# 2. 注册到 src/violation/__init__.py
from .illegal_parking import IllegalParkingDetector

# 3. 在 main.py Pipeline 循环中调用
violation = parking_detector.check_violation(...)
```

## 本地 vs 服务器 分工

| 阶段 | 本地 (Mac) | 服务器 (GPU) |
|------|-----------|-------------|
| 数据准备 | ✅ 视频抽帧、标注 | - |
| 违章逻辑开发 | ✅ 编写 & 调试 | - |
| 模型推理测试 | ✅ MPS/CPU 轻量推理 | ✅ 大批量推理 |
| 模型训练 | ❌ 不推荐 | ✅ YOLO 微调 |
| 可视化 | ✅ 本地预览 | - |

## 技术栈

- **YOLOv8** — 车辆 + 交通灯目标检测
- **DeepSORT** — 真实模式使用运动预测、关联和外观特征；Mock 模式使用轻量确定性匹配
- **OpenCV** — 图像处理、视频读写、颜色分割
- **PaddleOCR** — 车牌号码识别（可选）
- **PyTorch** — 深度学习框架
