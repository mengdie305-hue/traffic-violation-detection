# Google Colab T4 验证记录

验证日期：2026-11-08  
验证提交：`0ea56f1`（基础流程），`42fedd0`（H.264 预览修复）

## 真实视频候选数据

- 已选定 `video6.mp4` 作为第一条真实视频验证样本，文件大小 9,829,975 bytes，MD5 为 `caa146f7ba10f2081c4922f42d9a42f8`。
- 来源：Tiago Silva, *Dataset for Traffic Evaluation*, v2, Zenodo, DOI: [10.5281/zenodo.14041920](https://doi.org/10.5281/zenodo.14041920)。
- 许可：CC BY 4.0。论文、报告和演示中使用视频或实验结果时保留以上署名与 DOI。
- Colab notebook 已支持直接下载并校验该视频，无需从本地上传。
- 选材检查：`rlr_occasions.mp4` 是包含浏览器界面和多个场景切换的 YouTube 合集屏幕录制，不符合当前固定 ROI 假设；`video6.mp4` 是约 8 秒的单一固定机位片段，红灯、停止线和车辆均清晰可见，因此改用后者。
- 注意：选定样本不等于已经完成真实视频验证；仍需根据首帧标定停止线与交通灯 ROI，再运行 YOLO + DeepSORT 并核验事件日志。

## 环境

| 项目 | 实测结果 |
|---|---|
| 平台 | Google Colab 托管运行时 |
| GPU | NVIDIA Tesla T4 |
| PyTorch | 2.14.1+cu130 |
| OpenCV | 4.14.0 |
| Ultralytics | 8.4.207 |
| CUDA 可用 | 是 |

## 验证结果

- 成功从 GitHub 克隆 `main` 分支。
- 成功安装 `requirements-colab.txt` 中的推理与测试依赖。
- 成功下载并加载 YOLOv8n 权重（6,549,796 bytes）。
- 自动化测试执行成功，`pytest -q` 返回码为 0。
- Mock 视频生成和完整 Pipeline 执行成功，返回码为 0。
- Mock 日志输出 3 条 `red_light` 事件：
  - `track_id=5`，时间约 6.00 秒；
  - `track_id=6`，时间约 6.93 秒；
  - `track_id=17`，时间约 16.00 秒。
- 每条事件均包含红灯状态、车辆框、检测置信度、车辆类别和证据图路径。
- 关闭 OCR 时，车牌字段正确记录为 `UNKNOWN`，未生成模拟号码。

## 已发现并修复的问题

OpenCV 默认写出的 `mp4v` 视频在 Colab 的 Chrome 内嵌播放器中可能显示“无法播放媒体”。
Notebook 已增加 FFmpeg H.264 转码（`libx264`、`yuv420p`、`faststart`），用于浏览器预览和结果下载。

## 下一步

在 Colab 下载已选定的公开固定机位路口视频，依据 1280×720 预览图配置停止线和
交通灯 ROI，然后运行 YOLO + DeepSORT 真实推理。真实视频结果需要人工标注后再
计算车辆检测、追踪、交通灯识别和闯红灯事件指标。
