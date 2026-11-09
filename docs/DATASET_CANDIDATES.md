# 固定机位真实视频候选集

筛选日期：2026-11-09

目标是为车辆检测、DeepSORT 追踪、交通灯识别和闯红灯事件评估准备彼此独立的
固定机位视频。已经用于 ROI 和阈值标定的 `video6.mp4` 仅作为开发样本，不进入最终测试集。

## 已下载并检查

来源均为 Tiago Silva 的 *Dataset for Traffic Evaluation*, v2，许可为 CC BY 4.0：
[DOI 10.5281/zenodo.14041920](https://doi.org/10.5281/zenodo.14041920)。原始视频只保存在
`data/raw/`，不提交到 Git；文件校验值和用途记录在
`data/manifests/candidate_videos.csv`。

### 1. `teste2_6.mp4` - 建议保留

- 1280x720，30 FPS，2869 帧，95.63 秒。
- 白天雪景，高位固定机位，车辆、停止线和多个信号灯均可见。
- 与已经标定的 `video6.mp4` 属于不同场景，适合验证天气和场景泛化。
- 开头有播放器控制条，制作工作副本时应裁掉前 1-2 秒；原始文件保持不变。
- 闯红灯正负例仍需逐帧人工复核，不能仅凭抽样帧判断。

### 2. `2024-11-04 12-00-31.mp4` - 有条件保留

- 1280x720，30 FPS，3219 帧，107.3 秒。
- 夜间固定路口，信号灯存在状态变化，适合测试低照度鲁棒性。
- 首尾出现浏览器/播放器界面，制作工作副本时应只保留中间干净的固定场景。
- 由于它是二次录制画面，适合作为开发和鲁棒性测试，不作为论文唯一的正式测试来源。

### 3. `video4.mp4` - 开发短片

- 1280x720，30 FPS，536 帧，17.87 秒。
- 白天固定路口，停止线和信号灯清晰。
- 与 `video3.mp4`、`video6.mp4` 为同一机位系列，不能把三者当成三个独立摄像机测试集。

### 4. `video3.mp4` - 备用短片

- 1280x720，30 FPS，251 帧，8.37 秒。
- 场景条件合格，但时长较短且与 `video4.mp4` 同机位，优先级低于前两段。

## 推荐下载到服务器的正式数据

### Multi-view Traffic Intersection Dataset (MTID)

- 官方介绍：[Aalborg University MTID](https://vap.aau.dk/mtid/)
- 下载：[Kaggle 数据集](https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset)
- 许可：CC BY 4.0；总大小约 3.4 GB。
- 包含同一路口的固定基础设施视角与无人机视角，并提供像素级交通参与者标注。
- 约有两段各 100 秒的标注序列；本项目优先使用 infrastructure 固定视角评估车辆检测和追踪。
- 它主要解决检测/追踪评估，闯红灯事件和交通灯状态仍需根据本项目格式补标。

### Bellevue Traffic Video Dataset

- 官方仓库：[City of Bellevue Traffic Video Dataset](https://github.com/City-of-Bellevue/TrafficVideoDataset)
- 约 101 小时、1280x720、30 Hz，来自五个真实路口的杆载固定交通摄像机。
- 官方说明允许用于研究，但不是标准 CC 许可；论文中应引用 City of Bellevue，原视频不重新分发。
- 数据量较大，建议在服务器按机位各下载一小时文件，再裁出 1-3 分钟的独立片段。
- 最适合作为从未参与 ROI、阈值或模型调参的最终测试场景。

## 建议的数据角色

```text
development:
  video6.mp4
  video3.mp4
  video4.mp4
  夜间 Zenodo 片段

validation:
  teste2_6.mp4
  MTID infrastructure 的一部分

final test:
  MTID 未参与调参的完整片段
  Bellevue 至少两个不同路口的短片
```

最终划分必须以原始视频或摄像机为单位，禁止把同一视频的相邻帧拆到不同集合。

## 下一步筛选标准

1. ~~为两个已下载长片生成不改变原始文件的干净工作副本。~~ 已完成，参数见下表。
2. 完整观看并记录信号灯状态区间、正常停车/通过事件和疑似闯红灯事件。
3. 给每个机位单独标定停止线、交通灯 ROI 和允许越线方向。
4. 将视频上传 CVAT，以 Video Interpolation 模式标车辆轨迹和关键事件。
5. 完成标注后才确定 train/validation/test，不依据模型预测结果挑选测试片段。

## 第一批标注工作集

| 视频 ID | 源帧区间（结束帧不含） | 空间裁剪（上/下） | 工作副本 | 用途 |
|---|---:|---:|---|---|
| `zenodo_night_001` | `[90, 3105)` | `32 / 40 px` | 3015 帧，100.5 秒，1280x648 | development |
| `zenodo_snow_001` | `[120, 2850)` | `24 / 0 px` | 2730 帧，91 秒，1280x696 | validation |

原始视频位于 `data/raw/`，工作副本位于 `data/processed/annotation_workset/`，二者均不提交 Git。
工作副本清除了录屏首尾的浏览器界面以及播放器标题/控制条。文件列表、裁剪参数和 SHA256
记录在 `data/manifests/annotation_workset.csv`；事件、轨迹、信号灯区间和机位标定统一使用
`data/annotations/video_annotation.schema.json`。

可使用以下命令重新生成夜间工作副本：

```bash
python scripts/prepare_annotation_video.py \
  --input data/raw/candidates/zenodo_14041920/2024-11-04_12-00-31.mp4 \
  --output data/processed/annotation_workset/zenodo_night_001.mp4 \
  --start 3 --end 103.5 --crop-top 32 --crop-bottom 40
```

## CVAT 预标注

`scripts/preannotate_cvat.py` 使用 COCO 预训练 YOLOv8n 检测车辆，再用完整
`deep-sort-realtime` 生成轨迹建议。输出 XML 只用于减少人工画框工作量，不能直接作为论文真值。

```bash
python scripts/preannotate_cvat.py \
  --video data/processed/annotation_workset/zenodo_night_001.mp4 \
  --output data/processed/preannotations/zenodo_night_001.xml \
  --frame-step 6 --device cpu
```

本轮本地输出如下：

- `zenodo_night_001.xml`：136 条候选轨迹、2286 个关键框。
- `zenodo_snow_001.xml`：168 条候选轨迹、3839 个关键框。

在 CVAT 中分别创建两个 **video** task，标签预先建立为 `car`、`motorcycle`、`bus`、
`truck`，上传对应 MP4 后使用 **Upload annotations → CVAT 1.1** 导入 XML。导入后必须：

1. 删除路边静止车辆、重复框和错误类别；
2. 合并 ID Switch 造成的断裂轨迹，补齐漏检车辆；
3. 标记每个方向对应的信号灯状态区间；
4. 只在轨迹和信号灯复核完成后填写 `events`，并由第二次人工复核确认。

由于预标注按每 6 帧推理一次，CVAT 会在关键框之间插值；车辆刚进入或离开画面时要特别检查
轨迹起止帧。最终导出的真值应回填到 `data/annotations/zenodo_*.json`，不要覆盖预标注 XML。
