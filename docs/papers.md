# 交通违章检测相关论文推荐

按项目模块组织，每条标注对应代码位置。

---

## 1. 目标检测（车辆检测）→ [src/detection/yolo_detector.py](../src/detection/yolo_detector.py)

| 论文 | 年份 | 用一句话说 |
|------|------|-----------|
| [YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934) — Bochkovskiy et al. | 2020 | Bag of freebies + Bag of specials，实时检测标杆 |
| [YOLOv8 官方文档](https://docs.ultralytics.com/modes/train/) — Ultralytics | 2023 | 当前项目用的就是这个，没有正式论文，看文档就够了 |
| [DETR: End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) — Carion et al. | 2020 | Transformer 进检测领域，后续升级方向参考 |
| [RT-DETR: DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) — Zhao et al. | 2023 | 百度出品，实时 Transformer 检测器，比 YOLOv8 同期更快更准 |

> **直接建议**：把 [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/) 看完就够了，YOLO 系列工程文档比论文好用。

---

## 2. 多目标追踪 → [src/tracking/deepsort_tracker.py](../src/tracking/deepsort_tracker.py)

当前实现：简化版 IOU + 中心距离匹配，无 Kalman/Hungarian/ReID。

| 论文 | 年份 | 用一句话说 |
|------|------|-----------|
| [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763) — Bewley et al. | 2016 | 追踪领域基石，Kalman Filter + Hungarian 算法 |
| [DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402) — Wojke et al. | 2017 | SORT + ReID 外观特征，减少 ID Switch |
| [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864) — Zhang et al. | 2022 | **推荐升级目标**：利用低置信度框，MOTA 大幅提升，不需要额外 ReID 模型 |
| [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651) — Aharon et al. | 2022 | 改进 Kalman 状态向量 + 相机运动补偿 |
| [StrongSORT: Make DeepSORT Great Again](https://arxiv.org/abs/2202.13514) — Du et al. | 2022 | DeepSORT 现代化翻新，换组件到当前 SOTA |

> **直接建议**：如果要升级追踪模块，优先看 **ByteTrack**（[GitHub](https://github.com/ifzhang/ByteTrack)），代码质量好、工业界最常用、不需要额外 ReID 模型。

---

## 3. 交通灯识别 → [src/detection/traffic_light.py](../src/detection/traffic_light.py)

当前实现：HSV 颜色空间分割（两个红色范围 + 黄 + 绿），加亮度过滤 + 形态学去噪。

| 论文 | 年份 | 用一句话说 |
|------|------|-----------|
| [Traffic Light Recognition Using Deep Learning](https://ieeexplore.ieee.org/document/7789593) — Jensen et al. | 2016 | 早期综述，YOLO + 颜色分类流程 |
| [A Deep Learning Approach for Traffic Light Detection](https://ieeexplore.ieee.org/document/8575343) — Pon et al. | 2018 | Faster R-CNN + 时序信息融合，精度更高 |
| [Traffic Light Detection and Recognition for Autonomous Vehicles](https://link.springer.com/article/10.1007/s00521-015-2050-5) — John et al. | 2015 | 经典两阶段：检测框 → 颜色分类 |
| [Temporal Traffic Lights Detection and Recognition](https://ieeexplore.ieee.org/document/8814160) — Ouyang et al. | 2019 | 引入时序信息，减少单帧误判 |

> **直接建议**：当前 HSV 方法在小场景够用。要提升鲁棒性，用 YOLO 检测灯的位置 + 轻量 CNN（MobileNet）做颜色分类，比纯 HSV 抗光照变化。

---

## 4. 交通违章检测 — 与你项目最相关

| 论文 | 年份 | 覆盖 | 用一句话说 |
|------|------|------|-----------|
| [Automated Traffic Violation Detection Using Deep Learning](https://ieeexplore.ieee.org/document/9395212) — Rana et al. | 2021 | 闯红灯 + 超速 | **最接近你的架构**，YOLO + 轨迹分析 |
| [Real-time Red Light Violation Detection Using Computer Vision](https://ieeexplore.ieee.org/document/6866546) — Mammeri et al. | 2014 | 闯红灯 | 早期经典：帧差法 + 停止线检测 |
| [Intelligent Traffic Monitoring Using YOLO and DeepSORT](https://ieeexplore.ieee.org/document/9966582) — Khan et al. | 2022 | 闯红灯 + 超速 + 逆行 | 柬埔寨场景，YOLOv5 + DeepSORT，和你的方案最像 |
| [Traffic Violation Detection Using Video Analytics](https://ieeexplore.ieee.org/document/9102444) — Babu et al. | 2020 | 多类型 | 可扩展违章框架设计 |
| [Automated Traffic Rule Violation Detection](https://ieeexplore.ieee.org/document/9257346) — Vijayvergiya et al. | 2020 | 闯红灯 + 超速 + 不戴头盔 | 印度场景 |
| [Detection of Traffic Rule Violations Using Deep Learning](https://link.springer.com/article/10.1007/s00521-021-06813-1) — Khan et al. | 2022 | 闯红灯 + 逆行 + 违停 | 框架设计思路可参考 |

> **直接建议**：先看 **Rana (2021)** 和 **Khan (2022)** 这两篇，和你的 pipeline 最接近，代码思路可以直接对号入座。

---

## 5. 车牌识别 → [src/recognition/plate_recognizer.py](../src/recognition/plate_recognizer.py)

当前实现：Mock 后端（边缘检测假识别），PaddleOCR 后端已封装未启用。

| 论文 | 年份 | 用一句话说 |
|------|------|-----------|
| [Towards End-to-End License Plate Detection and Recognition](https://arxiv.org/abs/1804.06021) — Li et al. | 2019 | 端到端 CNN，不用分割字符 |
| [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447) — Zherzdev et al. | 2018 | 轻量级，适合嵌入式部署 |
| [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR) — 百度 | 持续更新 | 中文字符识别效果最好，你的项目已集成 |

> **直接建议**：直接启用 PaddleOCR 后端就好，中文车牌识别不用自己造轮子。

---

## 6. 智能交通系统综述（全景视角）

| 论文 | 年份 | 用一句话说 |
|------|------|-----------|
| [Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art](https://arxiv.org/abs/1704.05519) — Janai et al. | 2020 | 顶级综述，200+ 参考文献，摸清全局必读 |
| [Deep Learning for Intelligent Transportation Systems: A Survey](https://ieeexplore.ieee.org/document/8558515) — Nguyen et al. | 2018 | ITS 各子任务全覆盖 |
| [Vision-based Traffic Violation Detection: A Survey](https://link.springer.com/article/10.1007/s10462-022-10312-3) — 近 2 年 | 违章检测专用综述 |
| [Deep Learning in Traffic Management: A Survey](https://ieeexplore.ieee.org/document/9535286) — 2021 | 交通管理全场景 DL 应用 |
| [Object Detection in 20 Years: A Survey](https://arxiv.org/abs/1905.05055) — Zou et al. | 2019 | 目标检测 20 年综述，打基础用 |

---

## 7. 数据集相关

| 数据集 | 规模 | 场景 | 说明 |
|--------|------|------|------|
| [UA-DETRAC](https://detrac-db.rit.albany.edu/) | 10 小时，14 万帧 | 北京/天津道路 | 车辆检测+追踪，标注全 |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 5000 张精标注 | 德国城市街景 | 像素级标注，质量极高 |
| [BDD100K](https://bdd-data.berkeley.edu/) | 10 万张 | 美国多样场景 | 含天气/时间/场景标签 |
| [Aerial-Cars](https://www.kaggle.com/datasets/andrewmvd/aerial-cars-dataset) | Kaggle | 交通路口俯拍 | 路口视角接近你的场景 |

---

## 阅读优先级建议

```
Week 1-2:  Rana (2021) + Khan (2022)        → 搞懂别人怎么做的
Week 3:    ByteTrack (Zhang 2022)           → 追踪模块升级目标
Week 4:    Janai (2020) 综述                 → 全局视野
用到再看:  交通灯 / 车牌 / YOLO 文档          → 按需查阅
```

---

> **中文资料补充**：知网搜"基于深度学习的交通违章检测"，有不少硕士论文，比英文论文更接地气，代码和数据集通常也会公开。
