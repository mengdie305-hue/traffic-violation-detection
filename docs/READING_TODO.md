# Traffic Violation Detection — 论文阅读待办

阅读目标：为论文的 Related Work、Methodology、Dataset 和 Evaluation 章节建立可靠引用，并直接对应当前项目的实现模块。

## 第一阶段：核心必读

- [ ] **DeepSORT: Simple Online and Realtime Tracking with a Deep Association Metric**
  - 作者：Nicolai Wojke, Alex Bewley, Dietrich Paulus
  - 年份：2017
  - 模块：多目标追踪
  - 链接：https://arxiv.org/abs/1703.07402
  - 阅读重点：Kalman Filter、级联匹配、外观特征、ID Switch
  - 输出：写一段“为什么项目选择 DeepSORT”的方法说明

- [ ] **YOLOv4: Optimal Speed and Accuracy of Object Detection**
  - 作者：Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
  - 年份：2020
  - 模块：车辆与交通灯检测
  - 链接：https://arxiv.org/abs/2004.10934
  - 阅读重点：实时检测、数据增强、速度与精度权衡
  - 输出：整理 YOLO 系列适合实时交通监控的理由；项目实际模型注明为 YOLOv8n

- [ ] **Development of an Automated Red Light Violation Detection System (RLVDS) for Indian Vehicles**
  - 年份：2010
  - 模块：闯红灯事件判定
  - 链接：https://arxiv.org/abs/1003.6052
  - 阅读重点：固定监控视角、停止线、违章证据生成
  - 输出：与项目的 `BEFORE_LINE → CROSSING_LINE → AFTER_LINE` 状态机对比

- [ ] **A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification**
  - 作者：Karsten Behrendt, Libor Novak, Rami Botros
  - 年份：2017
  - 模块：交通灯检测与状态识别
  - 链接：https://hci.iwr.uni-heidelberg.de/node/6132
  - 阅读重点：小目标交通灯、时序跟踪、Bosch Small Traffic Lights Dataset
  - 输出：说明纯 HSV 方法的局限和深度学习升级方向

## 第二阶段：车牌与数据集

- [ ] **Towards End-to-End Car License Plates Detection and Recognition with Deep Neural Networks**
  - 作者：Hui Li, Peng Wang, Chunhua Shen
  - 年份：2017
  - 模块：车牌检测与识别
  - 链接：https://arxiv.org/abs/1709.08828
  - 阅读重点：车牌定位与字符识别联合训练、端到端误差
  - 输出：与当前“车辆下部裁剪 + PaddleOCR”方案对比

- [ ] **LPRNet: License Plate Recognition via Deep Neural Networks**
  - 作者：Sergey Zherzdev, Alexey Gruzdev
  - 年份：2018
  - 模块：轻量车牌字符识别
  - 链接：https://arxiv.org/abs/1806.10447
  - 阅读重点：无需字符分割、实时推理、中文车牌实验
  - 输出：记录整牌准确率、运行速度和部署硬件

- [ ] **BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning**
  - 作者：Fisher Yu et al.
  - 年份：2020
  - 模块：数据集与评估
  - 链接：https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_BDD100K_A_Diverse_Driving_Dataset_for_Heterogeneous_Multitask_Learning_CVPR_2020_paper.html
  - 阅读重点：数据划分、天气与场景多样性、检测和追踪标注
  - 输出：判断 BDD100K 是否适合项目训练，以及车载视角的域差异

## 第三阶段：扩展阅读

- [ ] **ByteTrack: Multi-Object Tracking by Associating Every Detection Box**
  - 作者：Yifu Zhang et al.
  - 年份：2022
  - 模块：追踪对比实验
  - 链接：https://arxiv.org/abs/2110.06864
  - 阅读重点：低置信度检测框关联、MOTA、IDF1、HOTA
  - 输出：设计 DeepSORT 与 ByteTrack 的对比实验

- [ ] **Computer Vision for Autonomous Vehicles: Problems, Datasets and State of the Art**
  - 作者：Joel Janai et al.
  - 年份：2020
  - 模块：综述与 Related Work
  - 链接：https://arxiv.org/abs/1704.05519
  - 阅读重点：检测、追踪、场景理解、常用数据集和评价指标
  - 输出：整理 Related Work 章节结构和后续引用

## 建议阅读顺序

1. DeepSORT
2. RLVDS 闯红灯检测
3. YOLOv4
4. Bosch 交通灯检测
5. 端到端车牌识别
6. LPRNet
7. BDD100K
8. ByteTrack
9. 自动驾驶视觉综述

## 单篇论文阅读记录模板

- [ ] 完成首次通读
- [ ] 摘录研究问题和主要贡献
- [ ] 记录使用的数据集及划分方式
- [ ] 记录模型、阈值和硬件配置
- [ ] 记录评价指标和主要结果
- [ ] 记录一个可用于本项目的设计
- [ ] 记录一个局限或失败场景
- [ ] 整理 BibTeX 引用
- [ ] 写 100–150 字论文笔记

## Notion 数据库字段建议

| 字段 | 类型 | 示例 |
|---|---|---|
| Paper | Title | DeepSORT |
| Status | Select | To Read / Reading / Done |
| Priority | Select | P0 / P1 / P2 |
| Module | Multi-select | Tracking、Red Light |
| Year | Number | 2017 |
| Link | URL | arXiv URL |
| Reading Output | Text | 写方法对比 |
| Notes | Text | 阅读笔记 |

