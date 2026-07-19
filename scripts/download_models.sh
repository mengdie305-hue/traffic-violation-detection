#!/bin/bash
# ============================================================
# 下载预训练模型权重
# ============================================================
set -e

MODEL_DIR="data/models"
mkdir -p "$MODEL_DIR"

echo "========================================"
echo "  Downloading Pre-trained Models"
echo "========================================"

# YOLOv8n (轻量级，适合本地测试)
echo ""
echo "[1/3] YOLOv8n — Vehicle Detection"
if [ -f "$MODEL_DIR/yolov8n.pt" ]; then
    echo "  ✓ Already exists: $MODEL_DIR/yolov8n.pt"
else
    wget -q --show-progress \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt \
        -O "$MODEL_DIR/yolov8n.pt"
    echo "  ✓ Downloaded: $MODEL_DIR/yolov8n.pt"
fi

# YOLOv8s (更精确，推荐服务器训练用)
echo ""
echo "[2/3] YOLOv8s — Vehicle Detection (More Accurate)"
if [ -f "$MODEL_DIR/yolov8s.pt" ]; then
    echo "  ✓ Already exists: $MODEL_DIR/yolov8s.pt"
else
    wget -q --show-progress \
        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt \
        -O "$MODEL_DIR/yolov8s.pt"
    echo "  ✓ Downloaded: $MODEL_DIR/yolov8s.pt"
fi

# DeepSORT 特征提取器（可选）
echo ""
echo "[3/3] DeepSORT — Feature Extractor (OSNet)"
if [ -f "$MODEL_DIR/osnet_x0_25_msmt17.pt" ]; then
    echo "  ✓ Already exists: $MODEL_DIR/osnet_x0_25_msmt17.pt"
else
    echo "  ⚠ DeepSORT ReID model needs manual download."
    echo "    See: https://github.com/KaiyangZhou/deep-person-reid"
fi

echo ""
echo "========================================"
echo "  Done — Models saved to $MODEL_DIR/"
echo "========================================"
ls -lh "$MODEL_DIR"/*.pt 2>/dev/null || echo "  (no .pt files yet)"
