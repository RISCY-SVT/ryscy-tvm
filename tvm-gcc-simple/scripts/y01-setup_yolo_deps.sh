#!/bin/bash
# y01-setup_yolo_deps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"/../data/

# Создаем директорию для моделей
mkdir -p models

# Установка зависимостей для работы с YOLOv5 и ONNX (без CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy pillow matplotlib opencv-python requests

# Клонирование репозитория YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
