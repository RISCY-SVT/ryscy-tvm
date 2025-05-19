#!/bin/bash
# y06-build_yolo_for_lichee.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

set -e

echo "=== Начало процесса компиляции YOLOv5n для LicheePi4A ==="

# Шаг 1: Установка Python зависимостей (без системных графических библиотек)
echo "=== Шаг 1: Установка Python зависимостей ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime numpy pillow requests

# Шаг 2: Экспорт модели в ONNX (используя вариант без графических зависимостей)
echo "=== Шаг 2: Экспорт YOLOv5n в ONNX ==="
python3 y02a-export_yolo_to_onnx_from_yolov5n.pt.py

# Шаг 3: Компиляция с TVM
echo "=== Шаг 3: Компиляция ONNX модели с TVM для RISC-V ==="
python3 y03-compile_yolo_with_tvm.py

# Шаг 4: Отправка на устройство
echo "=== Шаг 4: Отправка файлов на устройство LicheePi4A ==="
bash y05-send_to_lichee.sh

echo "=== Процесс компиляции YOLOv5n для LicheePi4A завершен ==="
echo "Модель готова к использованию на LicheePi4A"
echo ""
echo "Для запуска на устройстве выполните:"
echo "cd ~/TVM/models"
echo "python3 y04-run_yolo_on_device.py"
