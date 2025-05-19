#!/usr/bin/env python3
# y02-export_yolo_to_onnx.py

import os
import sys
import torch

# Настраиваем рабочую директорию
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data"))
os.chdir(DATA_DIR)

# Путь к директории с данными
models_dir = os.path.join(DATA_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

print(f"Используется директория для данных: {DATA_DIR}")
print(f"Директория для моделей: {models_dir}")

# Убедимся, что torch использует CPU
if torch.cuda.is_available():
    print("ВНИМАНИЕ: CUDA обнаружена, но будет использоваться CPU, т.к. целевое устройство - RISC-V")
torch.set_default_tensor_type('torch.FloatTensor')

# Добавляем каталог с yolov5 в PYTHONPATH
yolo_dir = os.path.join(DATA_DIR, "yolov5")
sys.path.append(yolo_dir)

# Проверяем наличие репозитория YOLOv5
if not os.path.exists(yolo_dir):
    print(f"Ошибка: Репозиторий YOLOv5 не найден в {yolo_dir}")
    print("Сначала запустите скрипт y01-setup_yolo_deps.sh")
    sys.exit(1)

print("Загрузка модели YOLOv5n...")
# Загрузка модели YOLOv5n
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=True)

# Отключаем режим обучения и переводим на CPU
model.eval()
model = model.cpu()

# Экспорт в ONNX
input_shape = (1, 3, 640, 640)  # batch_size, channels, height, width
dummy_input = torch.randn(input_shape)
onnx_path = os.path.join(models_dir, "yolov5n.onnx")

print(f"Экспорт модели в ONNX с входной формой {input_shape}...")
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_path, 
                  input_names=['images'],
                  output_names=['output'],
                  dynamic_axes={'images': {0: 'batch_size'}, 
                                'output': {0: 'batch_size'}},
                  opset_version=12)

print(f"Модель YOLOv5n экспортирована в ONNX: {onnx_path}")

# Вывод информации о модели
print("Размеры входа:", input_shape)
print("Классы:")
for i, cls in enumerate(model.names):
    print(f"  {i}: {cls}")
