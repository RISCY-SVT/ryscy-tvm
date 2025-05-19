#!/usr/bin/env python3
# y02-export_yolo_to_onnx.py (альтернативная версия без графических зависимостей)

import os
import sys
import torch
import urllib.request
import zipfile
import shutil

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
torch.set_default_tensor_type('torch.FloatTensor')

# Скачиваем предварительно сохраненную модель YOLOv5n (weights only) напрямую
weights_url = "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt"
weights_path = os.path.join(models_dir, "yolov5n.pt")

# Проверим, скачана ли уже модель
if not os.path.exists(weights_path):
    print(f"Скачивание модели YOLOv5n из {weights_url}...")
    urllib.request.urlretrieve(weights_url, weights_path)
    print(f"Модель сохранена в {weights_path}")
else:
    print(f"Модель уже скачана: {weights_path}")

# Скачиваем исходный код YOLOv5, если его еще нет
yolo_src_dir = os.path.join(DATA_DIR, "yolov5_src")
if not os.path.exists(yolo_src_dir):
    print("Скачивание исходного кода YOLOv5...")
    zip_path = os.path.join(DATA_DIR, "yolov5.zip")
    urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/archive/refs/heads/master.zip", zip_path)
    
    # Распаковка архива
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Переименование распакованной папки
    extracted_dir = os.path.join(DATA_DIR, "yolov5-master")
    if os.path.exists(extracted_dir):
        shutil.move(extracted_dir, yolo_src_dir)
    
    # Удаление архива
    os.remove(zip_path)
    print(f"Исходный код YOLOv5 распакован в {yolo_src_dir}")
else:
    print(f"Исходный код YOLOv5 уже скачан: {yolo_src_dir}")

# Добавляем каталог YOLOv5 в PYTHONPATH
sys.path.append(yolo_src_dir)

print("Загрузка модели YOLOv5n...")
# Загрузка модели YOLOv5n из сохраненных весов
from models.experimental import attempt_load

# Загружаем модель
try:
    model = attempt_load(weights_path, device='cpu')
    model.eval()
    print("Модель YOLOv5n успешно загружена")
except Exception as e:
    print(f"Ошибка загрузки модели: {str(e)}")
    sys.exit(1)

# Экспорт в ONNX
input_shape = (1, 3, 640, 640)  # batch_size, channels, height, width
dummy_input = torch.randn(input_shape)
onnx_path = os.path.join(models_dir, "yolov5n.onnx")

print(f"Экспорт модели в ONNX с входной формой {input_shape}...")
try:
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_path, 
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {0: 'batch_size'}, 
                                    'output': {0: 'batch_size'}},
                      opset_version=12)
                      
    print(f"Модель YOLOv5n экспортирована в ONNX: {onnx_path}")
except Exception as e:
    print(f"Ошибка экспорта модели в ONNX: {str(e)}")
    sys.exit(1)

# Вывод информации о модели
print("Размеры входа:", input_shape)
print("Классы:", model.names)
