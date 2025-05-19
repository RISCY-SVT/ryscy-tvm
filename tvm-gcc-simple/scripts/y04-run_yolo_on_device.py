#!/usr/bin/env python3
# y04-run_yolo_on_device.py
# Скрипт для запуска на устройстве LicheePi4A

import numpy as np
import tvm
from tvm.contrib import graph_executor
import cv2
import time
import os
import sys

# Пути к файлам модели (для запуска на устройстве)
model_dir = os.path.expanduser("~/TVM/models")
model_name = "yolov5n_riscv"
lib_path = os.path.join(model_dir, f"{model_name}.so")
json_path = os.path.join(model_dir, f"{model_name}.json")
params_path = os.path.join(model_dir, f"{model_name}.params")

# Проверка наличия файлов
for path in [lib_path, json_path, params_path]:
    if not os.path.exists(path):
        print(f"Ошибка: Файл не найден: {path}")
        print("Убедитесь, что файлы модели были корректно отправлены на устройство.")
        sys.exit(1)

# YOLOv5 классы (COCO)
classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
           'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
           'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
           'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def preprocess_image(img_path, input_shape=(640, 640)):
    """Предобработка изображения для YOLOv5"""
    print(f"Загрузка и предобработка изображения: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {img_path}")
    
    # Сохраняем оригинальный размер для постобработки
    original_shape = img.shape
    
    # Нормализация и изменение размера
    img = cv2.resize(img, input_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    print(f"Изображение предобработано. Форма входных данных: {img.shape}")
    return img, original_shape

def post_process(output, conf_threshold=0.25, iou_threshold=0.45):
    """Постобработка выходных данных YOLOv5"""
    print(f"Постобработка выходных данных формы {output.shape}")
    
    # output shape: (1, 25200, 85) - (batch, predictions, [x, y, w, h, conf, 80 class scores])
    predictions = output[0]
    
    # Фильтрация по порогу уверенности
    mask = predictions[:, 4] > conf_threshold
    predictions = predictions[mask]
    
    if len(predictions) == 0:
        return []
    
    # Выполнение non-maximum suppression (NMS)
    boxes = []
    for pred in predictions:
        box = pred[:4]  # [x, y, w, h] - центр, ширина, высота
        confidence = pred[4]
        class_id = np.argmax(pred[5:])
        class_score = pred[5 + class_id]
        
        # Преобразование [центр_x, центр_y, ширина, высота] в [x1, y1, x2, y2]
        x1 = float(box[0] - box[2] / 2)
        y1 = float(box[1] - box[3] / 2)
        x2 = float(box[0] + box[2] / 2)
        y2 = float(box[1] + box[3] / 2)
        
        boxes.append([x1, y1, x2, y2, confidence, class_id])
    
    # Простая реализация NMS
    boxes.sort(key=lambda x: x[4], reverse=True)
    result = []
    
    while len(boxes) > 0:
        result.append(boxes[0])
        remaining = []
        
        for box in boxes[1:]:
            # Рассчитываем IoU
            x1 = max(box[0], result[-1][0])
            y1 = max(box[1], result[-1][1])
            x2 = min(box[2], result[-1][2])
            y2 = min(box[3], result[-1][3])
            
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            
            intersection = w * h
            area1 = (box[2] - box[0]) * (box[3] - box[1])
            area2 = (result[-1][2] - result[-1][0]) * (result[-1][3] - result[-1][1])
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0
            
            if iou <= iou_threshold:
                remaining.append(box)
        
        boxes = remaining
    
    print(f"После постобработки найдено {len(result)} объектов")
    return result

def run_inference(img_path):
    """Запуск инференса на изображении"""
    # Загрузка модели
    print(f"Загрузка модели из {lib_path}")
    device = tvm.device("cpu", 0)
    
    try:
        lib = tvm.runtime.load_module(lib_path)
        print("Модуль загружен успешно")
    except Exception as e:
        print(f"Ошибка загрузки модуля: {str(e)}")
        sys.exit(1)
    
    try:
        with open(json_path, "r") as f:
            graph = f.read()
        print("JSON граф загружен успешно")
    except Exception as e:
        print(f"Ошибка загрузки JSON графа: {str(e)}")
        sys.exit(1)
    
    try:
        with open(params_path, "rb") as f:
            params = bytearray(f.read())
        print("Параметры модели загружены успешно")
    except Exception as e:
        print(f"Ошибка загрузки параметров: {str(e)}")
        sys.exit(1)
    
    # Создание Graph Runtime
    try:
        module = graph_executor.create(graph, lib, device)
        module.load_params(params)
        print("Graph Runtime создан успешно")
    except Exception as e:
        print(f"Ошибка создания Graph Runtime: {str(e)}")
        sys.exit(1)
    
    # Предобработка
    try:
        input_data, original_shape = preprocess_image(img_path)
    except Exception as e:
        print(f"Ошибка предобработки изображения: {str(e)}")
        sys.exit(1)
    
    # Задаем входные данные
    try:
        module.set_input("images", input_data)
    except Exception as e:
        print(f"Ошибка установки входных данных: {str(e)}")
        sys.exit(1)
    
    # Выполняем инференс
    print("Запуск инференса...")
    start_time = time.time()
    
    try:
        module.run()
        inference_time = time.time() - start_time
        print(f"Время инференса: {inference_time:.3f} секунд")
    except Exception as e:
        print(f"Ошибка выполнения инференса: {str(e)}")
        sys.exit(1)
    
    # Получаем результат
    try:
        output = module.get_output(0).numpy()
        print(f"Получены выходные данные формы {output.shape}")
    except Exception as e:
        print(f"Ошибка получения выходных данных: {str(e)}")
        sys.exit(1)
    
    # Постобработка
    try:
        detections = post_process(output)
    except Exception as e:
        print(f"Ошибка постобработки: {str(e)}")
        sys.exit(1)
    
    # Отображение результатов
    img = cv2.imread(img_path)
    height, width = original_shape[:2]
    
    # Коэффициенты для масштабирования координат обратно
    scale_x, scale_y = width / 640, height / 640
    
    print(f"Найдено {len(detections)} объектов:")
    for i, detection in enumerate(detections):
        x1, y1, x2, y2, confidence, class_id = detection
        
        # Масштабирование координат
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)
        
        class_name = classes[int(class_id)]
        print(f"  {i+1}. {class_name}: {confidence:.2f} - Координаты: [{x1}, {y1}, {x2}, {y2}]")
        
        # Рисуем рамку и метку на изображении
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Сохраняем результат
    output_path = os.path.splitext(img_path)[0] + "_result.jpg"
    cv2.imwrite(output_path, img)
    print(f"Результат сохранен: {output_path}")
    
    # Вывод статистики
    print(f"\nСтатистика:")
    print(f"  Время инференса: {inference_time:.3f} секунд")
    print(f"  Количество найденных объектов: {len(detections)}")
    
    return detections, inference_time

# Основная функция
if __name__ == "__main__":
    print("=" * 50)
    print("YOLOv5n TVM Runtime для RISC-V (LicheePi4A)")
    print("=" * 50)
    
    # Определение пути к тестовому изображению
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Используется указанное изображение: {img_path}")
    else:
        # Используем тестовое изображение
        img_path = os.path.join(model_dir, "test.jpg")
        
        # Если нет тестового изображения, предлагаем скачать
        if not os.path.exists(img_path):
            print("Тестовое изображение не найдено!")
            print("Укажите путь к изображению в качестве аргумента:")
            print("python3 y04-run_yolo_on_device.py /путь/к/изображению.jpg")
            sys.exit(1)
    
    # Проверяем наличие файла
    if not os.path.exists(img_path):
        print(f"Ошибка: Изображение не найдено: {img_path}")
        sys.exit(1)
    
    # Запускаем инференс
    run_inference(img_path)
