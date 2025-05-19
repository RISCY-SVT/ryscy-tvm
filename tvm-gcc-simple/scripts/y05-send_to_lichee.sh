#!/bin/bash
# y05-send_to_lichee.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# Пути к файлам
DATA_DIR="${SCRIPT_DIR}/../data"
OUTPUT_DIR="${DATA_DIR}/models/tvm_compiled"
DEVICE_USER="sipeed"
DEVICE_IP="lichee-svt"  # Измените на IP адрес или hostname вашего устройства
DEVICE_DIR="/home/sipeed/TVM/models"

# Проверка наличия скомпилированных файлов
if [ ! -f "${OUTPUT_DIR}/yolov5n_riscv.so" ]; then
    echo "Ошибка: Скомпилированные файлы модели не найдены."
    echo "Сначала запустите скрипт y03-compile_yolo_with_tvm.py"
    exit 1
fi

# Создание директории на устройстве
echo "Создание директории на устройстве..."
ssh $DEVICE_USER@$DEVICE_IP "mkdir -p $DEVICE_DIR"

# Копирование файлов модели
echo "Копирование файлов модели на устройство..."
scp $OUTPUT_DIR/yolov5n_riscv.so $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/
scp $OUTPUT_DIR/yolov5n_riscv.json $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/
scp $OUTPUT_DIR/yolov5n_riscv.params $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/

# Копирование скрипта запуска
echo "Копирование скрипта запуска..."
scp "${SCRIPT_DIR}/y04-run_yolo_on_device.py" $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/

# Копирование тестового изображения, если есть
if [ -f "${DATA_DIR}/test.jpg" ]; then
    echo "Копирование тестового изображения..."
    scp "${DATA_DIR}/test.jpg" $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/
elif [ -f "${DATA_DIR}/models/test.jpg" ]; then
    echo "Копирование тестового изображения из директории моделей..."
    scp "${DATA_DIR}/models/test.jpg" $DEVICE_USER@$DEVICE_IP:$DEVICE_DIR/
fi

echo "Файлы успешно отправлены на устройство LicheePi4A"
echo "Для запуска модели выполните на устройстве:"
echo "  cd $DEVICE_DIR"
echo "  python3 y04-run_yolo_on_device.py"
