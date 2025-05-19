#!/bin/bash
# y00-install_system_deps.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

echo "=== Установка системных зависимостей для работы с OpenCV и PyTorch ==="

# Установка системных зависимостей
sudo apt-get update && sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    wget \
    unzip

echo "=== Системные зависимости установлены ==="
