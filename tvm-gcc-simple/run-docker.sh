#!/bin/bash
set -e

# Получаем ID пользователя и группы
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Обновляем .env файл с помощью sed
echo "Updating .env file with UID=$CURRENT_UID and GID=$CURRENT_GID"
sed -i "s/^USER_ID=.*/USER_ID=$CURRENT_UID/" .env
sed -i "s/^GROUP_ID=.*/GROUP_ID=$CURRENT_GID/" .env

# Получаем количество ядер процессора и устанавливаем JOBS
CPU_CORES=$(nproc)
sed -i "s/^JOBS=.*/JOBS=$CPU_CORES/" .env

# Проверяем наличие директории data
if [ ! -d "./data" ]; then
    mkdir -p ./data
    echo "Created data directory"
fi

# Проверяем, запущен ли контейнер
if [ "$(docker ps -q -f name=tvm-builder-gcc-simple)" ]; then
    echo "Container is already running"
else
    # Останавливаем контейнер, если он существует, но не запущен
    if [ "$(docker ps -aq -f name=tvm-builder-gcc-simple)" ]; then
        echo "Removing stopped container..."
        docker rm tvm-builder-gcc-simple
    fi
    
    echo "Starting container..."
    # Запускаем контейнер
    docker-compose up -d
fi

# Подключаемся к контейнеру
echo "Connecting to container..."
docker-compose exec tvm-builder-gcc-simple bash
