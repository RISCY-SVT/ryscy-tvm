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

# Собираем Docker образ
echo "Building Docker image..."
docker-compose build

echo ""
echo "Docker image built successfully"
echo "Run the container with ./run-docker.sh"
