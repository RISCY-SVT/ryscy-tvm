#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}"/.env

# Проверяем, запущен ли контейнер
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Container is already running"
else
    # Останавливаем контейнер, если он существует, но не запущен
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        echo "Removing stopped container..."
        docker rm ${CONTAINER_NAME}
    fi
    
    echo "Starting container..."
    # Запускаем контейнер
    docker-compose up -d
fi

# Подключаемся к контейнеру
echo "Connecting to container..."
docker-compose exec ${CONTAINER_NAME} bash
