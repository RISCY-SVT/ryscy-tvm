#!/bin/bash
set -eo pipefail

# Get current user and group IDs
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)
USER_NAME=$USER

# Update .env file
echo "Updating .env file with UID=$CURRENT_UID and GID=$CURRENT_GID"
sed -i "s/^USER_ID=.*/USER_ID=$CURRENT_UID/" .env
sed -i "s/^GROUP_ID=.*/GROUP_ID=$CURRENT_GID/" .env
sed -i "s/^USER_NAME=.*/USER_NAME=$USER_NAME/" .env

# Get CPU cores
CPU_CORES=$(nproc)
sed -i "s/^JOBS=.*/JOBS=$CPU_CORES/" .env

# Create data directory if needed
if [ ! -d "./data" ]; then
    mkdir -p ./data
    echo "Created data directory"
fi

# Build Docker image
echo "Building Docker image..."
docker builder prune -f
docker-compose build

echo ""
echo "Docker image built successfully"
echo "Run the container with ./run-docker.sh"
