#!/bin/bash
set -e

CONTAINER_NAME="pytorch-cont"
SOURCE_DIR="$(pwd)"
TARGET_DIR="/home/torch/tensorvis"

echo "[-] Deploying Panopticon to Docker Container: $CONTAINER_NAME..."

# 1. Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "[!] Container '$CONTAINER_NAME' not found or not running."
    echo "    Please start the container first."
    exit 1
fi

# 2. Create target directory
echo "[-] Creating target directory..."
docker exec "$CONTAINER_NAME" mkdir -p "$TARGET_DIR"

# 3. Copy files
echo "[-] Copying files..."
docker cp "$SOURCE_DIR/." "$CONTAINER_NAME:$TARGET_DIR"

# 4. Install dependencies
echo "[-] Installing dependencies inside container..."
docker exec "$CONTAINER_NAME" pip install --break-system-packages -r "$TARGET_DIR/requirements.txt"

echo "[+] Deployment Complete!"
echo "    To run the dashboard:"
echo "    docker exec -it $CONTAINER_NAME streamlit run $TARGET_DIR/app.py --server.port 9000 --server.address 0.0.0.0"
