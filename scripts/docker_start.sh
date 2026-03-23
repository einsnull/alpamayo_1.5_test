#!/bin/bash
# Alpamayo Docker - Start Container Script
# Starts the Alpamayo container in detached mode using docker run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
echo "PROJECT_DIR: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Check for .hf_token
HF_TOKEN_FILE="$PROJECT_DIR/.hf_token"
if [ -f "$HF_TOKEN_FILE" ]; then
    echo "Found .hf_token file"
else
    echo "Warning: .hf_token file not found"
fi

# Build image if needed
docker compose build alpamayo 2>/dev/null || true

# Stop existing container if running
echo "Stopping existing container (if any)..."
docker stop alpamayo1.5 2>/dev/null || true
docker rm alpamayo1.5 2>/dev/null || true


# Start container in detached mode using docker run -itd
echo "Starting Alpamayo container..."
docker run -itd --rm \
    --gpus all \
    --runtime nvidia \
    --shm-size 8gb \
    -v "$PROJECT_DIR/.hf_token:/workspace/alpamayo/.hf_token:ro" \
    -v "$PROJECT_DIR/src:/workspace/alpamayo/src:ro" \
    -v "$PROJECT_DIR/scripts:/workspace/alpamayo/scripts" \
    -v "$PROJECT_DIR/configs:/workspace/alpamayo/configs" \
    -v "$PROJECT_DIR/../huggingface:/root/.cache/huggingface" \
    -v "$PROJECT_DIR/alpamayo-data:/workspace/alpamayo/data" \
    -v "$PROJECT_DIR/outputs:/workspace/alpamayo/outputs" \
    -w /workspace/alpamayo \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH="/workspace/alpamayo/src:/workspace/alpamayo" \
    --name alpamayo1.5 \
    alpamayo1.5:latest \
    bash

echo ""
echo "Container started!"
echo ""
echo "To enter the container, run:"
echo "  ./scripts/docker_into.sh"
