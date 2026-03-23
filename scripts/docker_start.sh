#!/bin/bash
# Alpamayo Docker - Start Container Script
# Starts the Alpamayo container in detached mode using docker run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for .hf_token
HF_TOKEN_FILE="$SCRIPT_DIR/.hf_token"
if [ -f "$HF_TOKEN_FILE" ]; then
    echo "Found .hf_token file"
else
    echo "Warning: .hf_token file not found"
fi

# Build image if needed
docker compose build alpamayo 2>/dev/null || true

# Start container in detached mode using docker run -itd
echo "Starting Alpamayo container..."
docker run -itd --rm \
    --gpus all \
    --runtime nvidia \
    --shm-size 8gb \
    -v "$SCRIPT_DIR/.hf_token:/workspace/.hf_token:ro" \
    -v "$SCRIPT_DIR/src:/workspace/alpamayo/src:ro" \
    -v "$SCRIPT_DIR/scripts:/workspace/alpamayo/scripts" \
    -v "$SCRIPT_DIR/configs:/workspace/alpamayo/configs" \
    -v /storage/alpamayo/huggingface:/root/.cache/huggingface \
    -v /storage/alpamayo/alpamayo-data:/workspace/alpamayo/data \
    -v "$SCRIPT_DIR/outputs:/workspace/alpamayo/outputs" \
    -w /workspace/alpamayo \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/workspace/alpamayo \
    --name alpamayo1.5 \
    alpamayo1.5:latest \
    bash -c "sleep infinity"

echo ""
echo "Container started!"
echo ""
echo "To enter the container, run:"
echo "  ./scripts/docker_into.sh"
