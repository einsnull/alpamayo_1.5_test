#!/bin/bash
# Alpamayo Docker - Start Container Script
# Starts the Alpamayo container in detached mode

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for .hf_token
HF_TOKEN_FILE="$SCRIPT_DIR/.hf_token"
if [ -f "$HF_TOKEN_FILE" ]; then
    echo "Found .hf_token file"
else
    echo "Warning: .hf_token file not found"
fi

# Start container in detached mode
echo "Starting Alpamayo container..."
docker compose run -d --rm alpamayo bash -c "sleep infinity"

echo ""
echo "Container started!"
echo ""
echo "To enter the container, run:"
echo "  ./scripts/docker_into.sh"
