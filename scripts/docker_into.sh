#!/bin/bash
# Alpamayo Docker - Enter Container Script
# Enters a running Alpamayo container

set -e

CONTAINER_NAME="alpamayo1.5"

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Alpamayo container is not running!"
    echo ""
    echo "Start the container first:"
    echo "  ./scripts/docker_start.sh"
    exit 1
fi

echo "Entering Alpamayo container..."
docker exec -it $CONTAINER_NAME bash
