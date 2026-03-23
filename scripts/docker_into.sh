#!/bin/bash
# Alpamayo Docker - Enter Container Script
# Enters a running Alpamayo container

set -e

CONTAINER_NAME=$(docker compose ps -q alpamayo 2>/dev/null)

if [ -z "$CONTAINER_NAME" ]; then
    echo "Error: Alpamayo container is not running!"
    echo ""
    echo "Start the container first:"
    echo "  ./scripts/docker_start.sh"
    exit 1
fi

echo "Entering Alpamayo container..."
docker exec -it $CONTAINER_NAME bash
