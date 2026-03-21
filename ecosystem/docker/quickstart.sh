#!/bin/bash
# Quick-start script for Sounio Ecosystem Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "Sounio Ecosystem Docker Quick Start"
echo "=========================================="
echo ""

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running. Please start Docker."
    exit 1
fi

echo "✓ Docker and Docker Compose are available"
echo ""

# Create work directory if it doesn't exist
if [ ! -d "$SCRIPT_DIR/work" ]; then
    echo "Creating work directory..."
    mkdir -p "$SCRIPT_DIR/work"
fi

echo "Repository root: $REPO_ROOT"
echo "Docker directory: $SCRIPT_DIR"
echo ""

# Build and start container
echo "Building and starting sounio-lab container..."
echo ""
cd "$SCRIPT_DIR"
docker-compose up --build

echo ""
echo "=========================================="
echo "Container is running!"
echo "=========================================="
echo ""
echo "Open your browser and navigate to:"
echo "  http://localhost:8888"
echo ""
echo "Token: sounio"
echo ""
echo "To stop the container, press Ctrl+C or run:"
echo "  docker-compose down"
echo ""
