#!/bin/bash
# NN3D Visualizer - Docker Startup Script (Linux/Mac)

echo ""
echo "========================================"
echo " NN3D VISUALIZER - DOCKER SETUP"
echo "========================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running. Please start Docker."
    exit 1
fi

echo "[*] Building and starting containers..."
echo ""

docker-compose up --build -d

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start containers."
    exit 1
fi

echo ""
echo "========================================"
echo " CONTAINERS STARTED SUCCESSFULLY"
echo "========================================"
echo ""
echo " Frontend: http://localhost:3000"
echo " Backend:  http://localhost:8001"
echo " API Docs: http://localhost:8001/docs"
echo ""
echo " Run 'docker-compose logs -f' to view logs"
echo " Run 'docker-compose down' to stop"
echo "========================================"
echo ""
