@echo off
REM NN3D Visualizer - Docker Startup Script (Windows)

echo.
echo ========================================
echo  NN3D VISUALIZER - DOCKER SETUP
echo ========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [*] Building and starting containers...
echo.

docker-compose up --build -d

if errorlevel 1 (
    echo [ERROR] Failed to start containers.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  CONTAINERS STARTED SUCCESSFULLY
echo ========================================
echo.
echo  Frontend: http://localhost:3000
echo  Backend:  http://localhost:8001
echo  API Docs: http://localhost:8001/docs
echo.
echo  Run 'docker-compose logs -f' to view logs
echo  Run 'docker-compose down' to stop
echo ========================================
echo.

pause
