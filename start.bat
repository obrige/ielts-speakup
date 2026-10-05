@echo off
REM ================================================
REM  IELTS Speaking Scorer — Windows 一键启动
REM ================================================

setlocal

REM 检查 Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not installed. Install Docker Desktop first.
    pause
    exit /b 1
)

REM 创建 .env(如果不存在)
if not exist "backend\.env" (
    echo [SETUP] Creating backend\.env from template...
    copy /Y "backend\.env.example" "backend\.env"
    echo.
    echo [IMPORTANT] Edit backend\.env and fill in your API keys:
    echo   DEEPSEEK_API_KEY=sk-your-key
    echo   DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    echo.
    pause
    notepad "backend\.env"
)

REM 创建数据目录
if not exist "data\audio" mkdir "data\audio"

echo.
echo [BUILD] Building Docker image (first time takes 10-15 min)...
docker-compose build

echo.
echo [START] Starting container...
docker-compose up -d

echo.
echo [STATUS] Container status:
docker-compose ps

echo.
echo [LOGS] Last 20 lines:
docker-compose logs --tail 20

echo.
echo ================================================
echo  Access http://localhost:8080 in your browser
echo  Monitor with: docker-compose logs -f
echo  Stop with:    docker-compose down
echo ================================================
echo.

pause