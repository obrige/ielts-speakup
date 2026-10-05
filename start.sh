#!/bin/bash
# ================================================
# IELTS Speaking Scorer — Linux/Mac 一键启动
# ================================================

set -e

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker not installed."
    exit 1
fi

# 创建 .env(如果不存在)
if [ ! -f "backend/.env" ]; then
    echo "[SETUP] Creating backend/.env from template..."
    cp backend/.env.example backend/.env
    echo ""
    echo "[IMPORTANT] Edit backend/.env and fill in your API keys:"
    echo "  DEEPSEEK_API_KEY=sk-your-key"
    echo "  DEEPSEEK_BASE_URL=https://api.deepseek.com/v1"
    echo ""
    read -p "Press Enter to open editor..."
    "${EDITOR:-vi}" backend/.env
fi

# 创建数据目录
mkdir -p data/audio

echo ""
echo "[BUILD] Building Docker image (first time takes 10-15 min)..."
docker-compose build

echo ""
echo "[START] Starting container..."
docker-compose up -d

echo ""
echo "[STATUS] Container status:"
docker-compose ps

echo ""
echo "================================================"
echo "  Access http://localhost:8080 in your browser"
echo "  Monitor: docker-compose logs -f"
echo "  Stop:    docker-compose down"
echo "================================================"