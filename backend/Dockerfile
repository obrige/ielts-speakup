FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 系统依赖:espeak-ng 给 phonemizer 用 + ffmpeg 给 WhisperX
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 requirements(锁定 torch 版本,CUDA 12.1 兼容)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再装 WhisperX(--no-deps 避免它覆盖 torch)
# whisperx 的 torch 依赖已经由 requirements 提供
RUN pip install --no-cache-dir --no-deps git+https://github.com/m-bain/whisperx.git@main

# whisperx 必需的非 torch 依赖
RUN pip install --no-cache-dir nltk dtw-python num2words

# 拷贝后端代码
COPY backend/ ./

# 拷贝前端(也内置到镜像,即使 volume 没挂也能用)
COPY frontend/ ./frontend/

# 确保上传目录存在(volume 未挂载时不报错)
RUN mkdir -p /tmp/audio

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]