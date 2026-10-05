"""
IELTS Speaking 评分后端入口

POST /api/score → 上传 WAV → 完整 pipeline → 返回 JSON 报告
"""
import os
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import aiofiles

from scoring.pipeline import ScoringPipeline
from scoring.questions import QUESTION_BANK, get_question

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ielts-api")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/audio"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_AUDIO_SIZE_MB = int(os.getenv("MAX_AUDIO_SIZE_MB", "50"))
MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024

# Pipeline 单例(避免每次请求都加载模型)
pipeline: ScoringPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 pipeline,关闭时清理"""
    global pipeline
    pipeline = ScoringPipeline()
    await pipeline.init()
    logger.info("Pipeline initialized")
    yield
    # 关闭时清理(目前无需)


app = FastAPI(title="IELTS Speaking Scorer", version="0.1.0", lifespan=lifespan)

# CORS:开发环境全开;生产收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok", "pipeline_ready": pipeline is not None and pipeline.ready}


@app.get("/api/questions")
async def questions(part: int = 1):
    """简易题库;实际项目应替换为后端题库服务"""
    return {"questions": QUESTION_BANK.get(part, [])}


@app.post("/api/score")
async def score(
    audio: Annotated[UploadFile, File(description="WAV 录音,16kHz mono 推荐")],
    question_id: Annotated[str, Form()],
):
    """
    完整评分 pipeline
    输入: WAV 文件 + 题目 ID
    输出: 完整 JSON 报告(scores + sub_scores + sentence_feedback + zpd_examples + summary)
    """
    if not pipeline or not pipeline.ready:
        raise HTTPException(503, "Pipeline not ready")

    # 校验题目 ID(早期失败,避免浪费录音上传)
    try:
        get_question(question_id)
    except KeyError:
        raise HTTPException(404, f"Question not found: {question_id}")

    # 保存上传
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    if suffix.lower() not in {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}:
        raise HTTPException(400, f"Unsupported audio format: {suffix}")

    audio_id = uuid.uuid4().hex
    audio_path = UPLOAD_DIR / f"{audio_id}{suffix}"

    # 流式保存 + 大小限制检查
    written = 0
    try:
        async with aiofiles.open(audio_path, "wb") as f:
            while chunk := await audio.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_AUDIO_SIZE_BYTES:
                    raise HTTPException(
                        413,
                        f"录音文件超过 {MAX_AUDIO_SIZE_MB} MB 限制"
                    )
                await f.write(chunk)
    except HTTPException:
        # 清理部分写入的文件
        try:
            audio_path.unlink()
        except Exception:
            pass
        raise

    logger.info(f"Received audio {audio_id} ({audio_path.stat().st_size} bytes)")

    report = None
    try:
        report = await pipeline.score_full(audio_path, question_id)
        return report
    except ValueError as e:
        # 用户输入类错误(空录音、太短)→ 400 更合理
        logger.warning(f"Bad request: {e}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(500, f"Scoring failed: {e!s}")
    finally:
        # 任何路径都要清理临时文件
        try:
            audio_path.unlink()
        except Exception:
            pass


# 静态前端(开发模式)
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/report")
    async def report_page():
        return FileResponse(FRONTEND_DIR / "report.html")
else:
    logger.warning(f"Frontend directory not found: {FRONTEND_DIR}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)