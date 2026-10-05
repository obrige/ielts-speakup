"""
WhisperX 调用封装(本地 in-process)
首次调用会下载模型 (~1.5GB),后续复用缓存。
"""
import asyncio
import threading
import logging
from pathlib import Path

logger = logging.getLogger("whisperx")

# 延迟加载 whisperx (首次调用)
_WHISPERX_MODEL = None
_ALIGN_MODEL = None
_ALIGN_META = None
_LOAD_LOCK = threading.Lock()


def _ensure_whisperx():
    global _WHISPERX_MODEL, _ALIGN_MODEL, _ALIGN_META
    if _WHISPERX_MODEL is not None:
        return

    with _LOAD_LOCK:
        # 双重检查:避免锁内重复加载
        if _WHISPERX_MODEL is not None:
            return

        import whisperx
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        logger.info(f"Loading WhisperX model on {device} ({compute_type})")
        _WHISPERX_MODEL = whisperx.load_model(
            "large-v3",
            device=device,
            compute_type=compute_type,
            language="en",
        )
        _ALIGN_MODEL, _ALIGN_META = whisperx.load_align_model(
            language_code="en",
            device=device,
        )
        logger.info("WhisperX loaded")


def transcribe_sync(audio_path: Path, language: str = "en") -> dict:
    """
    调用 WhisperX 做 ASR + 词级对齐
    返回结构:
    {
        "text": "...",
        "language": "en",
        "word_segments": [
            {"word": "...", "start": 0.1, "end": 0.5, "score": 0.95},
            ...
        ],
        "segments": [...],
    }
    """
    _ensure_whisperx()
    import whisperx

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    # 1. 加载音频
    audio = whisperx.load_audio(str(audio_path))

    # 2. 转写
    result = _WHISPERX_MODEL.transcribe(audio, batch_size=16, language=language)

    # 3. 词级对齐
    result = whisperx.align(
        result["segments"],
        _ALIGN_MODEL,
        _ALIGN_META,
        audio,
        device=str(_ALIGN_MODEL.device.type),
    )

    # 4. 标准化输出
    word_segments = []
    for seg in result.get("word_segments", []):
        word_segments.append({
            "word": seg.get("word", "").strip(),
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
            "score": float(seg.get("score", 1.0)),
        })

    text = " ".join(w["word"] for w in word_segments if w["word"])

    return {
        "text": text,
        "language": language,
        "word_segments": word_segments,
        "segments": result.get("segments", []),
    }


async def transcribe(audio_path: Path, language: str = "en") -> dict:
    """异步 wrapper - 实际调用在线程池里避免阻塞 event loop"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, transcribe_sync, audio_path, language)