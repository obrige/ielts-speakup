"""
音频层指标 + 发音评分
- librosa: F0 / RMS / VAD / 语速 / 停顿 / 节奏
- wav2vec2: 音素 CTC confidence + 识别-期望音素编辑距离
- phonemizer: g2p 转期望音素

所有指标都是 raw metrics,由 LLM 后续对照 rubric 判断 band 分。
"""
from __future__ import annotations
import threading
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger("audio-metrics")

# 延迟加载 wav2vec2 (首次调用时加载, ~360MB)
_WAV2VEC2 = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()


def _ensure_wav2vec2():
    global _WAV2VEC2, _PROCESSOR
    if _WAV2VEC2 is not None:
        return
    with _LOAD_LOCK:
        if _WAV2VEC2 is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        model_id = "facebook/wav2vec2-base-960h"
        _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_id)
        _WAV2VEC2 = Wav2Vec2ForCTC.from_pretrained(model_id)
        _WAV2VEC2.eval()
        if torch.cuda.is_available():
            _WAV2VEC2 = _WAV2VEC2.cuda()
        logger.info("wav2vec2-base-960h loaded")


def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """统一 16kHz mono, librosa 加载"""
    import librosa
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"无法加载音频文件 {path}: {e}") from e
    if len(y) == 0:
        raise RuntimeError(f"音频文件为空: {path}")
    return y, sr


# ===== DSP 层 =====

def compute_dsp_metrics(
    y: np.ndarray,
    sr: int,
    word_times: Sequence[dict],
    duration_sec: float,
) -> dict:
    """
    返回:
    - f0_std:  基频标准差
    - rms_var: 短时能量方差
    - rhythm_cv: 词间隔变异系数
    - speech_rate_wpm: 每分钟词数
    - pause_count: >0.5s 静音段数
    - pause_total: 静音总时长
    - filler_count: 填充词数(由 word_times + 文本统计)
    """
    import librosa

    # 语速
    word_count = len(word_times)
    speech_rate_wpm = word_count / (duration_sec / 60) if duration_sec > 0 else 0.0

    # 词间隔变异系数(节奏)
    if len(word_times) > 1:
        intervals = np.diff([w["end"] for w in word_times])
        rhythm_cv = float(intervals.std() / (intervals.mean() + 1e-6))
    else:
        rhythm_cv = 1.0  # 不可计算,置中

    # F0 (基频, 反映语调)
    f0, _, _ = librosa.pyin(
        y, sr=sr, fmin=75, fmax=400,
        frame_length=1024, hop_length=256,
    )
    f0_clean = f0[~np.isnan(f0)]
    f0_std = float(f0_clean.std()) if len(f0_clean) > 10 else 0.0

    # RMS (能量, 反映重音)
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    rms_var = float(rms.std() / (rms.mean() + 1e-6))

    # VAD (语音活动检测, 反映停顿)
    intervals = librosa.effects.split(y, top_db=30)  # 静音段
    voiced_dur = sum((end - start) / sr for start, end in intervals)
    silence_dur = max(0.0, duration_sec - voiced_dur)
    pause_count = max(0, len(intervals) - 1)  # 段间停顿数

    return {
        "f0_std": round(f0_std, 2),
        "rms_var": round(rms_var, 3),
        "rhythm_cv": round(rhythm_cv, 3),
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "pause_count": pause_count,
        "pause_total": round(silence_dur, 2),
        "duration_sec": round(duration_sec, 2),
        "word_count": word_count,
    }


# ===== 发音层 =====

def compute_pronunciation_metrics(
    y: np.ndarray,
    sr: int,
    text: str,
) -> dict:
    """
    - wav2vec2_confidence: 整段 CTC 平均置信度(Intelligibility 锚点)
    - phoneme_edit_distance: 期望音素 vs 识别音素的归一化编辑距离(Clarity 锚点)
    """
    _ensure_wav2vec2()
    import torch
    from phonemizer import phonemize as _phonemize_fn

    # wav2vec2 前向
    inputs = _PROCESSOR(y, sampling_rate=sr, return_tensors="pt")
    if _WAV2VEC2.device.type != "cpu":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        logits = _WAV2VEC2(**inputs).logits

    probs = torch.softmax(logits, dim=-1)
    frame_conf = probs.max(dim=-1).values
    if frame_conf.dim() > 1:
        frame_conf = frame_conf.squeeze()
    wav2vec2_confidence = float(frame_conf.mean().cpu().numpy())

    # 音素编辑距离
    pred_ids = logits.argmax(dim=-1)
    if pred_ids.dim() > 1:
        pred_ids = pred_ids.squeeze()
    recognized = _PROCESSOR.batch_decode([pred_ids.cpu().tolist()])[0]

    expected_phonemes = ""
    recognized_phonemes = ""
    phoneme_edit_distance = 0.0
    try:
        expected_phonemes = _phonemize_fn(
            text, language="en-us", backend="espeak", strip=True
        )
        recognized_phonemes = _phonemize_fn(
            recognized, language="en-us", backend="espeak", strip=True
        )
        import Levenshtein
        edit_dist = Levenshtein.distance(expected_phonemes, recognized_phonemes)
        phoneme_edit_distance = edit_dist / max(len(expected_phonemes), 1)
    except Exception as e:
        logger.warning(f"Phonemizer failed: {e}")

    return {
        "wav2vec2_confidence": round(wav2vec2_confidence, 3),
        "phoneme_edit_distance": round(phoneme_edit_distance, 3),
        "raw_recognized_text": recognized,
        "expected_phonemes": expected_phonemes[:500],
        "recognized_phonemes": recognized_phonemes[:500],
    }


def split_into_sentences(
    text: str, word_times: Sequence[dict], duration_sec: float
) -> list[dict]:
    """
    按句末标点切句,绑定时间区间。
    返回 [{sent_id, text, start, end, duration, word_indices}, ...]
    """
    sentences = []
    current = {"text": "", "words": [], "start": None, "end": None}

    for idx, w in enumerate(word_times):
        if current["start"] is None:
            current["start"] = w["start"]
        current["words"].append(idx)
        current["text"] += w["word"] + " "
        current["end"] = w["end"]

        if w["word"].endswith((".", "!", "?")) and len(current["words"]) >= 3:
            sentences.append({
                "sent_id": len(sentences) + 1,
                "text": current["text"].strip(),
                "start": current["start"],
                "end": current["end"],
                "duration": current["end"] - current["start"],
                "word_indices": current["words"],
            })
            current = {"text": "", "words": [], "start": None, "end": None}

    # 处理残余
    if current["words"]:
        sentences.append({
            "sent_id": len(sentences) + 1,
            "text": current["text"].strip(),
            "start": current["start"] or 0.0,
            "end": current["end"] or duration_sec,
            "duration": (current["end"] or duration_sec) - (current["start"] or 0.0),
            "word_indices": current["words"],
        })

    return sentences


FILLER_WORDS = {"um", "uh", "er", "ah", "like", "you know", "i mean", "actually"}


def count_fillers(text: str) -> list[str]:
    """检测填充词, 简单大小写不敏感匹配"""
    import re
    text_lower = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = text_lower.split()
    found = []
    for i, tok in enumerate(tokens):
        if tok in FILLER_WORDS:
            found.append(tok)
        # 处理 "you know" / "i mean" 多词组合
        if i + 1 < len(tokens):
            pair = f"{tok} {tokens[i+1]}"
            if pair in FILLER_WORDS:
                found.append(pair)
    return found