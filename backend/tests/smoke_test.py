"""
Smoke test: 不依赖 LLM,验证算法层能跑通。

用法:
    cd backend
    python -m tests.smoke_test
"""
import asyncio
import sys
from pathlib import Path

# 让 scoring 包可被 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.audio_metrics import (
    load_audio,
    compute_dsp_metrics,
    compute_pronunciation_metrics,
    split_into_sentences,
    count_fillers,
)
from scoring.text_stats import compute_text_metrics


async def main():
    # 找一个测试音频(开发时可手放一个 WAV 到 tests/test.wav)
    audio_path = Path(__file__).parent / "test.wav"

    if not audio_path.exists():
        print(f"⚠️  No test audio at {audio_path}, generating synthetic test...")

        # 用 numpy 生成 5 秒正弦波模拟音频
        import numpy as np
        import soundfile as sf

        sr = 16000
        t = np.linspace(0, 5, sr * 5)
        # 模拟语音基频 + 一些静音段
        signal = np.zeros_like(t)
        # 4 个说"词"的段
        for start_sec in [0.5, 1.8, 2.5, 3.8]:
            s = int(start_sec * sr)
            e = int((start_sec + 0.4) * sr)
            tt = np.linspace(0, 0.4, e - s)
            signal[s:e] = 0.3 * np.sin(2 * np.pi * 200 * tt)

        sf.write(audio_path, signal, sr)
        print(f"Generated synthetic test audio at {audio_path}")

    # 加载 + 模拟一段假转写(算法层不需要真实 ASR)
    y, sr = load_audio(audio_path)

    fake_word_times = [
        {"word": "I",        "start": 0.5, "end": 0.55, "score": 0.9},
        {"word": "really",   "start": 0.6, "end": 0.75, "score": 0.85},
        {"word": "like",     "start": 0.8, "end": 0.9, "score": 0.7},
        {"word": "cooking",  "start": 1.0, "end": 1.3, "score": 0.9},
        {"word": "because",  "start": 1.4, "end": 1.7, "score": 0.85},
        {"word": "it's",     "start": 1.8, "end": 1.95, "score": 0.8},
        {"word": "fun.",     "start": 2.0, "end": 2.3, "score": 0.9},
        {"word": "I",        "start": 2.5, "end": 2.55, "score": 0.9},
        {"word": "learn",    "start": 2.7, "end": 2.9, "score": 0.85},
        {"word": "from",     "start": 3.0, "end": 3.2, "score": 0.9},
        {"word": "it.",      "start": 3.3, "end": 3.6, "score": 0.9},
    ]
    fake_text = "I really like cooking because it's fun. I learn from it."
    duration = 5.0

    print("\n=== DSP metrics ===")
    dsp = compute_dsp_metrics(y, sr, fake_word_times, duration)
    for k, v in dsp.items():
        print(f"  {k}: {v}")

    print("\n=== Text stats ===")
    sentences = split_into_sentences(fake_text, fake_word_times, duration)
    print(f"  Sentences: {[s['text'] for s in sentences]}")
    text_stats = compute_text_metrics(fake_text, [s["text"] for s in sentences])
    for k, v in text_stats.items():
        if not isinstance(v, list):
            print(f"  {k}: {v}")

    print("\n=== Pronunciation metrics ===")
    pron = compute_pronunciation_metrics(y, sr, fake_text)
    for k, v in pron.items():
        if not isinstance(v, str) or len(str(v)) < 80:
            print(f"  {k}: {v}")

    print("\n=== Fillers ===")
    fillers = count_fillers(fake_text)
    print(f"  Found: {fillers}")

    print("\n✅ Algorithm layer smoke test passed")


if __name__ == "__main__":
    asyncio.run(main())