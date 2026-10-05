"""绕过 librosa,纯算法层 smoke test"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy.io import wavfile

# 模拟 audio_metrics 的核心算法(不依赖 librosa)
sr, y = wavfile.read(str(Path(__file__).parent / "test.wav"))
print(f"Loaded WAV: sr={sr}, samples={len(y)}, dtype={y.dtype}")
if y.dtype != np.float32:
    y = y.astype(np.float32) / np.max(np.abs(y))

# ===== 手算基础 DSP 指标 =====
duration_sec = len(y) / sr
print(f"Duration: {duration_sec:.2f}s")

# RMS
frame_length = 512
hop_length = 128
n_frames = (len(y) - frame_length) // hop_length + 1
rms_frames = np.array([
    np.sqrt(np.mean(y[i * hop_length: i * hop_length + frame_length] ** 2))
    for i in range(n_frames)
])
print(f"RMS mean: {rms_frames.mean():.4f}, std: {rms_frames.std():.4f}")
rms_var = rms_frames.std() / (rms_frames.mean() + 1e-6)

# VAD 静音段
top_db = 30
peak = np.max(np.abs(y))
threshold = peak * 10 ** (-top_db / 20)
voiced_mask = np.abs(y) > threshold

# 找连续 voiced 段
edges = np.diff(voiced_mask.astype(int))
starts = np.where(edges == 1)[0] + 1
ends = np.where(edges == -1)[0] + 1
if voiced_mask[0]:
    starts = np.concatenate(([0], starts))
if voiced_mask[-1]:
    ends = np.concatenate((ends, [len(y)]))

intervals = list(zip(starts, ends))
voiced_dur = sum((end - start) / sr for start, end in intervals)
silence_dur = max(0.0, duration_sec - voiced_dur)
pause_count = max(0, len(intervals) - 1)
print(f"Voiced segments: {len(intervals)}, voiced dur: {voiced_dur:.2f}s, silence: {silence_dur:.2f}s, pauses: {pause_count}")

# 假 word_times
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

word_count = len(fake_word_times)
speech_rate_wpm = word_count / (duration_sec / 60)
print(f"Word count: {word_count}, WPM: {speech_rate_wpm:.1f}")

# Text metrics
from scoring.text_stats import compute_text_metrics
from scoring.audio_metrics import count_fillers
sentences = [
    "I really like cooking because it's fun.",
    "I learn from it."
]
text_stats = compute_text_metrics(fake_text, sentences)
fillers = count_fillers(fake_text)

print()
print("=== Aggregated metrics ===")
all_metrics = {
    "f0_std": 0.0,  # 跳过 librosa.pyin
    "rms_var": round(rms_var, 3),
    "rhythm_cv": 0.0,
    "speech_rate_wpm": round(speech_rate_wpm, 1),
    "pause_count": pause_count,
    "pause_total": round(silence_dur, 2),
    "duration_sec": round(duration_sec, 2),
    "word_count": word_count,
    "fillers": fillers,
    **text_stats,
}
for k, v in all_metrics.items():
    if not isinstance(v, list):
        print(f"  {k}: {v}")
print()
print("✅ DSP-layer smoke test (scipy) passed")