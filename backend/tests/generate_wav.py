"""生成合成 WAV 音频供测试用"""
import numpy as np
import soundfile as sf
from pathlib import Path

# 30 秒模拟语音信号(16000Hz)
sr = 16000
duration = 30
t = np.linspace(0, duration, sr * duration)

# 模拟语音基频 + 静音段
signal = np.zeros_like(t)

# 用 AM 调制模拟响度变化(更像语音)
carrier = np.sin(2 * np.pi * 180 * t)  # 180Hz 基频
envelope = np.zeros_like(t)

# 5 个"说话"段,每段 4-5 秒,中间有停顿
segments = [
    (0.5, 5.0),    # 第一段
    (6.0, 10.0),   # 第二段(中间 1s 停顿)
    (11.5, 16.0),  # 第三段(中间 1.5s 停顿)
    (17.5, 22.0),  # 第四段
    (23.5, 28.0),  # 第五段
]

for start, end in segments:
    s = int(start * sr)
    e = int(end * sr)
    seg_t = t[s:e]
    # 振幅包络有变化,模拟重音
    amp = 0.3 + 0.1 * np.sin(2 * np.pi * 3 * seg_t)
    envelope[s:e] = amp

signal = carrier * envelope
# 加点白噪声更像真实录音
signal += np.random.normal(0, 0.005, len(signal))

out_path = Path(__file__).parent / "test.wav"
sf.write(out_path, signal, sr)
print(f"Generated {out_path} ({duration}s @ {sr}Hz, {len(signal)} samples)")
print(f"File size: {out_path.stat().st_size} bytes")