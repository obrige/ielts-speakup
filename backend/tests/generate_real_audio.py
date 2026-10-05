"""用 edge-tts 生成真实英语音频供 pipeline 测试"""
import asyncio
import edge_tts
from pathlib import Path

# 模拟一段雅思学员回答(70秒左右的内容)
TEXT = """
I really enjoy playing basketball in my free time because it is a great way to stay healthy and meet new people.
I started playing when I was a child, and I have been playing for many years now.
Every weekend I go to the local court with my friends, and we play for several hours.
It helps me to relax after a busy week of work, and I think it is also good for my health.
However, I do not always have time to play because of my job, so I try to play at least once a week.
Sometimes I feel tired, but I still go because basketball is very important to me.
Last month, I joined a basketball club in my city, and I made some new friends there.
The club meets every Saturday, and we practice for two hours.
I have learned many new skills, such as how to dribble better and how to shoot from different angles.
My favorite player is LeBron James because he is not only a great scorer but also a great team player.
In the future, I want to take part in a local basketball competition.
I believe that playing basketball has made me a more confident and disciplined person.
""".strip()

# 选个自然的英语男声
VOICE = "en-US-AndrewMultilingualNeural"

async def main():
    out_path = Path(__file__).parent / "real_audio.mp3"
    print(f"Generating {VOICE} → {out_path}...")
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(str(out_path))
    print(f"Done. File size: {out_path.stat().st_size} bytes")
    print(f"Text length: {len(TEXT)} chars, words: {len(TEXT.split())}")

    # 同时存个 16kHz wav 版本(给 pipeline 测试用)
    print()
    print("Generating WAV (24kHz PCM) version...")
    wav_path = Path(__file__).parent / "real_audio_raw.wav"
    communicate2 = edge_tts.Communicate(TEXT, VOICE)
    sub = communicate2.stream()
    with open(wav_path, "wb") as f:
        async for chunk in sub:
            if chunk["type"] == "audio":
                f.write(chunk["data"])
    print(f"  WAV: {wav_path} ({wav_path.stat().st_size} bytes)")

asyncio.run(main())