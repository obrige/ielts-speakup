"""
完整 E2E 测试:
- edge-tts 生成真实音频
- soundfile 读取 + 重采样到 16kHz
- 手算 DSP (绕过 librosa.soxr 问题)
- 用原文本作 ASR 结果(模拟 ASR)
- 跑完整 LLM pipeline (14 sub + 4 main + sent + ZPD + summary)
"""
import asyncio
import json
import re
from pathlib import Path
import numpy as np
import soundfile as sf

from scoring.text_stats import compute_text_metrics
from scoring.audio_metrics import count_fillers, split_into_sentences
from scoring.llm_scorer import call_json
from scoring import prompts

TEXT = (
    "I really enjoy playing basketball in my free time because it is a great way to "
    "stay healthy and meet new people. I started playing when I was a child, and I have "
    "been playing for many years now. Every weekend I go to the local court with my "
    "friends, and we play for several hours. It helps me to relax after a busy week "
    "of work, and I think it is also good for my health. However, I do not always "
    "have time to play because of my job, so I try to play at least once a week."
)

# ===== soundfile 读 MP3 + 重采样到 16kHz =====
mp3_path = Path(__file__).parent / "real_audio.mp3"
print(f"Reading {mp3_path}...")
y, sr = sf.read(str(mp3_path))
print(f"Original: sr={sr}Hz, samples={len(y)}, dur={len(y)/sr:.1f}s")

# 简单线性重采样到 16000 Hz(足够测试用)
target_sr = 16000
ratio = target_sr / sr
new_len = int(len(y) * ratio)
y_resampled = np.interp(
    np.linspace(0, len(y), new_len),
    np.arange(len(y)),
    y,
).astype(np.float32)
y = y_resampled / (np.max(np.abs(y_resampled)) + 1e-9)
sr = target_sr
DURATION = len(y) / sr
print(f"Resampled: sr={sr}Hz, samples={len(y)}, dur={DURATION:.1f}s")

print(f"Audio: {DURATION:.1f}s @ {sr}Hz, peak amp = {np.max(np.abs(y)):.3f}")

# 假 word_times: 按平均时长分配
words = TEXT.split()
word_dur = DURATION / len(words)
FAKE_WORD_TIMES = []
t = 0.0
for w in words:
    dur = word_dur * (0.7 + 0.6 * (hash(w) % 100) / 100)  # 模拟节奏变化
    FAKE_WORD_TIMES.append({"word": w, "start": t, "end": t + dur, "score": 0.85})
    t += dur

# VAD 简易静音段
peak_db = 20 * np.log10(np.max(np.abs(y)) + 1e-9)
noise_db = 20 * np.log10(np.std(y) + 1e-9)
threshold_db = noise_db + (peak_db - noise_db) * 0.4
mask = 20 * np.log10(np.abs(y) + 1e-9) > threshold_db
edges = np.diff(mask.astype(int))
starts = np.where(edges == 1)[0] + 1
ends = np.where(edges == -1)[0] + 1
if mask[0]: starts = np.concatenate(([0], starts))
if mask[-1]: ends = np.concatenate((ends, [len(y)]))
intervals = list(zip(starts, ends))
# 合并 < 200ms 的 gap
merged = []
for s, e in intervals:
    if merged and (s - merged[-1][1]) / sr < 0.2:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))
voiced_dur = sum((e - s) / sr for s, e in merged)
silence_dur = max(0.0, DURATION - voiced_dur)
pause_count = max(0, len(merged) - 1)

# RMS
frame = 512
hop = 128
n_frames = (len(y) - frame) // hop + 1
rms_frames = np.array([np.sqrt(np.mean(y[i*hop:i*hop+frame]**2)) for i in range(n_frames)])
rms_var = float(rms_frames.std() / (rms_frames.mean() + 1e-6))

# F0 via 自相关(避开 librosa.pyin)
def estimate_f0(y, sr, fmin=80, fmax=400):
    f0s = []
    win = int(0.025 * sr)
    for i in range(0, len(y) - win, win):
        seg = y[i:i+win]
        seg = seg * np.hanning(len(seg))
        if np.max(np.abs(seg)) < 0.05:  # 静音跳过
            continue
        ac = np.correlate(seg, seg, mode='full')[len(seg):]
        # 找最大 lag in [1/fmax*sr, 1/fmin*sr]
        lo = int(sr / fmax)
        hi = int(sr / fmin)
        if hi >= len(ac): continue
        peak = np.argmax(ac[lo:hi]) + lo
        if peak > 0 and ac[peak] > 0.3 * ac[0]:
            f0s.append(sr / peak)
    return np.array(f0s) if f0s else np.array([180.0])

f0 = estimate_f0(y, sr)
f0_std = float(f0.std()) if len(f0) > 5 else 50.0

# Rhythm CV from word intervals
ends = [w["end"] for w in FAKE_WORD_TIMES]
intervals = np.diff(ends)
rhythm_cv = float(intervals.std() / (intervals.mean() + 1e-6)) if len(intervals) > 1 else 0.5

word_count = len(words)
sentences = split_into_sentences(TEXT, FAKE_WORD_TIMES, DURATION)
text_stats = compute_text_metrics(TEXT, [s["text"] for s in sentences])
fillers = count_fillers(TEXT)

all_metrics = {
    "speech_rate_wpm": round(word_count / (DURATION / 60), 1),
    "duration_sec": round(DURATION, 2),
    "word_count": word_count,
    "pause_count": pause_count,
    "pause_total": round(silence_dur, 2),
    "fillers": fillers,
    "wav2vec2_confidence": 0.78,  # 模拟值(无 wav2vec2)
    "phoneme_edit_distance": 0.18,
    "f0_std": round(f0_std, 1),
    "rms_var": round(rms_var, 3),
    "rhythm_cv": round(rhythm_cv, 3),
    **text_stats,
}

print()
print("=== 算法层输出 ===")
for k, v in all_metrics.items():
    if not isinstance(v, list):
        print(f"  {k}: {v}")

# ===== LLM pipeline =====
async def run():
    print()
    print("=== 14 sub-dim LLM 调用 ===")
    async def call_sub(name, info):
        relevant = {k: all_metrics.get(k) for k in info["metrics"]}
        prompt = prompts.sub_dim_prompt(
            sub_dim_name=name,
            question="Do you have any hobbies?",
            text=TEXT,
            relevant_metrics=relevant,
        )
        return name, await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=400)

    sub_tasks = [call_sub(n, i) for n, i in prompts.SUB_DIMS.items()]
    results = await asyncio.gather(*sub_tasks)
    sub_map = {}
    for name, r in results:
        if r and "band" in r:
            n = prompts.normalize_band(r["band"])
            if n is not None:
                r["band"] = n
                sub_map[name] = r
    print(f"  Got {len(sub_map)}/14")

    print()
    print("=== 4 main-dim LLM 调用 ===")
    groups = {dim: {} for dim in prompts.MAIN_DIMS}
    for name, data in sub_map.items():
        groups[prompts.SUB_DIMS[name]["main_dim"]][name] = data

    async def call_main(dim):
        prompt = prompts.main_dim_prompt(
            main_dim=dim,
            question="Do you have any hobbies?",
            text=TEXT,
            all_metrics=all_metrics,
            sub_dim_evidence=groups[dim],
        )
        return dim, await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=500)

    main_results = await asyncio.gather(*[call_main(d) for d in prompts.MAIN_DIMS])
    main_map = {}
    for dim, r in main_results:
        if r and "band" in r:
            n = prompts.normalize_band(r["band"])
            if n is not None:
                r["band"] = n
                main_map[dim] = r
                print(f"  [{dim}] {r['band']}")

    overall = None
    if len(main_map) == 4:
        bands = [r["band"] for r in main_map.values()]
        overall = round(sum(bands) / 4 * 2) / 2
        print(f"  Overall: {overall}")

    print()
    print(f"=== {len(sentences)} sentence feedback ===")

    async def call_sent(sent):
        prompt = prompts.SENTENCE_FEEDBACK_PROMPT.format(
            sent_id=sent["sent_id"],
            sentence_text=sent["text"],
            start=sent["start"],
            end=sent["end"],
            full_transcript=TEXT[:1500],
            wav2vec2_conf=0.78,
            pause_before=0.0,
            pause_after=0.0,
            fillers_in_sent=[],
            grammar_errors=json.dumps([], ensure_ascii=False),
            dominant_dim="FC",
            affected_sub_dims="FC-SpeechRate, FC-CohesiveDevices",
        )
        return sent["sent_id"], await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=600)

    sent_results = await asyncio.gather(*[call_sent(s) for s in sentences])
    sent_feedback = [(sid, r) for sid, r in sent_results if r]
    print(f"  Got {len(sent_feedback)}/{len(sentences)}")

    print()
    print("=== ZPD 个性化范例 ===")
    # 简化:挑每个 moderate/major issue 一次
    zpd_tasks = []
    for sid, r in sent_feedback[:3]:
        for issue in r.get("issues", [])[:1]:
            if issue.get("severity") == "minor":
                continue
            affected_sub = {
                "grammar": "GRA-GrammaticalErrors",
                "vocabulary": "LR-LexicalRichness",
                "fluency": "FC-PausingWordSearch",
                "pronunciation": "P-Clarity",
            }.get(issue.get("type"), "FC-LogicalStructure")
            if affected_sub not in sub_map:
                continue
            target = prompts.compute_zpd_target(sub_map[affected_sub]["band"], affected_sub)
            length_extra = prompts.LENGTH_POLICY.get(affected_sub, 10)
            main_dim = prompts.SUB_DIMS[affected_sub]["main_dim"]
            main_band = main_map.get(main_dim, {}).get("band", 5.0)
            prompt = prompts.ZPD_PROMPT.format(
                overall_band=overall or 5.0,
                main_dim=main_dim,
                main_dim_band=main_band,
                sub_dim_name=affected_sub,
                sub_dim_band=sub_map[affected_sub]["band"],
                target_band=target,
                learner_sentence=r["original"],
                learner_words=len(r["original"].split()),
                length_extra=length_extra,
                issues_json=json.dumps(issue, ensure_ascii=False),
            )
            zpd_tasks.append(call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=500))
    zpd_out = [r for r in await asyncio.gather(*zpd_tasks) if r]
    print(f"  Got {len(zpd_out)} ZPD examples")

    print()
    print("=== Summary ===")
    summary_prompt = prompts.SUMMARY_PROMPT.format(
        main_scores_json=json.dumps(
            [{"criterion": k, "band": v["band"]} for k, v in main_map.items()],
            ensure_ascii=False, indent=2),
        sub_scores_json=json.dumps(
            [{"sub": k, "band": v["band"]} for k, v in sub_map.items()],
            ensure_ascii=False, indent=2),
        overall_band=overall or 5.0,
    )
    summary = await call_json(prompt=summary_prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=500)
    if summary:
        print(f"  CEFR: {summary.get('cefr_level')}")
        print(f"  Strengths: {len(summary.get('strengths', []))}")
        print(f"  Weaknesses: {len(summary.get('weaknesses', []))}")
        print(f"  Priorities: {len(summary.get('priority_improvements', []))}")

    # ===== 输出完整报告 =====
    report = {
        "metadata": {
            "question_id": "p1-hobby-1",
            "question_text": "Do you have any hobbies?",
            "audio_duration_sec": round(DURATION, 2),
            "word_count": word_count,
            "sentence_count": len(sentences),
            "cefr_level": (summary or {}).get("cefr_level"),
            "disclaimer": "AI 评分尚未经过真人考官校准验证,单次评分误差约 ±0.5 band",
        },
        "scores": {
            "Overall": overall,
            **{k: v["band"] for k, v in main_map.items()},
        },
        "main_score_details": list(main_map.values()),
        "sub_scores": list(sub_map.values()),
        "raw_metrics": all_metrics,
        "sentence_feedback": [r for _, r in sent_feedback],
        "zpd_examples": zpd_out,
        "summary": summary,
    }

    out_path = Path(__file__).parent / "real_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print()
    print(f"✅ 完整报告保存到 {out_path}")
    print(f"   Overall: {overall}")
    print(f"   File size: {out_path.stat().st_size} bytes")

asyncio.run(run())