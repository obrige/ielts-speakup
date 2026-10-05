"""端到端测试:跳过 ASR,直接喂假转写,看 LLM pipeline 跑不跑得通"""
import asyncio
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

from scoring.text_stats import compute_text_metrics
from scoring.audio_metrics import count_fillers, split_into_sentences
from scoring.llm_scorer import call_json
from scoring import prompts

# ===== 模拟一段雅思学员回答 =====
FAKE_TRANSCRIPT = (
    "I really enjoy playing basketball in my free time because it is a great way to "
    "stay healthy and meet new people. I started playing when I was a child, and I have "
    "been playing for many years now. Every weekend I go to the local court with my "
    "friends, and we play for several hours. It helps me to relax after a busy week "
    "of work, and I think it is also good for my health. However, I do not always "
    "have time to play because of my job, so I try to play at least once a week."
)

# 假 word_times(每词 0.3-0.5s 间隔)
FAKE_WORD_TIMES = []
t = 0.0
for word in FAKE_TRANSCRIPT.split():
    duration = 0.3 + (len(word) % 5) * 0.05
    FAKE_WORD_TIMES.append({"word": word, "start": t, "end": t + duration, "score": 0.85})
    t += duration + 0.1

DURATION_SEC = FAKE_WORD_TIMES[-1]["end"]

# ===== 算法层指标 =====
word_count = len(FAKE_WORD_TIMES)
sentences = split_into_sentences(FAKE_TRANSCRIPT, FAKE_WORD_TIMES, DURATION_SEC)
text_stats = compute_text_metrics(FAKE_TRANSCRIPT, [s["text"] for s in sentences])
fillers = count_fillers(FAKE_TRANSCRIPT)

all_metrics = {
    "speech_rate_wpm": round(word_count / (DURATION_SEC / 60), 1),
    "duration_sec": round(DURATION_SEC, 2),
    "word_count": word_count,
    "pause_count": 5,
    "pause_total": 1.5,
    "fillers": fillers,
    "wav2vec2_confidence": 0.78,  # 模拟值
    "phoneme_edit_distance": 0.15,
    "f0_std": 25.3,
    "rms_var": 0.42,
    "rhythm_cv": 0.35,
    **text_stats,
}

print(f"=== 输入 ===")
print(f"  字数: {word_count}")
print(f"  句数: {len(sentences)}")
print(f"  时长: {DURATION_SEC:.1f}s")
print(f"  WPM: {all_metrics['speech_rate_wpm']}")
print(f"  语法错误: {text_stats['grammar_error_count']}")
print(f"  高级词: {text_stats['advanced_word_count']}")
print(f"  搭配: {text_stats['collocation_hits']}")
print()


async def run():
    # ===== 14 sub-dim 调用 =====
    print("=== 14 sub-dim LLM 调用 ===")
    sub_tasks = []
    for name, info in prompts.SUB_DIMS.items():
        relevant = {k: all_metrics.get(k) for k in info["metrics"]}
        prompt = prompts.sub_dim_prompt(
            sub_dim_name=name,
            question="Do you have any hobbies?",
            text=FAKE_TRANSCRIPT,
            relevant_metrics=relevant,
        )
        sub_tasks.append((name, prompt))

    # 并发调用
    async def call_sub(name, prompt):
        return name, await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=400)

    results = await asyncio.gather(*[call_sub(n, p) for n, p in sub_tasks])
    sub_map = {}
    for name, r in results:
        if r and "band" in r:
            normalized = prompts.normalize_band(r["band"])
            if normalized is not None:
                r["band"] = normalized
                sub_map[name] = r
                print(f"  [{name}] band={r['band']}")
            else:
                print(f"  [{name}] band normalization failed: {r['band']}")
        else:
            print(f"  [{name}] LLM failed")

    print(f"\n  Got {len(sub_map)}/14 sub-dims")

    # ===== 4 main-dim 调用 =====
    print()
    print("=== 4 main-dim LLM 调用 ===")
    groups = {dim: {} for dim in prompts.MAIN_DIMS}
    for name, data in sub_map.items():
        dim = prompts.SUB_DIMS[name]["main_dim"]
        groups[dim][name] = data

    main_tasks = []
    for dim in prompts.MAIN_DIMS:
        prompt = prompts.main_dim_prompt(
            main_dim=dim,
            question="Do you have any hobbies?",
            text=FAKE_TRANSCRIPT,
            all_metrics=all_metrics,
            sub_dim_evidence=groups[dim],
        )
        main_tasks.append((dim, prompt))

    async def call_main(dim, prompt):
        return dim, await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=500)

    main_results = await asyncio.gather(*[call_main(d, p) for d, p in main_tasks])
    main_map = {}
    for dim, r in main_results:
        if r and "band" in r:
            normalized = prompts.normalize_band(r["band"])
            if normalized is not None:
                r["band"] = normalized
                main_map[dim] = r
                print(f"  [{dim}] band={r['band']}")
            else:
                print(f"  [{dim}] band normalization failed")
        else:
            print(f"  [{dim}] LLM failed")

    # ===== Overall =====
    if len(main_map) == 4:
        bands = [r["band"] for r in main_map.values()]
        overall = round(sum(bands) / 4 * 2) / 2
        print(f"\n  Overall: {overall}")
    else:
        print(f"\n  Overall: N/A (only {len(main_map)} main-dims)")

    # ===== Sentence feedback =====
    print()
    print(f"=== {len(sentences)} sentence feedback ===")
    sent_tasks = []
    for sent in sentences[:3]:  # 前 3 句示例
        prompt = prompts.SENTENCE_FEEDBACK_PROMPT.format(
            sent_id=sent["sent_id"],
            sentence_text=sent["text"],
            start=sent["start"],
            end=sent["end"],
            full_transcript=FAKE_TRANSCRIPT[:800],
            wav2vec2_conf=0.78,
            pause_before=0.0,
            pause_after=0.0,
            fillers_in_sent=[],
            grammar_errors=json.dumps([], ensure_ascii=False),
            dominant_dim="FC",
            affected_sub_dims="FC-SpeechRate, FC-CohesiveDevices",
        )
        sent_tasks.append((sent["sent_id"], prompt))

    async def call_sent(sid, prompt):
        return sid, await call_json(prompt=prompt, system=prompts.SYSTEM_PROMPT_BASE, max_tokens=600)

    sent_results = await asyncio.gather(*[call_sent(sid, p) for sid, p in sent_tasks])
    for sid, r in sent_results:
        if r:
            n_issues = len(r.get("issues", []))
            n_improvements = len(r.get("improvements", []))
            print(f"  [Sent #{sid}] {n_issues} issues, {n_improvements} improvements, "
                  f"model_answer: {len(r.get('model_answer', '') or '')} chars")
        else:
            print(f"  [Sent #{sid}] FAILED")

    print()
    print("✅ 完整 pipeline 跑通(除 ASR 外)")

asyncio.run(run())