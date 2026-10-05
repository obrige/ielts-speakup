"""直接测试 LLM 调用"""
import asyncio
from scoring.llm_scorer import call_json, gather_json
from scoring import prompts

async def test_simple():
    print('=== Test 1: 简单 JSON 输出 ===')
    result = await call_json(
        prompt='输出 band 6.0 的 JSON。严格格式:{"test": "ok", "band": 6.0}',
        system='你是一个 JSON 输出助手,只输出 JSON,不加解释。',
        max_tokens=50,
        temperature=0.1,
    )
    print(f'  Result: {result}')
    assert result is not None
    assert "band" in result
    print('  [OK]')


async def test_sub_dim():
    print()
    print('=== Test 2: sub-dim prompt ===')
    # 模拟一段英语
    test_text = "I really enjoy playing basketball in my free time. It is a great way to stay healthy and meet new people. I started playing when I was a child."
    fake_metrics = {
        "speech_rate_wpm": 120.5,
        "duration_sec": 15.0,
        "pause_count": 2,
        "pause_total": 1.2,
        "fillers": ["um"],
    }
    prompt = prompts.sub_dim_prompt(
        sub_dim_name="FC-SpeechRate",
        question="Do you have any hobbies?",
        text=test_text,
        relevant_metrics={k: fake_metrics[k] for k in ["speech_rate_wpm", "duration_sec"]},
    )
    result = await call_json(
        prompt=prompt,
        system=prompts.SYSTEM_PROMPT_BASE,
        max_tokens=400,
        temperature=0.2,
    )
    print(f'  Result: {result}')
    assert result is not None
    assert "sub_dimension" in result
    assert "band" in result
    print('  [OK]')


async def test_gather():
    print()
    print('=== Test 3: 并发 3 个 sub-dim 调用 ===')
    test_text = "I really enjoy playing basketball in my free time."
    prompts_list = []
    for name in ["FC-SpeechRate", "LR-LexicalRichness", "FC-CohesiveDevices"]:
        prompts_list.append({
            "prompt": prompts.sub_dim_prompt(
                sub_dim_name=name,
                question="Do you have any hobbies?",
                text=test_text,
                relevant_metrics={},
            ),
            "max_tokens": 300,
        })

    results = await gather_json(
        prompts_list,
        system=prompts.SYSTEM_PROMPT_BASE,
        max_tokens=300,
        concurrency=3,
    )
    print(f'  Got {len(results)} results')
    for i, r in enumerate(results):
        status = "OK" if r else "FAIL"
        print(f'  [{status}] call {i+1}: band={r.get("band") if r else "N/A"}')
    success = sum(1 for r in results if r)
    print(f'  Success rate: {success}/{len(results)}')


asyncio.run(test_simple())
asyncio.run(test_sub_dim())
asyncio.run(test_gather())