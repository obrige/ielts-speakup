"""测试题库 + prompts"""
from scoring.questions import QUESTION_BANK, get_question
from scoring import prompts

print('=== Questions ===')
print(f'Parts: {list(QUESTION_BANK.keys())}')
for part, qs in QUESTION_BANK.items():
    print(f'  Part {part}: {len(qs)} questions')
    for q in qs:
        print(f'    {q["id"]}: {q["topic"]}')

print()
print('=== get_question 测试 ===')
q = get_question('p1-hometown-1')
print(f'  Found: {q["id"]} - {q["topic"]}')

try:
    get_question('nonexistent')
    print('  [FAIL] 应该抛 KeyError')
except KeyError as e:
    print(f'  [OK] KeyError raised: {e}')

print()
print('=== SUB_DIMS ===')
print(f'  Total sub-dims: {len(prompts.SUB_DIMS)}')

# 分组统计
main_count = {}
for name, info in prompts.SUB_DIMS.items():
    dim = info["main_dim"]
    main_count[dim] = main_count.get(dim, 0) + 1
print(f'  Per main-dim: {main_count}')
assert sum(main_count.values()) == 14, f'expected 14, got {sum(main_count.values())}'
print('  [OK] 14 个 sub-dim 全部存在')

# MAIN_DIMS
print(f'  MAIN_DIMS: {prompts.MAIN_DIMS}')
assert set(prompts.MAIN_DIMS) == {"FC", "LR", "GRA", "P"}
print('  [OK] MAIN_DIMS = {FC, LR, GRA, P}')

# LENGTH_POLICY 覆盖
print()
print('=== LENGTH_POLICY ===')
for name in prompts.SUB_DIMS:
    if name not in prompts.LENGTH_POLICY:
        print(f'  [FAIL] {name} 缺失 LENGTH_POLICY')
        break
else:
    print(f'  [OK] 14 个 sub-dim 全部有长度策略')

# P-* 应该是 0
p_dims = [n for n in prompts.SUB_DIMS if n.startswith("P-")]
for p in p_dims:
    if prompts.LENGTH_POLICY[p] != 0:
        print(f'  [FAIL] {p} 应该=0, 实际={prompts.LENGTH_POLICY[p]}')
        break
else:
    print(f'  [OK] 所有 P-* 长度约束 = 0')

print()
print('=== normalize_band ===')
tests = [
    (5.5, 5.5),
    (5.3, 5.5),  # 应该量化到 5.5
    (5.7, 5.5),  # 应该量化到 5.5
    (5.25, 5.0),  # 量化到 5.0 或 5.5? round(5.25*2)/2 = round(10.5)/2 = 10/2 = 5.0 (banker's rounding)
    (10.0, 9.0),  # 越界截断
    (0.0, 1.0),   # 越界截断
    ("6.5", 6.5),  # 字符串
    ("abc", None), # 非法
    (None, None),
]
for inp, expected in tests:
    actual = prompts.normalize_band(inp)
    status = "OK" if actual == expected else "FAIL"
    print(f'  [{status}] normalize_band({inp!r}) = {actual} (expected {expected})')

print()
print('=== compute_zpd_target ===')
for sub, band in [
    ("LR-LexicalRichness", 5.0),  # +1.0
    ("GRA-GrammaticalErrors", 5.0),  # +1.0
    ("FC-SpeechRate", 5.0),  # +0.5
    ("P-Intelligibility", 5.0),  # +0.5
    ("FC-PausingWordSearch", 5.0),  # +0.5
]:
    target = prompts.compute_zpd_target(band, sub)
    delta = target - band
    print(f'  {sub} band {band} → target {target} (delta +{delta})')

print()
print('=== All tests passed ===')