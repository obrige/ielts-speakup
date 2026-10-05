"""测试 grammar 正则 — 含已知错误的句子"""
from scoring.text_stats import grammar_error_count

tests = [
    ("I really like cooking.", 0),  # 完全正确
    ("She have a book.", 1),        # she + have 错误
    ("They was happy.", 1),         # they + was 错误
    ("I is tired.", 1),             # I + is 错误
    ("a apple", 1),                 # a + 元音 → an
    ("an book", 1),                 # an + 辅音 → a
    ("a book", 0),                  # 正确
    ("an apple", 0),                # 正确
    ("hello world", 0),             # 正确
    ("I have a apple and an book.", 2),  # 两个错误
    ("He have an apple.", 1),       # 1 个错误 (he + have)
    ("She have an apple.", 1),      # 1 个错误 (she + have)
    ("", 0),                         # 空文本
    ("I  have  two  spaces.", 2),   # 双空格
]

for text, expected in tests:
    g = grammar_error_count(text)
    actual = g["count"]
    status = "OK" if actual == expected else "FAIL"
    print(f'  [{status}] "{text}"  expected={expected}, actual={actual}  {g["examples"]}')