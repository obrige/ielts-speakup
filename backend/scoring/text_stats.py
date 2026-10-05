"""
文本层指标 - 全部纯算法,零 LLM
- TTR / MTLD: 词汇丰富度
- Collocation: 常见搭配命中率(简易 ngram 库)
- Grammar errors: language-tool 计数(若无则降级到正则)
- Sentence types: 简单/复合/复杂句统计
- Advanced word count: AWL-like 简易检测

⚠️ 注意:这些指标是 raw metrics,需由 LLM 对照官方 rubric 判断 band。
本模块不输出 band 分。
"""
from __future__ import annotations
import re
from typing import Iterable


# ===== 词汇丰富度 =====

def ttr(text: str) -> float:
    """Type-Token Ratio (受长度影响, 仅作参考)"""
    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mtld(text: str, threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity
    沿文本滑动, 计算 TTR 降到阈值时的因子数;比 TTR 更稳定。
    """
    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    if len(tokens) < 10:
        return ttr(text) * 100

    def ttr_at(segment: Iterable[str]) -> float:
        seg = list(segment)
        if not seg:
            return 0.0
        return len(set(seg)) / len(seg)

    factor_count = 0
    for i in range(len(tokens)):
        forward = tokens[i:]
        if len(forward) < 10:
            break
        current_ttr = 1.0
        for j in range(len(forward)):
            current_ttr = ttr_at(forward[: j + 1])
            if current_ttr <= threshold:
                factor_count += 1
                break

    # 双向取平均更稳定
    backward_count = 0
    for i in range(len(tokens), 0, -1):
        backward = tokens[:i]
        if len(backward) < 10:
            break
        current_ttr = 1.0
        for j in range(len(backward), 0, -1):
            current_ttr = ttr_at(backward[:j])
            if current_ttr <= threshold:
                backward_count += 1
                break

    forward_score = factor_count / (len(tokens) / 100) if tokens else 0
    backward_score = backward_count / (len(tokens) / 100) if tokens else 0
    return (forward_score + backward_score) / 2


# ===== 搭配检测 =====

# 简易 AWL (Academic Word List) 子集,实际项目应加载完整 AWL
COMMON_ADVANCED = {
    "furthermore", "moreover", "nevertheless", "consequently", "significantly",
    "predominantly", "substantially", "comprehensive", "contemporary", "fundamental",
    "perspective", "perception", "phenomenon", "substantial", "conceptual",
    "paradigm", "criterion", "criteria", "hypothesis", "methodology",
    "subsequently", "approximately", "predominantly", "tremendous", "remarkable",
    "enthusiasm", "vulnerable", "controversial", "discretion", "explicit",
    "implicit", "comprise", "constitute", "depict", "demonstrate",
    "eligible", "extensive", "feasible", "fluctuate", "fundamental",
    "generate", "implement", "indispensable", "inhabit", "integrate",
}


def advanced_word_count(text: str) -> tuple[int, list[str]]:
    """返回高级词命中数 + 命中词列表"""
    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    found = [t for t in tokens if t in COMMON_ADVANCED]
    return len(found), list(set(found))


# 简易 Bigram collocation 库 (仅作示意,实际用 ngram_reference)
COMMON_COLLOCATIONS = {
    ("make", "decision"), ("take", "responsibility"), ("pay", "attention"),
    ("keep", "mind"), ("take", "into"), ("play", "role"), ("come", "up"),
    ("carry", "out"), ("put", "into"), ("get", "rid"), ("look", "forward"),
    ("make", "difference"), ("take", "place"), ("come", "across"),
    ("find", "out"), ("figure", "out"), ("deal", "with"), ("depend", "on"),
    ("cope", "with"), ("look", "into"), ("carry", "on"), ("hold", "on"),
}


def collocation_hits(text: str) -> tuple[int, list[str]]:
    """bigram 命中数 + 命中列表"""
    tokens = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    found = []
    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram in COMMON_COLLOCATIONS:
            found.append(f"{tokens[i]} {tokens[i+1]}")
    return len(found), list(set(found))


# ===== 语法错误计数 =====

def grammar_error_count(text: str) -> dict:
    """
    尝试用 language-tool(若可用),否则降级到简易正则。
    返回:
    - count: 错误数
    - examples: [{rule, message, location, suggestion}, ...]
    """
    try:
        import language_tool_python
        tool = language_tool_python.LanguageTool("en-US")
        matches = tool.check(text)
        examples = [
            {
                "rule": m.rule_id,
                "message": m.message,
                "location": text[max(0, m.offset): m.offset + m.errorLength],
                "suggestion": m.replacements[0] if m.replacements else "",
            }
            for m in matches[:5]  # 截前5个
        ]
        return {"count": len(matches), "examples": examples}
    except Exception:
        # 降级:简易正则检查
        return _simple_grammar_check(text)


def _simple_grammar_check(text: str) -> dict:
    """无 language-tool 时的兜底:检测基础错误模式

    仅检测明确的错误模式,不做启发式判断(避免大量 false positive)。
    """
    patterns = [
        # 主谓一致(case-insensitive)
        (r"(?i)\bi\s+is\b", "subject-verb agreement: 'I' should be 'am'"),
        (r"(?i)\bhe\s+have\b", "subject-verb agreement: 'have' should be 'has'"),
        (r"(?i)\bshe\s+have\b", "subject-verb agreement: 'have' should be 'has'"),
        (r"(?i)\bthey\s+was\b", "subject-verb agreement: 'was' should be 'were'"),
        (r"(?i)\bwe\s+was\b", "subject-verb agreement: 'was' should be 'were'"),
        # 冠词错误(a/an 颠倒)
        (r"\ba\s+([aeiouAEIOU])", "article: use 'an' before vowel"),
        (r"\ban\s+([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ])",
         "article: use 'a' before consonant"),
        # 风格
        (r"\s\s+", "double space"),
    ]
    errors = []
    for pat, msg in patterns:
        for m in re.finditer(pat, text):
            errors.append({
                "rule": "simple_regex",
                "message": msg,
                "location": m.group(0),
                "suggestion": "",
            })
    return {"count": len(errors), "examples": errors[:5]}


# ===== 句子类型 =====

def sentence_complexity(sentences: list[str]) -> dict:
    """
    简易分类:
    - simple:   1 个独立子句, 无连接词
    - compound: 多个独立子句 + and/but/or/so  连接
    - complex:  含从属连词 (because/when/although/if/that/which...)
    """
    subordinate_markers = {
        "because", "although", "though", "while", "when", "if", "unless",
        "that", "which", "who", "whom", "whose", "where", "after", "before",
        "since", "until", "whether", "as", "than",
    }
    compound_markers = {"and", "but", "or", "so", "yet", "nor"}

    stats = {"simple": 0, "compound": 0, "complex": 0}
    for sent in sentences:
        tokens = re.findall(r"\b[a-zA-Z']+\b", sent.lower())
        has_sub = any(t in subordinate_markers for t in tokens)
        has_comp = any(t in compound_markers for t in tokens)

        # 数逗号粗略判断分句数
        commas = sent.count(",")

        if has_sub:
            stats["complex"] += 1
        elif has_comp or commas >= 1:
            stats["compound"] += 1
        else:
            stats["simple"] += 1

    return stats


# ===== 整合 =====

def compute_text_metrics(text: str, sentences: list[str]) -> dict:
    """一次性返回所有文本指标"""
    awl_count, awl_list = advanced_word_count(text)
    col_count, col_list = collocation_hits(text)
    grammar = grammar_error_count(text)
    sent_types = sentence_complexity(sentences)

    return {
        "ttr": round(ttr(text), 3),
        "mtld": round(mtld(text), 2),
        "advanced_word_count": awl_count,
        "advanced_words_sample": awl_list[:10],
        "collocation_hits": col_count,
        "collocations_sample": col_list[:10],
        "grammar_error_count": grammar["count"],
        "grammar_examples": grammar["examples"],
        "sentence_types": sent_types,
        "total_sentences": len(sentences),
    }