"""
所有 LLM prompt 模板。
每个 prompt 输出都是 JSON,通过 response_format={"type": "json_object"} 约束。
所有 prompt 的输出都是为单一任务设计,跨 prompt 不共享会话上下文。
"""
from __future__ import annotations

# IELTS Speaking Band Descriptors 官方文本
# 来源: Cambridge / IDP / British Council 公开文档
IELTS_SPEAKING_RUBRIC = """
═══════════════════════════════════════════════════════════════════
OFFICIAL IELTS SPEAKING BAND DESCRIPTORS
═══════════════════════════════════════════════════════════════════

## Fluency and Coherence (FC) — 流利度与连贯性

Band 9: speaks fluently with only rare repetition or self-correction;
        uses a wide range of connectives and discourse markers with flexibility;
        develops topics coherently and appropriately
Band 8: develops topics coherently and appropriately;
        uses a wide range of connectives and discourse markers with flexibility;
        speaks fluently with only occasional repetition or self-correction
Band 7: speaks at length without noticeable effort or loss of coherence;
        may demonstrate language-related hesitation at times;
        uses a range of connectives and discourse markers with some flexibility
Band 6: speaks at length, despite some repetition, self-correction or hesitation;
        uses a range of connectives and discourse markers, but not always appropriately
Band 5: usually maintains flow but uses simple connectives;
        may over-rely on a few basic connectives;
        repetition, self-correction and/or hesitation are noticeable
Band 4: is unable to keep going without noticeable pauses;
        uses basic connectives and repeats words/phrases frequently
Band 3: cannot respond without noticeable pauses;
        produces extended, but frequently hesitant speech;
        has limited ability to link simple sentences
Band 2: often produces fragmented speech with frequent pausing
Band 1: speech is basically unintelligible

## Lexical Resource (LR) — 词汇丰富度

Band 9: uses a wide range of vocabulary with very natural, sophisticated lexical features;
        rare minor errors occur only in unfamiliar topics
Band 8: uses a wide range of vocabulary fluently and flexibly to convey precise meanings;
        skillfully uses less common and idiomatic items
Band 7: uses a sufficient range of vocabulary to allow flexibility and precision;
        uses less common lexical items with some awareness of style and collocation;
        may make occasional errors in word choice or collocation
Band 6: uses an adequate range of vocabulary for the topic;
        attempts to use paraphrase and less common vocabulary with mixed success
Band 5: manages to talk about familiar and unfamiliar topics but uses vocabulary with limited flexibility;
        attempts paraphrase but with mixed success
Band 4: is able to talk about familiar topics but can only convey basic meaning on unfamiliar topics;
        makes frequent errors in word choice
Band 3: uses simple vocabulary to convey basic information;
        limited vocabulary means inability to express much beyond the most basic ideas
Band 2: uses very limited vocabulary; isolated words/phrases

## Grammatical Range and Accuracy (GRA) — 语法多样性与准确性

Band 9: uses a wide range of structures with full flexibility and accuracy;
        rare minor errors occur only in unfamiliar contexts
Band 8: uses a wide range of structures; majority of sentences are error-free
Band 7: uses a variety of complex structures with good control;
        produces frequent error-free sentences; has good control of grammar and punctuation
Band 6: uses a mix of simple and complex sentence forms;
        makes some errors in complex structures though these rarely cause comprehension problems
Band 5: attempts complex sentences but these tend to be less accurate than simple sentences;
        may make frequent grammatical mistakes but meaning is generally clear
Band 4: produces basic sentence forms with reasonable accuracy;
        subordinate clauses are rare; grammatical errors may cause strain
Band 3: attempts complex sentences but these contain frequent errors;
        basic sentence forms may be accurate but limited
Band 2: produces basic sentence forms but errors are frequent and may obscure meaning
Band 1: shows almost no grammatical control

## Pronunciation (P) — 发音

Band 9: uses a wide range of pronunciation features with precision and subtlety;
        sustains flexible use of features throughout; can be understood effortlessly
Band 7: shows all the positive features of Band 6, and some of Band 8
Band 5: is usually intelligible; some problems with pronunciation may cause difficulty;
        attempts to use a range of pronunciation features
Band 3: has some difficulty being understood; pronunciation causes some difficulty

═══════════════════════════════════════════════════════════════════
SCORING RULES (MUST FOLLOW):
- Scores in increments of 0.5 only (e.g. 5.0, 5.5, 6.0, 6.5)
- Match the candidate's performance to descriptor text
- Half-band scores (e.g. 6.5) indicate performance between two full bands
- Do NOT score above 7.0 unless the candidate clearly demonstrates ALL features of the next full band
═══════════════════════════════════════════════════════════════════
"""


SYSTEM_PROMPT_BASE = """You are an expert IELTS Speaking examiner certified to band 9 standard.
You MUST score strictly according to the OFFICIAL band descriptors provided.
Do NOT use general impression — match the candidate's performance to descriptor text exactly.

Return ONLY valid JSON. No markdown fences, no preamble, no commentary.
Scores: float with 1 decimal place, range 1.0-9.0, increments of 0.5 only.
"""


# ===== 14 sub-dimension prompts =====

SUB_DIMS = {
    # FC
    "FC-SpeechRate": {
        "definition": "Pace of delivery: sustained speaking speed without rushing or unnatural pauses. Measured in words per minute.",
        "metrics": ["speech_rate_wpm", "duration_sec"],
        "main_dim": "FC",
    },
    "FC-PausingWordSearch": {
        "definition": "Frequency and length of pauses, use of fillers (um/uh/like). Reflects ability to maintain flow.",
        "metrics": ["pause_count", "pause_total", "fillers"],
        "main_dim": "FC",
    },
    "FC-LogicalStructure": {
        "definition": "Organization of ideas, logical sequencing of arguments, clear beginning-middle-end structure.",
        "metrics": [],
        "main_dim": "FC",
    },
    "FC-CohesiveDevices": {
        "definition": "Use of connectives (and/but/because/however/...), referencing, and discourse markers.",
        "metrics": [],
        "main_dim": "FC",
    },
    # LR
    "LR-LexicalRichness": {
        "definition": "Vocabulary range and variety. Measured by TTR (type-token ratio) and MTLD (mean textual lexical diversity).",
        "metrics": ["ttr", "mtld", "advanced_word_count"],
        "main_dim": "LR",
    },
    "LR-Collocation": {
        "definition": "Appropriate and natural word combinations (verb+noun, adj+noun, etc.).",
        "metrics": ["collocation_hits"],
        "main_dim": "LR",
    },
    "LR-Paraphrasing": {
        "definition": "Ability to rephrase and avoid repetition.",
        "metrics": [],
        "main_dim": "LR",
    },
    "LR-IdiomaticExpression": {
        "definition": "Use of natural, colloquial language and idioms appropriate to context.",
        "metrics": ["advanced_words_sample"],
        "main_dim": "LR",
    },
    # GRA
    "GRA-GrammaticalStructure": {
        "definition": "Range and complexity of sentence types (simple/compound/complex).",
        "metrics": ["sentence_types"],
        "main_dim": "GRA",
    },
    "GRA-GrammaticalAccuracy": {
        "definition": "Accuracy of grammar in complex structures; error density per sentence.",
        "metrics": ["grammar_error_count", "total_sentences"],
        "main_dim": "GRA",
    },
    "GRA-GrammaticalErrors": {
        "definition": "Frequency and severity of grammar errors, particularly in complex structures.",
        "metrics": ["grammar_error_count", "grammar_examples"],
        "main_dim": "GRA",
    },
    # P
    "P-Intelligibility": {
        "definition": "Ease of understanding for the listener. Anchored by wav2vec2 CTC confidence.",
        "metrics": ["wav2vec2_confidence"],
        "main_dim": "P",
    },
    "P-Clarity": {
        "definition": "Accurate pronunciation of individual sounds. Anchored by phoneme edit distance.",
        "metrics": ["phoneme_edit_distance"],
        "main_dim": "P",
    },
    "P-StressIntonationRhythm": {
        "definition": "Word stress, sentence intonation, and overall rhythm/prosody.",
        "metrics": ["f0_std", "rms_var", "rhythm_cv"],
        "main_dim": "P",
    },
}


def sub_dim_prompt(
    sub_dim_name: str,
    question: str,
    text: str,
    relevant_metrics: dict,
) -> str:
    """生成单个 sub-dim 的 prompt"""
    info = SUB_DIMS[sub_dim_name]
    metrics_str = "\n".join(
        f"- {k}: {relevant_metrics.get(k, 'N/A')}"
        for k in info["metrics"]
    ) if info["metrics"] else "(no objective metrics — judge purely from transcript)"

    return f"""# Sub-dimension: {sub_dim_name}
{info['definition']}

# IELTS FC / LR / GRA / P Descriptors (Official)
{IELTS_SPEAKING_RUBRIC}

# Question
{question}

# Candidate's Transcript
"{text}"

# Relevant Metrics (informational only — not hard thresholds)
{metrics_str}

# Your Task
1. Compare the candidate's performance to the {info['main_dim']} descriptor that best matches
2. Pick the band in 0.5 increments (1.0-9.0)
3. MUST cite specific transcript evidence (quote actual words or sentence fragments)
4. Reasoning must explicitly link evidence to the descriptor

# Output JSON (ONLY this object)
{{
  "sub_dimension": "{sub_dim_name}",
  "band": <float 0.5 increments>,
  "evidence": "<quote 1-3 specific phrases from the transcript>",
  "reasoning": "<1-2 sentence justification linking evidence to {info['main_dim']} descriptor>"
}}
"""


# ===== 4 main-dimension prompts =====

MAIN_DIMS = ["FC", "LR", "GRA", "P"]
MAIN_DIM_FULL_NAMES = {
    "FC":  "Fluency and Coherence",
    "LR":  "Lexical Resource",
    "GRA": "Grammatical Range and Accuracy",
    "P":   "Pronunciation",
}


def main_dim_prompt(
    main_dim: str,
    question: str,
    text: str,
    all_metrics: dict,
    sub_dim_evidence: dict,
) -> str:
    """
    sub_dim_evidence: {sub_dim_name: {band, evidence, reasoning}, ...}
    只传该 main_dim 对应的几个 sub-dim
    """
    full_name = MAIN_DIM_FULL_NAMES[main_dim]
    sub_evidence_str = "\n\n".join(
        f"### {name}\n- Band: {data['band']}\n- Evidence: {data['evidence']}\n- Reasoning: {data['reasoning']}"
        for name, data in sub_dim_evidence.items()
    )

    return f"""# Criterion: {main_dim} ({full_name})

This is an INDEPENDENT scoring session. Your judgment must be your own,
based on:
1. Official {full_name} descriptor below
2. The candidate's transcript
3. All raw metrics
4. Sub-dimension scores AND their explanations (treat as EVIDENCE, not as the answer)

# Official {full_name} Descriptor
{IELTS_SPEAKING_RUBRIC}

# Question
{question}

# Candidate's Transcript
"{text}"

# All Raw Metrics
{_format_metrics(all_metrics)}

# Sub-dimension Scores & Explanations (for {main_dim})
{sub_evidence_str}

# Your Task
1. Independently judge {main_dim} band (0.5 increments, 1.0-9.0)
2. Cross-check against sub-dimension explanations — they are EVIDENCE
3. If a sub-dim explanation contradicts your judgment, prefer the rubric
4. Output JSON

# Output
{{
  "criterion": "{main_dim}",
  "band": <float 0.5 increments>,
  "descriptor_match": "<quote the specific descriptor text that best matches the candidate>",
  "reasoning": "<2-3 sentences; if your band differs from sub-dim average, explain why>"
}}
"""


def _format_metrics(metrics: dict) -> str:
    """格式化指标为可读字符串"""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, dict):
            parts.append(f"- {k}:")
            for kk, vv in v.items():
                parts.append(f"    - {kk}: {vv}")
        elif isinstance(v, list):
            parts.append(f"- {k}: {v[:5]}{'...' if len(v) > 5 else ''}")
        else:
            parts.append(f"- {k}: {v}")
    return "\n".join(parts)


# ===== Sentence-level feedback =====

SENTENCE_FEEDBACK_PROMPT = """You are providing sentence-level IELTS Speaking feedback.

# Sentence #{sent_id} (audio {start:.1f}s - {end:.1f}s)
"{sentence_text}"

# Context (full transcript for reference)
"{full_transcript}"

# This sentence's audio analysis
- wav2vec2 confidence: {wav2vec2_conf:.2f}  (lower = less intelligible pronunciation)
- Pauses before/after: {pause_before:.2f}s / {pause_after:.2f}s
- Filler words in this sentence: {fillers_in_sent}
- Grammar errors detected: {grammar_errors}

# Main dimension this sentence most affects: {dominant_dim}
# Sub-dimensions this sentence most affects: {affected_sub_dims}

# Your Task
Identify issues and provide improvements. Be CONCISE: improvements must be 1-2 sentences each.

# Output JSON
{{
  "sentence_id": {sent_id},
  "original": "<verbatim>",
  "issues": [
    {{
      "type": "grammar|vocabulary|pronunciation|fluency",
      "description": "<what's wrong>",
      "location": "<exact phrase>",
      "severity": "minor|moderate|major"
    }}
  ],
  "improvements": [
    {{
      "type": "grammar|vocabulary|fluency",
      "original": "<exact phrase>",
      "suggested": "<better version, 1 sentence>",
      "reason": "<1 sentence>"
    }}
  ],
  "model_answer": "<an ideal version of this sentence at Band 7-8, SAME length or shorter>"
}}
"""


# ===== ZPD personalized examples =====

LENGTH_POLICY = {
    "P-Intelligibility":        0,  # 发音不重写
    "P-Clarity":                0,
    "P-StressIntonationRhythm": 0,
    # 其他:+10
    "FC-SpeechRate":            10,
    "FC-PausingWordSearch":     10,
    "FC-LogicalStructure":      10,
    "FC-CohesiveDevices":       10,
    "LR-LexicalRichness":       10,
    "LR-Collocation":           10,
    "LR-Paraphrasing":          10,
    "LR-IdiomaticExpression":   10,
    "GRA-GrammaticalStructure": 10,
    "GRA-GrammaticalAccuracy":  10,
    "GRA-GrammaticalErrors":    10,
}


def compute_zpd_target(sub_dim_band: float, sub_dim_name: str) -> float:
    delta = 1.0 if sub_dim_name not in {
        "FC-SpeechRate", "FC-PausingWordSearch", "FC-LogicalStructure",
        "P-Intelligibility", "P-Clarity", "P-StressIntonationRhythm",
        "LR-Paraphrasing", "LR-IdiomaticExpression",
    } else 0.5
    return round((sub_dim_band + delta) * 2) / 2


ZPD_PROMPT = """You are generating a ZPD-calibrated practice example for an IELTS learner.

# Zone of Proximal Development (Vygotsky)
- Target band = current + 0.5 to +1.0 (challenging but achievable)
- Do NOT generate Band 9 sentences for a Band 5 learner (too far)
- Do NOT generate simple rewrites (no learning value)
- Sweet spot: 1-2 new vocab items, 1 new structure the learner could master

# Learner Profile
- Overall band: {overall_band}
- {main_dim} band: {main_dim_band}
- Affected sub-dimension: {sub_dim_name}: {sub_dim_band}
- ZPD target band: {target_band}

# Original Learner Sentence ({learner_words} words)
"{learner_sentence}"

# Identified Issues
{issues_json}

# LENGTH CONSTRAINT (HARD)
- Output "improved_version": {learner_words} to {learner_words} + {length_extra} words
- Output "harder_version": same range
- If your draft is longer, rewrite TIGHTER before outputting
- Reason: a Band 9 25-word sentence is overwhelming for a Band 5 learner
- Goal: minimal change that maximizes improvement

# Output JSON
{{
  "sub_dimension": "{sub_dim_name}",
  "current_band": {sub_dim_band},
  "zpd_target_band": {target_band},
  "improved_version": "<sentence at ZPD target, within length range>",
  "what_changed": [
    {{
      "type": "vocab|grammar|cohesion|fluency",
      "from": "<learner's choice>",
      "to": "<better choice>",
      "why": "<1 sentence>"
    }}
  ],
  "next_step": {{
    "harder_version": "<sentence at band +1.5, same length range>",
    "challenge": "<what the learner needs to learn to attempt this>"
  }},
  "practice_tip": "<one concrete suggestion: e.g. 'drill this connector 10x' or 'memorize this collocation'>"
}}
"""


# Pronunciation-specific ZPD (不重写句子, 改给发音指引)
P_ZPD_PROMPT = """You are giving pronunciation feedback for an IELTS learner.
DO NOT rewrite the sentence. Provide pronunciation guidance.

# Learner said (verbatim, {learner_words} words)
"{learner_sentence}"

# Audio Analysis
- wav2vec2 confidence: {conf:.2f} (lower = less intelligible)
- F0 variance: {f0_std:.1f} Hz (higher = more expressive intonation)
- Energy (RMS) variance: {rms_var:.3f} (higher = more stress variation)

# Your Task
Provide concise pronunciation guidance.

# Output JSON
{{
  "learner_sentence": "<verbatim>",
  "problem_phonemes": [
    {{"sound": "<IPA>", "word": "<word in context>", "tip": "<how to pronounce>"}}
  ],
  "stress_pattern": "<which syllable or word should be emphasized>",
  "intonation_note": "<rising/falling/flat — what's natural here>",
  "drill_sentence": "<3-8 word minimal pair or short sentence for practice>",
  "practice_tip": "<one concrete tip>"
}}
"""


# ===== Band 量化与校验 =====

def normalize_band(value) -> float | None:
    """
    把任意输入量化成合法的 IELTS band (1.0-9.0, 0.5 增量)。

    - 接受 int / float / str("6.5") / None
    - 超出范围截断
    - 量化到最近的 0.5(0.25/0.75 → 0.5/1.0)
    - 无法转换返回 None
    """
    try:
        if value is None:
            return None
        v = float(value)
    except (TypeError, ValueError):
        return None

    if v != v:  # NaN
        return None

    # 截断到合法区间
    v = max(1.0, min(9.0, v))

    # 量化到 0.5
    return round(v * 2) / 2


# ===== Summary =====

SUMMARY_PROMPT = """Based on the following IELTS Speaking assessment, generate a concise summary report.

# Main Scores
{main_scores_json}

# Sub Scores
{sub_scores_json}

# Overall: {overall_band}

# Output JSON
{{
  "strengths": ["<3-5 bullets, each ≤15 words>"],
  "weaknesses": ["<3-5 bullets, each ≤15 words>"],
  "priority_improvements": ["<3 actionable items, each ≤15 words>"],
  "cefr_level": "<A1|A2|B1|B2|C1|C2>"
}}
"""