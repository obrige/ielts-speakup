"""
完整评分 pipeline 编排。

流程:
1. 算法层(WhisperX + librosa + wav2vec2 + text_stats)
2. 14 次 sub-dim LLM
3. 4 次 main-dim LLM
4. N 次 sentence feedback LLM
5. M 次 ZPD LLM
6. 1 次 summary LLM
7. Overall = 算法
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
from pathlib import Path

from .whisperx_client import transcribe
from .audio_metrics import (
    load_audio,
    compute_dsp_metrics,
    compute_pronunciation_metrics,
    split_into_sentences,
    count_fillers,
)
from .text_stats import compute_text_metrics
from . import prompts
from .llm_scorer import call_json, gather_json
from .questions import get_question

logger = logging.getLogger("pipeline")


class ScoringPipeline:
    def __init__(self):
        self.ready = False

    def _estimate_duration(self, text_data: dict, audio_path: Path) -> float:
        """估算录音时长:优先用 WhisperX segments,否则兜底用 librosa"""
        segments = text_data.get("segments", [])
        if segments:
            # 取最后一段的 end 时间
            last_end = max(
                (float(seg.get("end", 0)) for seg in segments),
                default=0.0,
            )
            if last_end > 0:
                return last_end
        # 兜底:用 librosa 加载计算
        try:
            from .audio_metrics import load_audio
            y, sr = load_audio(audio_path)
            return len(y) / sr
        except Exception:
            return 0.0

    async def init(self):
        """预热:加载 wav2vec2 等大模型"""
        try:
            # 预加载 wav2vec2
            from .audio_metrics import _ensure_wav2vec2
            _ensure_wav2vec2()
            self.ready = True
            logger.info("Pipeline ready")
        except Exception as e:
            logger.warning(f"Init incomplete: {e}")
            self.ready = True  # 即使 wav2vec2 加载失败也允许启动(降级)

    async def score_full(self, audio_path: Path, question_id: str) -> dict:
        """完整评分"""
        question = get_question(question_id)
        text_data = await transcribe(audio_path)
        text = text_data["text"].strip()
        word_times = text_data.get("word_segments", [])

        # 边界:转写为空(可能是纯静音或麦克风故障)
        if not text:
            raise ValueError(
                "未能识别到任何语音。请检查麦克风权限并重新录制。"
            )

        # 边界:录音过短(< 3 秒通常没意义)
        duration_sec = self._estimate_duration(text_data, audio_path)
        if duration_sec < 3.0:
            raise ValueError(
                f"录音时长过短({duration_sec:.1f}s),至少需要 3 秒才能评分。"
            )

        # ===== 算法层 =====
        y, sr = load_audio(audio_path)
        actual_duration_sec = len(y) / sr
        # 用实际音频时长(优先) 或 ASR 推断时长
        duration_sec = max(actual_duration_sec, duration_sec)

        dsp = compute_dsp_metrics(y, sr, word_times, duration_sec)
        pron = compute_pronunciation_metrics(y, sr, text)
        sentences = split_into_sentences(text, word_times, duration_sec)
        text_stats = compute_text_metrics(text, [s["text"] for s in sentences])

        all_metrics = {
            **dsp,
            **pron,
            **text_stats,
            "fillers": count_fillers(text),
        }

        logger.info(
            f"Algo done. text={len(text.split())}w, "
            f"sentences={len(sentences)}, "
            f"wpm={dsp['speech_rate_wpm']}, "
            f"grammar_errors={text_stats['grammar_error_count']}"
        )

        # ===== 14 sub-dim LLM =====
        sub_results = await self._score_sub_dims(question, text, all_metrics)

        # 收集成功的 sub-dim 结果
        sub_map: dict[str, dict] = {}
        for r in sub_results:
            if r and "sub_dimension" in r and "band" in r:
                # 校验 + 量化 band
                normalized = prompts.normalize_band(r["band"])
                if normalized is not None:
                    r["band"] = normalized
                    sub_map[r["sub_dimension"]] = r

        logger.info(f"Sub-dim scored: {len(sub_map)}/14")

        # ===== 4 main-dim LLM =====
        main_results = await self._score_main_dims(
            question, text, all_metrics, sub_map
        )
        main_map: dict[str, dict] = {}
        for r in main_results:
            if r and "criterion" in r and "band" in r:
                normalized = prompts.normalize_band(r["band"])
                if normalized is not None:
                    r["band"] = normalized
                    main_map[r["criterion"]] = r

        logger.info(f"Main-dim scored: {list(main_map.keys())}")

        # ===== Overall = 算法 =====
        if len(main_map) == 4:
            bands = [r["band"] for r in main_map.values()]
            # 确保都是合法值
            if all(isinstance(b, (int, float)) for b in bands):
                overall = round(sum(bands) / 4 * 2) / 2
            else:
                overall = None
        else:
            overall = None

        # ===== Sentence feedback (每句) =====
        sent_feedback = await self._score_sentences(
            sentences, text, all_metrics
        )

        # ===== ZPD examples (per issue) =====
        zpd_examples = await self._generate_zpd(
            sent_feedback, sub_map, main_map, overall, all_metrics
        )

        # ===== Summary =====
        summary = await self._generate_summary(main_map, sub_map, overall)

        # ===== Final report =====
        return {
            "metadata": {
                "question_id": question["id"],
                "question_text": question["prompt"],
                "audio_duration_sec": round(duration_sec, 2),
                "word_count": len(text.split()),
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
            "sentence_feedback": sent_feedback,
            "zpd_examples": zpd_examples,
            "summary": summary,
        }

    async def _score_sub_dims(self, question, text, all_metrics) -> list[dict | None]:
        """14 次独立 sub-dim LLM 调用"""
        meta_list = []
        for name, info in prompts.SUB_DIMS.items():
            relevant = {
                k: all_metrics.get(k) for k in info["metrics"]
            }
            prompt = prompts.sub_dim_prompt(
                sub_dim_name=name,
                question=question["prompt"],
                text=text,
                relevant_metrics=relevant,
            )
            meta_list.append({"prompt": prompt, "max_tokens": 400})

        return await gather_json(
            meta_list,
            system=prompts.SYSTEM_PROMPT_BASE,
            max_tokens=400,
            concurrency=10,
        )

    async def _score_main_dims(self, question, text, all_metrics, sub_map) -> list[dict | None]:
        """4 次独立 main-dim LLM 调用"""
        # 按 main_dim 分组 sub-dim
        groups: dict[str, dict] = {dim: {} for dim in prompts.MAIN_DIMS}
        for name, data in sub_map.items():
            dim = prompts.SUB_DIMS[name]["main_dim"]
            groups[dim][name] = data

        meta_list = []
        for dim in prompts.MAIN_DIMS:
            prompt = prompts.main_dim_prompt(
                main_dim=dim,
                question=question["prompt"],
                text=text,
                all_metrics=all_metrics,
                sub_dim_evidence=groups[dim],
            )
            meta_list.append({"prompt": prompt, "max_tokens": 500})

        return await gather_json(
            meta_list,
            system=prompts.SYSTEM_PROMPT_BASE,
            max_tokens=500,
            concurrency=4,
        )

    async def _score_sentences(self, sentences, full_text, all_metrics) -> list[dict]:
        """每句一次 LLM 反馈"""
        if not sentences:
            return []

        meta_list = []
        for sent in sentences:
            # 取该句的 wav2vec2 平均置信度
            word_indices = sent.get("word_indices", [])
            conf = all_metrics.get("wav2vec2_confidence", 0.5)

            # 取该句的 grammar errors
            grammar_errors = self._extract_sentence_grammar(
                sent["text"], all_metrics
            )

            # 推断主要受影响的维度和子维度
            dominant_dim, affected_subs = self._infer_affected_dims(
                sent["text"], all_metrics
            )

            prompt = prompts.SENTENCE_FEEDBACK_PROMPT.format(
                sent_id=sent["sent_id"],
                sentence_text=sent["text"],
                start=sent["start"],
                end=sent["end"],
                full_transcript=full_text[:1500],
                wav2vec2_conf=conf,
                pause_before=0.0,  # 简化:不细化到每句前后
                pause_after=0.0,
                fillers_in_sent=[],
                grammar_errors=json.dumps(grammar_errors, ensure_ascii=False),
                dominant_dim=dominant_dim,
                affected_sub_dims=", ".join(affected_subs),
            )
            meta_list.append({"prompt": prompt, "max_tokens": 600})

        results = await gather_json(
            meta_list,
            system=prompts.SYSTEM_PROMPT_BASE,
            max_tokens=600,
            concurrency=10,
        )

        return [r for r in results if r is not None]

    def _extract_sentence_grammar(self, sent_text: str, all_metrics: dict) -> list[dict]:
        """从全局 grammar examples 里筛出属于该句的"""
        examples = all_metrics.get("grammar_examples", [])
        sent_lower = sent_text.lower()
        matched = []
        for ex in examples:
            if ex.get("location", "").lower() in sent_lower:
                matched.append(ex)
        if not matched and examples:
            matched = examples[:1]  # 至少给一个
        return matched

    def _infer_affected_dims(self, sent_text: str, all_metrics: dict) -> tuple[str, list[str]]:
        """简单推断该句主要受哪些维度影响"""
        text_lower = sent_text.lower()
        affected = []
        dominant = "FC"

        # 检测连接词 -> CohesiveDevices
        if re.search(r"\b(and|but|because|however|therefore|moreover|furthermore)\b", text_lower):
            affected.append("FC-CohesiveDevices")
            dominant = "FC"

        # 检测高级词 -> LexicalRichness / IdiomaticExpression
        adv = all_metrics.get("advanced_words_sample", [])
        if any(w in text_lower for w in adv):
            affected.append("LR-LexicalRichness")
            dominant = "LR"

        # 检测复杂结构 (有 that/which/who 从句)
        if re.search(r"\b(that|which|who|because|although)\b", text_lower):
            affected.append("GRA-GrammaticalStructure")
            dominant = "GRA"

        # 检测长难句 (词数 > 20)
        if len(sent_text.split()) > 20:
            affected.append("GRA-GrammaticalStructure")

        if not affected:
            affected = ["FC-SpeechRate"]

        return dominant, affected

    async def _generate_zpd(
        self, sent_feedback, sub_map, main_map, overall_band, all_metrics
    ) -> list[dict]:
        """M 次 ZPD LLM, 每个 issue 一次"""
        meta_list = []
        item_map = []  # 记录哪个 prompt 对应哪个 item

        for sent in sent_feedback:
            for issue in sent.get("issues", []):
                if issue.get("severity") == "minor":
                    continue  # minor 问题不出 ZPD

                # 找到相关 sub-dim
                affected_sub = self._map_issue_to_sub_dim(issue, sent)
                if affected_sub not in sub_map:
                    continue

                sub_band = sub_map[affected_sub]["band"]
                target = prompts.compute_zpd_target(sub_band, affected_sub)
                length_extra = prompts.LENGTH_POLICY.get(affected_sub, 10)
                main_dim = prompts.SUB_DIMS[affected_sub]["main_dim"]
                main_band = main_map.get(main_dim, {}).get("band", sub_band)

                # 发音子维度走专门 prompt
                if affected_sub.startswith("P-"):
                    prompt = prompts.P_ZPD_PROMPT.format(
                        learner_sentence=sent["original"],
                        learner_words=len(sent["original"].split()),
                        conf=all_metrics.get("wav2vec2_confidence", 0.7),
                        f0_std=all_metrics.get("f0_std", 50.0),
                        rms_var=all_metrics.get("rms_var", 0.5),
                    )
                    meta_list.append({"prompt": prompt, "max_tokens": 400})
                    item_map.append(("P", sent, issue, affected_sub))
                else:
                    prompt = prompts.ZPD_PROMPT.format(
                        overall_band=overall_band or 0.0,
                        main_dim=main_dim,
                        main_dim_band=main_band,
                        sub_dim_name=affected_sub,
                        sub_dim_band=sub_band,
                        target_band=target,
                        learner_sentence=sent["original"],
                        learner_words=len(sent["original"].split()),
                        length_extra=length_extra,
                        issues_json=json.dumps(issue, ensure_ascii=False),
                    )
                    meta_list.append({"prompt": prompt, "max_tokens": 500})
                    item_map.append(("ZPD", sent, issue, affected_sub))

        if not meta_list:
            return []

        results = await gather_json(
            meta_list,
            system=prompts.SYSTEM_PROMPT_BASE,
            max_tokens=500,
            concurrency=10,
        )

        zpd_out = []
        for (kind, sent, issue, sub), r in zip(item_map, results):
            if r is None:
                continue
            r["sentence_id_ref"] = sent.get("sentence_id")
            r["issue_type"] = issue.get("type")
            zpd_out.append(r)

        return zpd_out

    def _map_issue_to_sub_dim(self, issue: dict, sent: dict) -> str:
        """根据 issue type 映射到对应 sub-dim"""
        t = issue.get("type", "")
        if t == "grammar":
            return "GRA-GrammaticalErrors"
        if t == "vocabulary":
            return "LR-LexicalRichness"
        if t == "fluency":
            return "FC-PausingWordSearch"
        if t == "pronunciation":
            return "P-Clarity"
        # 默认
        return "FC-LogicalStructure"

    async def _generate_summary(self, main_map, sub_map, overall_band) -> dict | None:
        """1 次 summary LLM"""
        if not main_map or overall_band is None:
            return None

        prompt = prompts.SUMMARY_PROMPT.format(
            main_scores_json=json.dumps(
                [{"criterion": k, "band": v["band"]} for k, v in main_map.items()],
                ensure_ascii=False, indent=2
            ),
            sub_scores_json=json.dumps(
                [{"sub": k, "band": v["band"]} for k, v in sub_map.items()],
                ensure_ascii=False, indent=2
            ),
            overall_band=overall_band,
        )
        return await call_json(
            prompt=prompt,
            system=prompts.SYSTEM_PROMPT_BASE,
            max_tokens=500,
        )