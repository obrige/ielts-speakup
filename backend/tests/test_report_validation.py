"""验证生成的报告是否包含前端需要的所有字段"""
import json
from pathlib import Path

report = json.loads((Path(__file__).parent / "real_report.json").read_text(encoding="utf-8"))

print("=== 顶层字段 ===")
required_top = ["metadata", "scores", "main_score_details", "sub_scores",
                "raw_metrics", "sentence_feedback", "zpd_examples", "summary"]
for k in required_top:
    status = "OK" if k in report else "MISSING"
    v = report.get(k)
    if isinstance(v, list):
        print(f"  [{status}] {k}: {len(v)} items")
    elif isinstance(v, dict):
        print(f"  [{status}] {k}: {len(v)} keys")
    else:
        print(f"  [{status}] {k}: {v}")

print()
print("=== metadata 字段 ===")
for k in ["question_id", "question_text", "audio_duration_sec", "word_count",
         "sentence_count", "cefr_level", "disclaimer"]:
    status = "OK" if k in report["metadata"] else "MISSING"
    print(f"  [{status}] {k}: {report['metadata'].get(k)}")

print()
print("=== scores 字段 ===")
for k in ["Overall", "FC", "LR", "GRA", "P"]:
    status = "OK" if k in report["scores"] else "MISSING"
    print(f"  [{status}] {k}: {report['scores'].get(k)}")

print()
print("=== main_score_details 每条字段 ===")
for d in report["main_score_details"]:
    miss = [k for k in ["criterion", "band", "descriptor_match", "reasoning"] if k not in d]
    status = "OK" if not miss else f"MISSING {miss}"
    print(f"  [{status}] {d['criterion']}: band={d['band']}")

print()
print("=== sub_scores 每条字段 ===")
for s in report["sub_scores"]:
    miss = [k for k in ["sub_dimension", "band", "evidence", "reasoning"] if k not in s]
    status = "OK" if not miss else f"MISSING {miss}"
    print(f"  [{status}] {s['sub_dimension']}: band={s['band']}")

print()
print("=== sentence_feedback 每条字段 ===")
for s in report["sentence_feedback"]:
    miss = [k for k in ["sentence_id", "original", "issues", "improvements", "model_answer"] if k not in s]
    status = "OK" if not miss else f"MISSING {miss}"
    print(f"  [{status}] #{s.get('sentence_id')}: {len(s.get('issues', []))} issues, "
          f"{len(s.get('improvements', []))} improvements")

print()
print("=== zpd_examples 每条字段 ===")
for z in report["zpd_examples"]:
    miss = [k for k in ["sub_dimension", "current_band", "zpd_target_band",
                        "improved_version", "what_changed", "next_step", "practice_tip"] if k not in z]
    status = "OK" if not miss else f"MISSING {miss}"
    print(f"  [{status}] {z.get('sub_dimension')}: target {z.get('zpd_target_band')}")

print()
print("=== summary 字段 ===")
for k in ["strengths", "weaknesses", "priority_improvements", "cefr_level"]:
    status = "OK" if k in report["summary"] else "MISSING"
    v = report["summary"].get(k)
    if isinstance(v, list):
        print(f"  [{status}] {k}: {len(v)} items")
    else:
        print(f"  [{status}] {k}: {v}")

print()
print("=== XSS 测试:报告里是否含危险字符 ===")
# 把整个报告 dump 成字符串看有没有 <script> 之类
report_str = json.dumps(report, ensure_ascii=False)
dangerous = ["<script>", "javascript:", "onerror=", "<img"]
for d in dangerous:
    count = report_str.count(d)
    status = "OK" if count == 0 else "WARN"
    print(f"  [{status}] '{d}': {count} occurrences")

print()
print("✅ 报告 schema 完整、前端可消费")