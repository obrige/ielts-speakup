"""
Master test runner - 跑全部测试
"""
import subprocess
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
tests = [
    "tests.test_basics",
    "tests.test_text_stats",
    "tests.test_grammar",
    "tests.test_llm",
    "tests.test_pipeline_e2e",
    "tests.test_full_e2e",
    "tests.test_report_validation",
    # test_api_errors 需要 backend server 在 :8080,需手动跑
]

failed = []
for t in tests:
    print(f"\n{'='*60}\n  {t}\n{'='*60}")
    r = subprocess.run(
        [sys.executable, "-m", t],
        cwd=backend_dir,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  ❌ FAILED")
        # 打印最后 30 行
        for line in r.stdout.split("\n")[-30:]:
            print(f"    {line}")
        for line in r.stderr.split("\n")[-15:]:
            print(f"    ERR: {line}")
        failed.append(t)
    else:
        print(f"  ✅ PASSED")

print(f"\n{'='*60}")
print(f"Total: {len(tests) - len(failed)}/{len(tests)} passed")
if failed:
    print(f"Failed: {failed}")
    sys.exit(1)