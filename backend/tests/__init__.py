"""
测试套件 - 不依赖 LLM/WhisperX 的端到端验证

可单独跑某个测试:
    cd backend
    py -m tests.test_basics
    py -m tests.test_text_stats
    py -m tests.test_grammar
    py -m tests.test_llm             # 需要 API key
    py -m tests.test_pipeline_e2e    # 需要 API key
    py -m tests.test_full_e2e        # 需要 API key + edge-tts(首次生成音频)
    py -m tests.test_api_errors      # 需要 backend 跑在 :8080
    py -m tests.test_report_validation  # 需要先跑 test_full_e2e
"""

__version__ = "1.0.0"
__all__ = [
    "test_basics",
    "test_text_stats",
    "test_grammar",
    "test_llm",
    "test_pipeline_e2e",
    "test_full_e2e",
    "test_api_errors",
    "test_report_validation",
]