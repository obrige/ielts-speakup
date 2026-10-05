"""
LLM 调用层
- 双模型并行(DeepSeek 主, Qwen 兜底)
- 全部 response_format=json_object
- 失败重试, 主备切换
"""
from __future__ import annotations
import os
import json
import asyncio
import logging

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm-scorer")

# 延迟初始化(新版 openai SDK 不允许空 key)
LLM_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "45"))
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-plus")

_DEEPSEEK_CLIENT = None
_QWEN_CLIENT = None


def _get_deepseek_client() -> AsyncOpenAI | None:
    global _DEEPSEEK_CLIENT
    if _DEEPSEEK_CLIENT is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return None
        _DEEPSEEK_CLIENT = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            timeout=LLM_TIMEOUT_SEC,
        )
    return _DEEPSEEK_CLIENT


def _get_qwen_client() -> AsyncOpenAI | None:
    global _QWEN_CLIENT
    if _QWEN_CLIENT is None:
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            return None
        _QWEN_CLIENT = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            timeout=LLM_TIMEOUT_SEC,
        )
    return _QWEN_CLIENT


# 启动时校验关键 API Key(避免运行期才发现)
if not os.getenv("DEEPSEEK_API_KEY"):
    logger.warning("DEEPSEEK_API_KEY 未配置,LLM 调用将全部失败")
if not os.getenv("QWEN_API_KEY"):
    logger.warning("QWEN_API_KEY 未配置,主模型失败时无法降级")


async def call_json(
    prompt: str,
    system: str,
    max_tokens: int = 600,
    temperature: float = 0.2,
    primary: str = "deepseek",
) -> dict | None:
    """
    调用 LLM, 返回解析后的 dict。
    失败重试 3 次,主备切换。
    """
    clients = [
        (_get_deepseek_client(), DEEPSEEK_MODEL, "deepseek"),
        (_get_qwen_client(), QWEN_MODEL, "qwen"),
    ] if primary == "deepseek" else [
        (_get_qwen_client(), QWEN_MODEL, "qwen"),
        (_get_deepseek_client(), DEEPSEEK_MODEL, "deepseek"),
    ]

    last_err = None
    for client, model, name in clients:
        if client is None:
            continue  # API Key 未配置,跳过此 client
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content
                result = json.loads(content)
                logger.debug(f"LLM [{name}] OK, attempt {attempt + 1}")
                return result
            except json.JSONDecodeError as e:
                # JSON 错误是 prompt 问题,重试同一模型无用,立即跳备模型
                last_err = f"[{name}] JSON decode error: {e}"
                logger.warning(last_err)
                break
            except Exception as e:
                # 网络/限流等:重试可能有用
                last_err = f"[{name}] {type(e).__name__}: {e}"
                logger.warning(f"{last_err} (attempt {attempt + 1}/3)")
                if attempt < 2:
                    await asyncio.sleep(1.0)

    logger.error(f"All LLMs failed: {last_err}")
    return None


async def gather_json(
    prompts_with_meta: list[dict],
    system: str,
    max_tokens: int = 600,
    concurrency: int = 8,
) -> list[dict | None]:
    """
    并发调用多个 prompt。
    prompts_with_meta: [{prompt, max_tokens?, primary?}, ...]
    返回与输入等长的列表(失败位置为 None)。
    """
    sem = asyncio.Semaphore(concurrency)

    async def run_one(meta: dict) -> dict | None:
        async with sem:
            return await call_json(
                prompt=meta["prompt"],
                system=system,
                max_tokens=meta.get("max_tokens", max_tokens),
                primary=meta.get("primary", "deepseek"),
            )

    tasks = [run_one(m) for m in prompts_with_meta]
    return await asyncio.gather(*tasks)