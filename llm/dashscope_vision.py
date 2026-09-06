# -*- coding: utf-8 -*-
"""DashScope Qwen-VL 真视觉实现（OpenAI 兼容 /chat/completions，httpx 直连）。

和 AnthropicVision 同一语义：图 → 一句"用户最可能要找的数码商品需求"，并入检索 query。
为什么在百炼路线上用 Qwen-VL 而不是 Claude 视觉：
- 本机没有官方 Claude key（ANTHROPIC_API_KEY 实为 DeepSeek 兼容端点，无图片能力）；
- 用户已有一把 DashScope key（DASHSCOPE_API_KEY）—— 同一把 key 同时供语义 embedding
  （text-embedding-v3）与视觉（Qwen-VL），国内直连、无需海外卡。

协议差异（诚实备注）：DashScope 走 **OpenAI 兼容**（content 里 image_url=data URI），
不是 Anthropic Messages 的 image/source/base64 —— 所以是独立 provider 分支，不是把
AnthropicVision 的模型名换掉就行。

边界与 AnthropicVision 一致：只做"图 → 一句需求文本"，不宣称真 OCR/多轮；key 只从
环境变量读（DASHSCOPE_API_KEY），缺 key 在 describe 时才抛（不误伤 build/启动）。
"""
from __future__ import annotations

import base64
import os

import httpx

from config.settings import Settings
from llm.vision import VisionExtraction

_DASHSCOPE_COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _guess_image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:12] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"  # 缺省按 png 发（OpenAI 兼容端点非识别格式走服务端报错）


class DashScopeQwenVision:
    """Qwen-VL 视觉：图片 → 一句用户购物需求（并入检索 query 的文本兜底）。"""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.calls: int = 0

    async def describe(self, image_bytes: bytes) -> VisionExtraction:
        self.calls += 1
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "vision.provider=dashscope(Qwen-VL) 但环境变量 DASHSCOPE_API_KEY 未设置。\n"
                "解决办法：复制 .env.example 为 .env 填入阿里云百炼 key；或保持 vision.provider=mock。"
            )
        b64 = base64.b64encode(image_bytes).decode("ascii")
        mime = _guess_image_mime(image_bytes)
        payload = {
            "model": self._s.vision.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "你是电商智能导购的图片理解模块。看图后输出：用户最可能在找的"
                                "数码商品需求，一句中文短句（含品类/核心诉求即可），不要价格、"
                                "不要下单动作、不要反问。只输出这一句。"
                            ),
                        },
                    ],
                }
            ],
            "max_tokens": 120,
        }
        async with httpx.AsyncClient(timeout=self._s.llm.timeout_sec) as client:
            resp = await client.post(
                f"{_DASHSCOPE_COMPAT_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "content-type": "application/json",
                },
                json=payload,
            )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope Qwen-VL HTTP {resp.status_code}: {resp.text[:200]}")
        content = resp.json()["choices"][0]["message"]["content"]
        return VisionExtraction(text=str(content).strip())
