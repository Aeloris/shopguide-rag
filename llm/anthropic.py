# -*- coding: utf-8 -*-
"""Anthropic/Claude 真实现：视觉抽取 + LLM 决定器（都走 Messages API，httpx 直连）。

诚实边界（与 Mock 同一条线，避免"真 LLM 反而更会瞎编"）：
1. **不可答拒绝仍是代码硬规则**：交易/实时行情/未来/真机实测，先于模型判定，任何
   provider 都无权放行 —— 见 core.agent.decision.classify_refusal。
2. **LLM 只决定"下一步调哪个工具"，不算任何数**：compare/filter 的参数（比哪些、
   预算多少）仍由确定性代码从 query 推导 —— 模型给不了错误算术。
3. **接不住就安全回落**：网络错/解析失败/选了不允许的动作 → 回落到 MockDecision 的
   确定性路径，绝不因 LLM 掉线让工具循环卡死或编造。
4. key/base_url 只从环境变量读：ANTHROPIC_API_KEY（key）、ANTHROPIC_BASE_URL（可选，
   走代理/中转时设置）。key 值永不打印。
"""
from __future__ import annotations

import base64
import os

import httpx

from config.settings import Settings
from core.agent.decision import MockDecision, classify_refusal, find_out_of_stock_phone
from core.agent.schemas import AgentDecision
from core.agent.tools import _category_from_query, parse_budget_cny
from llm.vision import VisionExtraction

_ANTHROPIC_DEFAULT = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


def _base_url() -> str:
    return (os.getenv("ANTHROPIC_BASE_URL") or _ANTHROPIC_DEFAULT).rstrip("/")


async def _messages(
    *,
    model: str,
    system: str,
    user_content: list[dict],
    api_key: str | None,
    timeout_sec: float,
    max_tokens: int = 500,
) -> str:
    """POST /v1/messages → 返回纯文本（非 200 抛 RuntimeError，不泄露 key）。"""
    if not api_key:
        raise RuntimeError(
            "anthropic 调用需要 ANTHROPIC_API_KEY：复制 .env.example 为 .env 填入真实 key，"
            "或把 config 的 llm/vision.provider 保持 mock。"
        )
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }
    async with httpx.AsyncClient(timeout=timeout_sec) as client:
        resp = await client.post(
            f"{_base_url()}/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Anthropic HTTP {resp.status_code}: {resp.text[:200]}")
    blocks = resp.json().get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def _guess_media_type(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:12] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:4] == b"GIF8":
        return "image/gif"
    return "image/png"  # 缺省按 png 发（Anthropic 需 image/*，非识别格式会走服务端报错）


class AnthropicVision:
    """Claude 视觉：图片 → 一句用户购物需求（并入检索 query 的文本兜底）。"""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self.calls: int = 0

    async def describe(self, image_bytes: bytes) -> VisionExtraction:
        self.calls += 1
        b64 = base64.b64encode(image_bytes).decode("ascii")
        system = (
            "你是电商智能导购的图片理解模块。看图后输出：用户最可能在找的数码商品需求，"
            "一句中文短句（含品类/核心诉求即可），不要价格、不要下单动作、不要反问。只输出这一句。"
        )
        text = await _messages(
            model=self._s.vision.model,
            system=system,
            user_content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _guess_media_type(image_bytes),
                        "data": b64,
                    },
                },
                {"type": "text", "text": "请描述这张图对应的购物需求。"},
            ],
            api_key=os.getenv(self._s.vision.api_key_env),
            timeout_sec=self._s.llm.timeout_sec,
            max_tokens=120,
        )
        return VisionExtraction(text=text.strip())


class AnthropicDecision:
    """LLM 决定器：从当前状态选下一步工具（search/compare/filter/final）。

    不直接给模型自由裁量参数：选完动作后，args 由确定性代码按 Mock 同一逻辑推导。
    选了不允许/冗余的动作或解析失败 → 回落 MockDecision（确定性，永不卡死）。
    """

    def __init__(self, catalog, settings: Settings, *, chat=None) -> None:
        self._catalog = catalog
        self._s = settings
        self._fallback = MockDecision(catalog)
        # 测试可注入假 chat：(system, user_content) -> text；None 走真 API
        self._chat = chat

    async def _llm_next(self, allowed: list[str], prompt_extra: str) -> str | None:
        system = (
            "你是电商导购 Agent 的调度器。任务：基于当前对话状态，从给出的动作里选**一个**"
            "最合适的，只输出动作名本身（不加标点、不解释）。可选动作与含义见用户消息。"
            "宁可选 final，不要编造动作。"
        )
        user = (
            "当前可行动作：" + "、".join(allowed) + "。\n" + prompt_extra
        )
        text = await self._chat(system, [{"type": "text", "text": user}])
        for token in allowed:
            if token in text:
                return token
        return None

    async def decide(
        self,
        *,
        query: str,
        done_tools: list[str],
        candidates: list[dict],
        rounds_left: int,
    ) -> AgentDecision:
        # 1) 不可答硬规则（代码优先，先于模型）
        kind, hint = classify_refusal(query)
        if kind:
            return AgentDecision(kind="refuse", reason=hint, refusal_kind=kind)

        # 1b) 点名库外数字代际型号（iPhone 17…）→ no_match，先于模型（代码规则）
        _disp, reason = find_out_of_stock_phone(self._catalog, query)
        if reason:
            return AgentDecision(kind="refuse", reason=reason, refusal_kind="no_match")

        done = set(done_tools)
        ids = [c["product_id"] for c in candidates]
        budget = parse_budget_cny(query)

        # 2) 依据状态算"本次确实允许"的动作集合（与 Mock 的规划一致）
        allowed: list[str] = []
        if "search_products" not in done:
            allowed.append("search_products")
        if len(ids) >= 2 and "compare_products" not in done:
            allowed.append("compare_products")
        if budget is not None and "filter_products" not in done:
            allowed.append("filter_products")
        allowed.append("final")

        cand_desc = ", ".join(
            f"{c.get('product_id')}({c.get('name', '')})" for c in candidates
        ) or "无"
        prompt_extra = (
            f"用户问题：{query}\n"
            f"已执行工具：{sorted(done) or '无'}\n"
            f"当前候选：{cand_desc}\n"
            f"剩余工具轮次：{rounds_left}\n"
            "含义：search_products=检索召回；compare_products=对比候选参数（需≥2个候选）；"
            "filter_products=按预算筛候选（仅当问题里有预算数字）；final=证据够了，收束作答。"
        )

        # 3) 调真 LLM；任何异常/超时/解析失败 → 回落确定性 MockDecision
        try:
            chat = self._chat
            if chat is None:
                async def chat(system: str, content: list[dict]) -> str:
                    return await _messages(
                        model=self._s.llm.model,
                        system=system,
                        user_content=content,
                        api_key=os.getenv(self._s.llm.api_key_env),
                        timeout_sec=self._s.llm.timeout_sec,
                        max_tokens=30,
                    )

            pick = await self._llm_next(allowed, prompt_extra)
        except Exception:  # noqa: BLE001 - LLM 掉线不拖垮对话
            pick = None

        if pick is None:
            # 网络/解析失败 → 回落到确定性 MockDecision 规划，不让循环卡死
            return await self._fallback.decide(
                query=query,
                done_tools=done_tools,
                candidates=candidates,
                rounds_left=rounds_left,
            )
        if pick == "final":
            return AgentDecision(kind="final", reason="证据足够，收束作答")

        # 4) 动作 → AgentDecision，参数全由确定性代码推导（模型不碰算术）
        if pick == "search_products":
            return AgentDecision(kind="tool", tool="search_products", args={"query": query})
        if pick == "compare_products":
            return AgentDecision(
                kind="tool",
                tool="compare_products",
                args={"product_ids": ids[:2]},
            )
        if pick == "filter_products":
            return AgentDecision(
                kind="tool",
                tool="filter_products",
                args={"budget_cny": budget, "category": _category_from_query(query)},
            )
        return AgentDecision(kind="final", reason="收束作答")  # pragma: no cover
