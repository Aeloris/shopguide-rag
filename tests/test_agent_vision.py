# -*- coding: utf-8 -*-
"""多模态入口：图片经视觉 Provider(mock fixture) 抽成需求并入检索 query。"""
from __future__ import annotations

import asyncio

from core.agent import AgentReply, AgentRuntime


def _ask(runtime: AgentRuntime, query: str = "", *, images: list[bytes] | None = None) -> AgentReply:
    return asyncio.run(runtime.ask(query, images=images))


def test_image_only_uses_vision_text(runtime: AgentRuntime) -> None:
    """仅传图：query 由 vision 描述兜底（fixture：拍照好的旗舰手机），应正常召回。"""
    r = _ask(runtime, images=[b"\x89PNG fake screenshot bytes"])
    assert r.answerable and not r.refused
    assert r.tool_trace and r.tool_trace[0] == "search_products"
    assert r.recommendations
    # fixture 语义指向"手机/拍照/旗舰" → 推荐应以手机为主
    cats = {rec["category"] for rec in r.recommendations}
    assert "phone" in cats


def test_image_and_text_prefers_text(runtime: AgentRuntime) -> None:
    """图文同传：以用户文字为主（含诉求），图片描述只兜底。"""
    r = _ask(runtime, "适合办公的轻薄笔记本", images=[b"fake"])
    assert r.answerable and not r.refused
    # 文字是笔记本诉求 → 不应被图(手机)描述带偏到手机
    assert any(rec["category"] == "laptop" for rec in r.recommendations)
