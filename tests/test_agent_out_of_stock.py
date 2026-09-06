# -*- coding: utf-8 -*-
"""库外数字代际型号门禁回归（真机 iPhone 17 图 e2e 曝光缺陷修复）。

背景：用户供真实照片说是 iPhone 17，而目录里 iPhone 只有 15 → 修前点名库外代际型号
（"iPhone 17"）也会拿同品牌最近一代（iphone-15）冒充命中返回。修复：决定器先于检索/模型
检查"点名了在售系列之外的数字代际"（iPhone 17/小米 15/Mate 70/Redmi K80/vivo X200/OnePlus 13）
→ `no_match` 拒答，理由点明型号与在售代际。库内代际从目录自校准，不硬编码。

诚实边界：只覆盖 6 个"数字代际清晰"的在售手机系列；笔记本/平板（M3/X1 Carbon/Pro16/Air5/
MatePad 13.2）无稳定数字代际规则，不在本门禁内（README 已述，仍按最近在售）。
"""
from __future__ import annotations

import asyncio

from config.settings import Settings
from core.agent.decision import find_out_of_stock_phone

# 库里在售代际之外的数字型号 → 应 no_match 拒答（不拿相近代冒充命中）
_OUT_OF_STOCK = [
    "iPhone 17", "想买一台 iPhone 17", "苹果17",
    "小米 15", "Redmi K80", "红米 K80",
    "华为 Mate 70", "vivo X200", "OnePlus 13", "一加 13",
]

# 在售数字代际型号 → 不得误拒
_IN_STOCK = [
    "iPhone 15", "小米 14", "Redmi K70", "华为 Mate 60 Pro", "vivo X100", "一加 12",
]


def test_find_out_of_stock_phone_pure(catalog) -> None:
    """纯函数：点名库外代际 → 有原因；在售代际/无型号泛问 → 无原因。"""
    for q in _OUT_OF_STOCK:
        disp, reason = find_out_of_stock_phone(catalog, q)
        assert reason, f"应判库外: {q!r}"
        assert disp, q
    for q in _IN_STOCK:
        disp, reason = find_out_of_stock_phone(catalog, q)
        assert not reason, f"在售型号不应误拒: {q!r}"
    for q in ["紫色iPhone推荐", "苹果手机哪款好", "办公轻薄笔记本", "3000内手机"]:
        _, reason = find_out_of_stock_phone(catalog, q)
        assert not reason, f"泛问（无数字代际）不应判库外: {q!r}"


def test_agent_refuses_out_of_stock_model(runtime) -> None:
    """Agent 层：点名库外代际 → no_match 拒答，且在调任何工具前（不浪费轮次）。"""
    reply = asyncio.run(runtime.ask("想买一台 iPhone 17"))
    assert reply.refused is True
    assert reply.refusal_kind == "no_match"
    assert "iPhone 17" in (reply.refusal_reason or "")
    assert reply.tool_trace == [], "应在调工具前拒答"


def test_agent_still_answers_in_stock_models(runtime) -> None:
    """在售代际型号照常作答，不误伤。"""
    for q in _IN_STOCK:
        reply = asyncio.run(runtime.ask(q))
        assert reply.answerable is True, q


def test_anthropic_decision_out_of_stock_is_code_rule_before_llm(catalog) -> None:
    """真 LLM 决定器也必须先走代码门禁：点名库外型号不应调 LLM。"""
    from llm.anthropic import AnthropicDecision

    async def chat(system, content):  # pragma: no cover - 不应被调用
        raise AssertionError("点名库外型号不应调 LLM")

    d = AnthropicDecision(catalog, Settings.from_yaml(), chat=chat)
    r = asyncio.run(d.decide(query="iPhone 17 推荐", done_tools=[], candidates=[], rounds_left=3))
    assert r.kind == "refuse" and r.refusal_kind == "no_match"
