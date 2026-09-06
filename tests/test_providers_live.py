# -*- coding: utf-8 -*-
"""真 provider 层的离线可测行为（不联网、不需要 key）：
- AnthropicDecision：LLM 只选动作；非法/掉线回落 Mock；不可答仍是代码硬规则；
- DashScope 缺 key fail-fast；vision 工厂 anthropic 分支构造不抛（key 到用时才要）。
"""
from __future__ import annotations

import pytest

from config.settings import Settings
from core.agent.schemas import AgentDecision
from llm.anthropic import AnthropicDecision, AnthropicVision


def _decision(catalog, *, chat=None) -> AnthropicDecision:
    return AnthropicDecision(catalog, Settings.from_yaml(), chat=chat)


def test_anthropic_decision_refusal_is_code_rule_first(catalog) -> None:
    """含交易词 → 直接 refuse，根本不问 LLM（chat 若被调应抛）。"""
    async def chat(system, content):  # pragma: no cover - 不应被调用
        raise AssertionError("不可答不应调 LLM")

    d = _decision(catalog, chat=chat)
    r = asyncio_run(d.decide, query="帮我下单买一台 iPhone 15", done_tools=[], candidates=[], rounds_left=3)
    assert r.kind == "refuse" and r.refusal_kind == "trade"


def test_anthropic_decision_uses_llm_tool_choice(catalog) -> None:
    async def chat(system, content):
        return "compare_products"  # LLM 选对比

    d = _decision(catalog, chat=chat)
    r = asyncio_run(d.decide, query="对比这两款哪个拍照好", done_tools=["search_products"], candidates=[{"product_id": "a", "name": "A"}, {"product_id": "b", "name": "B"}], rounds_left=2)
    assert r.kind == "tool" and r.tool == "compare_products"
    assert r.args["product_ids"] == ["a", "b"]  # 参数由代码推导，不来自模型


def test_anthropic_decision_llm_final(catalog) -> None:
    async def chat(system, content):
        return "final"

    d = _decision(catalog, chat=chat)
    r = asyncio_run(d.decide, query="推荐个轻薄本", done_tools=["search_products"], candidates=[{"product_id": "x"}], rounds_left=2)
    assert r.kind == "final"


def test_anthropic_decision_falls_back_on_llm_failure(catalog) -> None:
    async def chat(system, content):
        raise RuntimeError("api down")

    d = _decision(catalog, chat=chat)
    r = asyncio_run(d.decide, query="3000内直屏手机推荐", done_tools=[], candidates=[], rounds_left=3)
    # 回落 MockDecision 确定性规划 → 先 search
    assert r.kind == "tool" and r.tool == "search_products"


def test_anthropic_decision_ignores_disallowed_action(catalog) -> None:
    """LLM 挑了当前不允许的 filter（无预算数字）→ 不允许即回落，不采纳。"""
    async def chat(system, content):
        return "filter_products"

    d = _decision(catalog, chat=chat)
    r = asyncio_run(d.decide, query="轻薄本哪款好", done_tools=["search_products"], candidates=[{"product_id": "a"}, {"product_id": "b"}], rounds_left=2)
    # filter 不在 allowed → 回落 Mock：无预算、无对比词 → final
    assert r.kind in ("tool", "final")
    assert not (r.kind == "tool" and r.tool == "filter_products")


def test_dashscope_embedding_without_key_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    from core.embeddings import get_embedding_provider

    s = Settings.from_yaml()
    s.embedding.provider = "dashscope"
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        get_embedding_provider(s)


def test_vision_factory_anthropic_constructs_without_key() -> None:
    """anthropic 分支构造不抛（key 只在 describe 时读）→ 缺 key 不误伤启动。"""
    from llm.vision import get_vision_provider

    s = Settings.from_yaml()
    s.vision.provider = "anthropic"
    provider = get_vision_provider(s)
    assert isinstance(provider, AnthropicVision)


def asyncio_run(fn, **kw) -> AgentDecision:
    import asyncio

    return asyncio.run(fn(**kw))
