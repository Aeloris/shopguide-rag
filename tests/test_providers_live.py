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


# ---- DashScope embedding 分批（真联调暴露：端点单请求 ≤10 条，400 超限）----


class _FakeEmbedResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeEmbedClient:
    """记录每次请求的输入条数；向量首位编码"全局序号"以校验顺序与不丢不重。"""

    def __init__(self):
        self.request_sizes: list[int] = []
        self._base = 0
        self._max_seen = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, *, headers=None, json=None):
        n = len(json["input"])
        self.request_sizes.append(n)
        start = self._base
        self._base += n
        data = [
            {"index": i, "embedding": [float(start + i), 0.0]}
            for i in range(n)
        ]
        return _FakeEmbedResp({"data": data})


def test_dashscope_embedding_batches_within_limit_and_keeps_order(monkeypatch) -> None:
    """25 条应拆成 ≤10 的多个请求，返回序与输入序一致（首维=全局序号）。"""
    import core.embeddings as ce

    fake = _FakeEmbedClient()
    monkeypatch.setattr(ce.httpx, "AsyncClient", lambda **kw: fake)

    p = ce.DashScopeEmbedding(dimension=2, api_key="sk-fake-for-test")
    n_texts = 25
    vecs = asyncio_run(p.embed, texts=[f"text-{i}" for i in range(n_texts)])

    assert fake.request_sizes == [10, 10, 5]  # 严格分批，无超限
    assert len(vecs) == n_texts
    assert [round(v[0]) for v in vecs] == list(range(n_texts))  # 顺序一致、不丢不重


def test_dashscope_embedding_single_small_batch(monkeypatch) -> None:
    """≤10 条应单请求（不无谓分批）。"""
    import core.embeddings as ce

    fake = _FakeEmbedClient()
    monkeypatch.setattr(ce.httpx, "AsyncClient", lambda **kw: fake)

    p = ce.DashScopeEmbedding(dimension=2, api_key="sk-fake-for-test")
    vecs = asyncio_run(p.embed, texts=["a", "b"])

    assert fake.request_sizes == [2]
    assert len(vecs) == 2
