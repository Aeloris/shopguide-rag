# -*- coding: utf-8 -*-
"""/api/search 热点缓存：同 query 二次命中只算一次检索；异 query 各算各的。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from core.agent import AgentRuntime
from core.store.cache import InMemoryCache


def test_search_second_same_query_served_from_cache(runtime: AgentRuntime, monkeypatch) -> None:
    calls = {"n": 0}
    real = runtime.retriever.search

    async def counting(query: str):
        calls["n"] += 1
        return await real(query)

    monkeypatch.setattr(runtime.retriever, "search", counting)

    cache = InMemoryCache()
    app = create_app(runtime=runtime, cache=cache)
    with TestClient(app) as c:
        q = {"query": "适合办公的轻薄笔记本"}
        r1 = c.post("/api/search", json=q)
        r2 = c.post("/api/search", json=q)
        assert r1.status_code == 200 and r1.json() == r2.json()
        assert calls["n"] == 1  # 第二次命中缓存，不再检索

        c.post("/api/search", json={"query": "预算 2000 的拍照手机"})
        assert calls["n"] == 2  # 不同 query → 不同缓存 key


def test_search_cache_respects_top_k(runtime: AgentRuntime, monkeypatch) -> None:
    calls = {"n": 0}
    real = runtime.retriever.search

    async def counting(query: str):
        calls["n"] += 1
        return await real(query)

    monkeypatch.setattr(runtime.retriever, "search", counting)

    app = create_app(runtime=runtime, cache=InMemoryCache())
    with TestClient(app) as c:
        c.post("/api/search", json={"query": "手机", "top_k": 3})
        c.post("/api/search", json={"query": "手机", "top_k": 5})
        assert calls["n"] == 2  # top_k 参与 key → 不误命中
