# -*- coding: utf-8 -*-
"""Agent 状态机 e2e：工具路由 / 预算筛选(代码判定) / 参数对比 / 不可答拒绝 / grounding。"""
from __future__ import annotations

import asyncio

from core.agent import AgentReply, AgentRuntime
from core.catalog.loader import Catalog


def _ask(runtime: AgentRuntime, query: str) -> AgentReply:
    return asyncio.run(runtime.ask(query))


def _ids(reply: AgentReply) -> list[str]:
    return reply.cited_product_ids


def test_search_recommend_flow(runtime: AgentRuntime, catalog: Catalog) -> None:
    r = _ask(runtime, "适合办公的轻薄笔记本")
    assert r.answerable and not r.refused
    assert r.tool_trace == ["search_products"]
    assert r.recommendations
    # 引用白名单 grounding：推荐里的每个 SKU 都真实存在于库、且价格为正
    assert set(_ids(r)) <= {p.id for p in catalog.products}
    assert all(rec["price_cny"] > 0 for rec in r.recommendations)
    assert r.comparison is None and r.filter_note == ""


def test_budget_flow_filters_by_code(runtime: AgentRuntime, catalog: Catalog) -> None:
    """预算筛选必须由代码判价：回执里每个推荐都 ≤3000 且是手机（不是模型口算）。"""
    r = _ask(runtime, "预算3000以内的安卓直屏手机")
    assert r.answerable and not r.refused
    assert "filter_products" in r.tool_trace
    assert "search_products" in r.tool_trace
    assert r.filter_note and "3000" in r.filter_note
    assert r.recommendations
    for rec in r.recommendations:
        assert rec["price_cny"] <= 3000
        assert rec["category"] == "phone"
        p = catalog.get(rec["product_id"])
        assert p is not None and p.price_cny == rec["price_cny"]


def test_budget_no_match_refuses(runtime: AgentRuntime) -> None:
    """预算内一件都没有 → 宁可拒绝转引导，不给越预算的"硬塞"候选。"""
    r = _ask(runtime, "预算2000以内的手机")
    assert not r.answerable and r.refused
    assert r.refusal_kind == "no_match"
    assert r.recommendations == [] and r.tool_trace == ["search_products", "filter_products"]


def test_compare_flow_rows_aligned(runtime: AgentRuntime, catalog: Catalog) -> None:
    r = _ask(runtime, "小米14和vivo X100哪个拍照好")
    assert r.answerable and not r.refused
    assert "compare_products" in r.tool_trace
    c = r.comparison
    assert c is not None and len(c.product_ids) == 2 and c.dims
    assert len(c.rows) == len(c.product_ids)
    for row in c.rows:
        p = catalog.get(row.product_id)
        assert p is not None
        assert row.price_cny == p.price_cny
        # 对齐的维度一定是"两品共有规格键"的子集（不缺列硬补）
        common = set(p.specs) & set(catalog.get(c.product_ids[0]).specs)  # type: ignore[union-attr]
        assert set(row.dims.keys()) <= common and set(row.dims.keys()) == set(c.dims)


def test_trade_refusal(runtime: AgentRuntime) -> None:
    r = _ask(runtime, "帮我下单买一台 iPhone 15")
    assert not r.answerable and r.refused
    assert r.refusal_kind == "trade"
    assert r.recommendations == [] and r.cited_product_ids == []
    assert r.tool_trace == []  # 不可答不浪费工具轮次


def test_realtime_price_refusal(runtime: AgentRuntime) -> None:
    r = _ask(runtime, "Redmi K70 现在拼多多百亿补贴多少钱")
    assert r.refused and r.refusal_kind == "realtime"


def test_benchmark_refusal(runtime: AgentRuntime) -> None:
    """问"这台具体型号能不能跑"需真机实测 → 拒绝（不给规格想当然）。"""
    r = _ask(runtime, "这台华为笔记本能玩 3A 大作吗")
    assert r.refused and r.refusal_kind == "benchmark"
    assert r.recommendations == []


def test_recommend_capable_laptop_not_refused(runtime: AgentRuntime) -> None:
    """"推荐一台能玩3A的笔记本"是合法导购（没指向具体型号）→ 应召回独显本而非拒绝。"""
    r = _ask(runtime, "推荐一台能玩 3A 大作的笔记本")
    assert r.answerable and not r.refused
    assert r.recommendations
    assert any(rec["category"] == "laptop" for rec in r.recommendations)


def test_future_refusal(runtime: AgentRuntime) -> None:
    r = _ask(runtime, "华为 Mate 60 Pro 下一代什么时候发布")
    assert r.refused and r.refusal_kind == "future"


def test_grounding_never_fabricates(runtime: AgentRuntime, catalog: Catalog) -> None:
    """对多种可答问法：回执引用的每个 id 都真实存在 —— 无凭空商品。"""
    known = {p.id for p in catalog.products}
    for q in [
        "适合办公的轻薄笔记本",
        "预算3000以内的安卓直屏手机",
        "小米14和vivo X100哪个拍照好",
        "512g 拍人像好的手机",
        "游戏本 RTX 4060",
    ]:
        r = _ask(runtime, q)
        assert r.cited_product_ids, f"{q!r} 应该给出可引用商品"
        assert set(r.cited_product_ids) <= known, f"{q!r} 引用了库外商品"
