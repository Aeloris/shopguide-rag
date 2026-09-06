# -*- coding: utf-8 -*-
"""Agent 评测 harness：指标算术（假 runtime 注入）+ 报告结构。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core.agent import AgentReply
from core.catalog.loader import Catalog
from core.catalog.schemas import Product
from evals.agent_eval import evaluate_agent
from evals.dataset import UnanswerableCase


def _product(pid: str) -> Product:
    return Product(
        id=pid, brand="B", model=pid, name=pid, category="phone",  # type: ignore[arg-type]
        price_cny=3000, summary="s", specs={"存储": "256GB"},
    )


def _fake_runtime(replies: dict[str, AgentReply]):
    catalog = Catalog([_product("p1"), _product("p2")])

    async def ask(query: str) -> AgentReply:
        return replies.get(query, AgentReply(answerable=False, refused=True))

    return SimpleNamespace(catalog=catalog, ask=ask)


def test_rate_arithmetic() -> None:
    """构造 2 可答 + 1 不可答，手动核对 grounded/bad_refusal/拦截率。"""
    rt = _fake_runtime(
        {
            # 正常作答且引用在库 → grounded & served
            "正常推荐": AgentReply(answerable=True, recommendations=[{"product_id": "p1", "price_cny": 3000}]),
            # 可答却拒绝 → grounded False & served False（bad_refusal 拉低）
            "应答却拒": AgentReply(answerable=False, refused=True, refusal_kind="no_match"),
            # 库外引用 → grounded False（不应发生，护栏）
            "库外引用": AgentReply(answerable=True, recommendations=[{"product_id": "ghost", "price_cny": 1}]),
            # 不可答正确拒绝
            "帮我下单": AgentReply(answerable=False, refused=True, refusal_kind="trade"),
        }
    )
    rep = asyncio.run(
        evaluate_agent(
            rt,
            answerable_queries=["正常推荐", "应答却拒", "库外引用"],
            unanswerable_cases=[UnanswerableCase(query="帮我下单", reason="x")],
        )
    )
    # 正常推荐 grounded；应答却拒、库外引用均 grounded=False → 1/3
    assert rep.grounded_rate == round(1 / 3, 4)
    # served：正常推荐 + 库外引用（虽引用脏但没拒绝）→ 2/3
    assert rep.bad_refusal_rate == round(2 / 3, 4)
    assert rep.unanswerable_refusal_rate == 1.0
    assert rep.n_answerable == 3 and rep.n_unanswerable == 1


def test_all_refused_gives_zero() -> None:
    rt = _fake_runtime({})
    rep = asyncio.run(
        evaluate_agent(rt, answerable_queries=["a"], unanswerable_cases=[])
    )
    assert rep.grounded_rate == 0.0 and rep.bad_refusal_rate == 0.0
    assert rep.unanswerable_refusal_rate is None
