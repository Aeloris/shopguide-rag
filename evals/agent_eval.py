# -*- coding: utf-8 -*-
"""Agent 评测 harness：回答有据率 / 好例误拒率 / 不可答拦截率。

口径（诚实边界）：
- **grounded_rate**：预期可答的问题里，"确实作答且引用非空且每个引用都在库"占比。
  grounding 由构造保证（引用=candidates 白名单），此指标是回归护栏：任何路径改动若让
  回答引用库外/空引用/被误拒，立刻变红。
- **bad_refusal_rate**：预期可答的问题里被正确服务（未误拒）占比 —— 防"宁可拒绝"式的
  偷懒让可答问题也被拒。
- **unanswerable_refusal_rate**：对抗不可答集里被正确拒绝的占比（检测率）。

指标全是确定性代码计数（metrics.py 复刻语义），离线 mock 下可回放、可回归。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.agent import AgentReply, AgentRuntime
from core.catalog.loader import Catalog
from evals.dataset import UnanswerableCase


@dataclass
class AgentCaseResult:
    query: str
    expected_answerable: bool  # True=应作答；False=应拒绝
    reply: AgentReply
    grounded: bool = False
    served: bool = False


@dataclass
class AgentEvalReport:
    cases: list[AgentCaseResult] = field(default_factory=list)
    grounded_rate: float | None = None
    bad_refusal_rate: float | None = None
    unanswerable_refusal_rate: float | None = None

    @property
    def n_answerable(self) -> int:
        return sum(1 for c in self.cases if c.expected_answerable)

    @property
    def n_unanswerable(self) -> int:
        return sum(1 for c in self.cases if not c.expected_answerable)


async def evaluate_agent(
    runtime: AgentRuntime,
    *,
    answerable_queries: list[str],
    unanswerable_cases: list[UnanswerableCase],
) -> AgentEvalReport:
    catalog: Catalog = runtime.catalog
    known = {p.id for p in catalog.products}

    cases: list[AgentCaseResult] = []
    for q in answerable_queries:
        reply = await runtime.ask(q)
        served = bool(reply.answerable and not reply.refused)
        grounded = bool(
            served
            and reply.cited_product_ids
            and set(reply.cited_product_ids) <= known
        )
        cases.append(
            AgentCaseResult(
                query=q, expected_answerable=True, reply=reply,
                grounded=grounded, served=served,
            )
        )
    for uc in unanswerable_cases:
        reply = await runtime.ask(uc.query)
        served = bool(reply.answerable and not reply.refused)
        cases.append(
            AgentCaseResult(
                query=uc.query, expected_answerable=False, reply=reply,
                grounded=False, served=served,
            )
        )

    ans = [c for c in cases if c.expected_answerable]
    unans = [c for c in cases if not c.expected_answerable]
    grounded_rate = (
        round(sum(c.grounded for c in ans) / len(ans), 4) if ans else None
    )
    bad_refusal_rate = (
        round(sum(c.served for c in ans) / len(ans), 4) if ans else None
    )
    unans_refusal = (
        round(sum(not c.served for c in unans) / len(unans), 4) if unans else None
    )
    return AgentEvalReport(
        cases=cases,
        grounded_rate=grounded_rate,
        bad_refusal_rate=bad_refusal_rate,
        unanswerable_refusal_rate=unans_refusal,
    )
