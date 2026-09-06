# -*- coding: utf-8 -*-
"""决定层：每一步"下一步调哪个工具 / 收束作答 / 拒绝"。

诚实边界：
- 默认实现 MockDecision 是**确定性规则决定器**（可离线测、可回归）——它扮演一个
  "听话的 LLM"：先识别不可答(拒绝)，再做工具规划，规划用尽就收束。
- 接真实 LLM 时，把这里换成走 Provider.chat(schema=AgentDecision) 的实现即可
  （同一个 LangGraph 决定节点，同一份工具与 grounding 校验，业务零改动）。v1 不接。
- 拒绝分类(不可答)是**代码规则**而非模型 —— 三类"不该答"（交易/实时行情/未来）
  由关键词判定，任何 provider 都必须遵守（面试讲：硬规则兜底防幻觉）。

AgentDecision.kind：
    tool   → run_tool 节点执行（有界循环，≤ max_tool_rounds）
    final  → finalize 收束（据工具结果拼结构化回执）
    refuse → 直接拒绝（不调工具，杜绝为不可答问题浪费轮次）
"""
from __future__ import annotations

from typing import Any

from core.agent.schemas import AgentDecision, RefusalKind
from core.agent.tools import _category_from_query, parse_budget_cny

# ---- 不可答硬规则（词面命中即拒绝；宁可转引导，不硬答/不瞎编）----
_REFUSAL_RULES: list[tuple[RefusalKind, list[str]]] = [
    (
        "trade",
        ["下单", "帮我买", "我要买", "购买", "帮我下单", "支付", "付款", "结账", "加购物车", "把货发"],
    ),
    (
        "realtime",
        ["补贴", "百亿", "渠道价", "拼多多", "京东价", "实时价格", "现在多少钱", "现价", "最低价", "秒杀", "行情"],
    ),
    (
        "future",
        ["下季度", "下一代", "什么时候发布", "何时发布", "会发布", "即将发布", "新品预告", "爆料", "明年新款", "发布会"],
    ),
]

# ---- 意图词（用于规划工具次序）----
_COMPARE_WORDS = ["对比", "区别", "哪个", "谁更", "差异", "有什么区别", " vs ", "比一比"]
_BUDGET_WORDS = ["预算", "以内", "不超过", "元以内", "之内", "封顶", "多少元"]


def classify_refusal(text: str) -> tuple[RefusalKind | None, str]:
    """不可答硬规则判定：返回 (类别, 提示) 或 (None, '')。"""
    for kind, words in _REFUSAL_RULES:
        for w in words:
            if w in text:
                hint = {
                    "trade": "本项目只做导购推荐，不做下单/支付等交易动作",
                    "realtime": "库内价格为静态演示样例，无实时渠道/补贴数据源",
                    "future": "无新品发布/爆料类信息源，无法预测",
                }[kind]
                return kind, hint
    return None, ""


def _has_compare_intent(text: str) -> bool:
    return any(w in text for w in _COMPARE_WORDS)


def _has_budget_intent(text: str) -> bool:
    return any(w in text for w in _BUDGET_WORDS)


class MockDecision:
    """确定性规则决定器（离线默认）：拒绝 → 规划工具 → 收束。"""

    def __init__(self, catalog: Any) -> None:
        self._catalog = catalog

    async def decide(
        self,
        *,
        query: str,
        done_tools: list[str],
        candidates: list[dict],
        rounds_left: int,
    ) -> AgentDecision:
        # 1) 不可答硬规则最优先（不浪费工具轮次）
        kind, hint = classify_refusal(query)
        if kind:
            return AgentDecision(kind="refuse", reason=hint, refusal_kind=kind)

        done = set(done_tools)
        if "search_products" not in done:
            return AgentDecision(kind="tool", tool="search_products", args={"query": query})

        # 2) 已召回 → 看是否需要继续（对比 / 预算筛选）
        ids = [c["product_id"] for c in candidates]
        if _has_compare_intent(query) and len(ids) >= 2 and "compare_products" not in done:
            return AgentDecision(
                kind="tool", tool="compare_products", args={"product_ids": ids[:2]}
            )
        if _has_budget_intent(query) and "filter_products" not in done:
            # 预算数字可解析才筛；品类从问句里推断（手机/笔记本/平板）
            budget = parse_budget_cny(query)
            if budget is not None:
                return AgentDecision(
                    kind="tool",
                    tool="filter_products",
                    args={
                        "budget_cny": budget,
                        "category": _category_from_query(query),
                    },
                )

        # 3) 工具轮次上限兜底（理论上 decide 不会在有轮次时才给 tool；防御）
        if rounds_left <= 0 and candidates:
            return AgentDecision(kind="final")

        # 4) 没有更多可做 → 收束
        return AgentDecision(kind="final")
