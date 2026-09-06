# -*- coding: utf-8 -*-
"""导购 Agent 状态机（LangGraph StateGraph 落地版）。

图结构：
    START → decide ──(tool)──▶ run_tool ──▶ decide   （有界工具循环）
                 └──(final)─▶ finalize ─▶ END
                 └──(refuse)─▶ refuse  ─▶ END

为什么是"状态机 + 有界循环"而不是一个巨型 LLM 调用（面试可讲）：
1. **工具是确定性代码**（检索/预算筛选/参数对比全可验证），模型/规则只决定"下一步
   调哪个"；预算算术、规格对齐绝不交给模型 —— 幻觉与算术错被结构性地排除；
2. **证据白名单在 state 里显式流转**：finalize 只允许引用 candidates 中的商品，
   回答的 grounding 由构造保证（不是事后去抓 LLM 的引用）；
3. **不可答走独立 refuse 出口**，不浪费工具轮次、不硬答（坏答防伪的编排形状）。
4. 节点全是 async，统一 `await graph.ainvoke(...)` —— FastAPI 里可直接 await，
   测试里 asyncio.run 包，不会出现"环内再起事件循环"。

诚实边界：决定层默认是 MockDecision（确定性规则），扮演"听话的 LLM"；接真实 LLM
时替换 decision 实现即可，图、工具、grounding 校验一条边都不动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, StateGraph

from core.agent.decision import MockDecision
from core.agent.schemas import AgentDecision, AgentReply, ComparisonResult
from core.agent.state import AgentState
from core.agent.tools import GuideTools
from core.catalog.loader import Catalog
from core.retriever import Retriever


@dataclass
class NodeContext:
    """节点闭包依赖：工具集 + 决定器 + 有界轮次。"""

    tools: GuideTools
    decision: Any  # MockDecision（或将来实现同接口的真 LLM 决定器）
    max_tool_rounds: int


async def _execute(tools: GuideTools, tool: str, args: dict) -> dict[str, Any]:
    fn = getattr(tools, tool, None)
    if fn is None:
        return {"ok": False, "tool": tool, "error": f"未知工具：{tool}"}
    res = fn(**args)
    if hasattr(res, "__await__"):  # search_products 是 async；其余同步
        res = await res
    return res


# ---------------- 节点 ----------------
async def decide_node(state: dict, ctx: NodeContext) -> dict:
    rounds_left = ctx.max_tool_rounds - int(state.get("tool_calls", 0))
    if rounds_left <= 0:
        # 兜底：轮次耗尽仍未收束 → 强制 final（不无限循环）
        decision = AgentDecision(kind="final", reason="已达工具轮次上限，基于现有证据作答")
    else:
        done = [r["tool"] for r in state.get("tool_results", [])]
        decision = await ctx.decision.decide(
            query=state["query"],
            done_tools=done,
            candidates=state.get("candidates", []),
            rounds_left=rounds_left,
        )
    return {"action": decision.model_dump()}


async def run_tool_node(state: dict, ctx: NodeContext) -> dict:
    action = AgentDecision(**state["action"])
    result = await _execute(ctx.tools, action.tool or "", action.args or {})
    tool_results = list(state.get("tool_results", []))
    tool_results.append({"tool": action.tool, "args": action.args, **result})

    candidates = state.get("candidates", [])
    # 证据白名单随工具结果滚动：search/filter 的产物成为可引用候选
    if action.tool in ("search_products", "filter_products") and result.get("ok"):
        candidates = result.get("products", [])

    return {
        "tool_results": tool_results,
        "tool_calls": int(state.get("tool_calls", 0)) + 1,
        "candidates": candidates,
    }


def _comparison_from_result(result: dict) -> ComparisonResult | None:
    if result.get("ok") and result.get("rows"):
        return ComparisonResult(
            product_ids=result.get("product_ids", []),
            dims=result.get("dims", []),
            rows=result.get("rows", []),
        )
    return None


async def finalize_node(state: dict, ctx: NodeContext) -> dict:
    candidates = state.get("candidates", [])
    tool_results = state.get("tool_results", [])
    trace = [r["tool"] for r in tool_results]

    comparison: ComparisonResult | None = None
    filter_note = ""
    for r in reversed(tool_results):
        if r.get("tool") == "compare_products" and comparison is None:
            comparison = _comparison_from_result(r)
        elif r.get("tool") == "filter_products" and not filter_note:
            ok, mc, tc = r.get("ok"), r.get("matched_count"), r.get("total_count")
            if ok:
                cat = r.get("category") or "全品类"
                filter_note = (
                    f"按预算¥{r.get('budget_cny')}筛选：命中 {mc}/{tc} 件（品类={cat}）"
                )

    if not candidates:
        reply = AgentReply(
            answerable=False,
            refused=True,
            refusal_kind="no_match",
            refusal_reason="库内没有匹配的商品，换更明确的需求试试（品牌/预算/品类）",
            tool_trace=trace,
        )
    else:
        reply = AgentReply(
            answerable=True,
            refused=False,
            recommendations=candidates[:5],
            comparison=comparison,
            filter_note=filter_note,
            tool_trace=trace,
        )
    return {"reply": reply.model_dump()}


async def refuse_node(state: dict, ctx: NodeContext) -> dict:
    action = AgentDecision(**state.get("action", {}))
    reply = AgentReply(
        answerable=False,
        refused=True,
        refusal_kind=action.refusal_kind,
        refusal_reason=action.reason,
        tool_trace=[],
    )
    return {"reply": reply.model_dump()}


# ---------------- 条件路由 ----------------
def _route_after_decide(state: dict) -> str:
    action = state.get("action") or {}
    kind = action.get("kind")
    if kind == "tool":
        return "run_tool"
    if kind == "refuse":
        return "refuse"
    return "finalize"


# ---------------- 组装 ----------------
def build_graph(ctx: NodeContext):
    def bind(node):
        async def wrapped(state: dict) -> dict:
            return await node(state, ctx)

        return wrapped

    g = StateGraph(AgentState)
    for name in ("decide", "run_tool", "finalize", "refuse"):
        impl = {
            "decide": decide_node,
            "run_tool": run_tool_node,
            "finalize": finalize_node,
            "refuse": refuse_node,
        }[name]
        g.add_node(name, bind(impl))
    g.set_entry_point("decide")
    g.add_conditional_edges("decide", _route_after_decide)
    g.add_edge("run_tool", "decide")
    g.add_edge("finalize", END)
    g.add_edge("refuse", END)
    return g.compile()


def build_ctx(catalog: Catalog, retriever: Retriever, *, max_tool_rounds: int = 3) -> NodeContext:
    tools = GuideTools(catalog, retriever)
    return NodeContext(
        tools=tools,
        decision=MockDecision(catalog),
        max_tool_rounds=max_tool_rounds,
    )
