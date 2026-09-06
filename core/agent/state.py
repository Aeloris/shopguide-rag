# -*- coding: utf-8 -*-
"""导购 Agent 共享 State（LangGraph）：一张可序列化字典在节点间流转。

与兄弟项目同一约定：TypedDict(total=False)，LangGraph 按 key 覆盖合并(last-write-wins)，
值全部是 JSON-safe 的 dict/list/str/int —— state 可落盘/可 checkpoint/可观测。

字段分组：
- 输入：query / images（视觉已抽取文本，v1 并入 query 语义，占位保留结构）
- 循环与动作：action(最近一次 AgentDecision)/tool_calls(已执行工具数)/tool_results[]
- 证据：candidates[]（可引用商品白名单 —— grounding 的唯一来源）
- 出口：reply(最终 AgentReply 的 dict 形态)
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query: str
    images: list[str]

    action: dict[str, Any]        # 最近一次 AgentDecision.model_dump()
    tool_calls: int               # 已执行工具次数（有界，≤ max_tool_rounds）
    tool_results: list[dict]      # [{tool,args,ok,data}]

    candidates: list[dict]        # 检索/筛选产出的候选商品摘要（引用白名单）

    reply: dict[str, Any] | None  # 最终 AgentReply 的 dict 形态
