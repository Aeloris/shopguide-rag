# -*- coding: utf-8 -*-
"""导购 Agent 的数据契约（pydantic v2）。

设计约束（面试诚实点）：
- 状态机共享 State 里流转的是 JSON-safe 的 dict/list（与兄弟项目同一约定）；
  pydantic 只在"进出"层出现：决定层输入/输出、最终结构化回执 AgentReply。
- AgentReply 是**结构化**的（recommendations/comparison/refused/reason + 工具 trace），
  渲染成人类文本在 app/API 层做 —— 测评/门禁断言的是结构，不是脆弱的措辞。
- "可引用商品"是一个显式白名单（candidates）：最终回执里出现的每个 product_id
  都必须来自该白名单（工具结果），回答层**绝不凭空引入**商品 —— grounding 由构造保证。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RefusalKind = Literal[
    "trade",      # 交易动作：下单/购买/支付 —— 本项目不做
    "realtime",   # 实时渠道价/补贴行情 —— 库内价格是静态样例，无爬虫
    "future",     # 未来发布/爆料 —— 无新闻预测数据源
    "no_match",   # 库内无匹配（检索零命中/用户问的型号不在库）
]

# AgentDecision.kind：agent 循环的三种出口动作
DecisionKind = Literal["tool", "final", "refuse"]


class AgentDecision(BaseModel):
    """决定层输出：下一步干什么（tool=调哪个工具；final=收束作答；refuse=拒绝/转引导）。"""

    kind: DecisionKind = "tool"
    tool: str | None = Field(default=None, description="kind=tool 时的工具名")
    args: dict = Field(default_factory=dict, description="工具参数（JSON-safe）")
    reason: str = Field(default="", description="refuse/final 的短标签或提示")
    refusal_kind: RefusalKind | None = Field(default=None, description="kind=refuse 时的拒绝类别")


class CompareRow(BaseModel):
    """compare 工具产出：一行 = 一个商品的取值（只含两品共有的维度键）。"""

    product_id: str
    name: str
    price_cny: int
    dims: dict[str, str]


class ComparisonResult(BaseModel):
    """compare 的结构化结果：对齐哪几个维度 + 每品取值（行对齐，缺维度不硬补）。"""

    product_ids: list[str]
    dims: list[str]
    rows: list[CompareRow]


class AgentReply(BaseModel):
    """一次导购对话的结构化回执（渲染层再转文本）。"""

    answerable: bool
    refused: bool = False
    refusal_kind: RefusalKind | None = None
    refusal_reason: str = ""
    recommendations: list[dict] = Field(
        default_factory=list, description="被推荐/检索到的商品摘要（引用的唯一来源）"
    )
    comparison: ComparisonResult | None = None
    filter_note: str = Field(default="", description="预算筛选口径说明（命中 X 件/共 Y 件）")
    tool_trace: list[str] = Field(default_factory=list, description="本次实际执行的工具序列（可观测）")

    @property
    def cited_product_ids(self) -> list[str]:
        """回执中出现的全部商品 id（grounding 审计用）。"""
        out = [r["product_id"] for r in self.recommendations]
        if self.comparison:
            out.extend(self.comparison.product_ids)
        # 去重保序
        seen: set[str] = set()
        return [pid for pid in out if not (pid in seen or seen.add(pid))]
