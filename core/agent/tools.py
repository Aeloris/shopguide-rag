# -*- coding: utf-8 -*-
"""导购 Agent 的三个工具 —— 全部是**确定性代码路径**，模型不做算术。

    search_products(query)     关键词召回候选（走混合检索链路）
    filter_products(...)       预算/品类/标签筛选（代码判断 price<=budget，模型不算）
    compare_products(ids,dims) 参数对比（只对齐两品**共有**的规格键，缺列不硬补）

为什么"这些用代码而不是让模型算"（面试可讲）：
- 预算比较、规格对齐是可验证的确定逻辑 —— 交给模型会引入幻觉与算术错；
  检索层召回、Agent 决定"调哪个工具"才是模型（LLM/规则）该做的。
- 工具的入参/出参都是 JSON-safe dict，直接可写进 LangGraph 共享 State。

每个工具返回 {ok, ...} 包裹：错误也在结果里（run_tool 节点据此决定是否继续），
不抛异常让整个 state 机炸掉 —— 工具级失败是数据，不是异常。
"""
from __future__ import annotations

import re
from typing import Any

from core.agent.schemas import ComparisonResult
from core.catalog.loader import Catalog
from core.catalog.schemas import Product
from core.retriever import Retriever

_INT_RE = re.compile(r"\d{2,}")


def parse_budget_cny(text: str) -> int | None:
    """从问句里抠预算数字（'预算3000以内' → 3000）；抠不到返回 None（视为不限）。"""
    for m in _INT_RE.findall(text):
        n = int(m)
        # 排除明显是年份/容量的数字噪声只留第一个像价格的数（2~6 位）
        if 100 <= n <= 9_999_999:
            return n
    return None


def _category_from_query(text: str) -> str | None:
    """问句里的品类词 → category 字面量（手机/笔记本/平板）；无则 None=全品类。"""
    if "笔记本" in text or "电脑" in text or "本 " in text:
        return "laptop"
    if "平板" in text:
        return "tablet"
    if "手机" in text or "机" in text:
        return "phone"
    return None


def _product_summary(p: Product, snippet: str = "") -> dict[str, Any]:
    return {
        "product_id": p.id,
        "name": p.name,
        "brand": p.brand,
        "model": p.model,
        "category": p.category,
        "price_cny": p.price_cny,
        "summary": p.summary,
        "snippet": snippet,
    }


class GuideTools:
    """持有 Catalog + Retriever 的工具集；方法名即决定层可见的工具名。"""

    def __init__(self, catalog: Catalog, retriever: Retriever) -> None:
        self._catalog = catalog
        self._retriever = retriever

    # ---- 工具 1：关键词召回 ----
    async def search_products(self, query: str, top_n: int = 5) -> dict[str, Any]:
        hits = await self._retriever.search(query)
        return {
            "ok": True,
            "tool": "search_products",
            "products": [
                _product_summary(self._catalog.get(h.product_id), h.snippet)  # type: ignore[arg-type]
                for h in hits[:top_n]
            ],
        }

    # ---- 工具 2：预算/品类/标签筛选（代码判定，模型不算）----
    def filter_products(
        self,
        budget_cny: int | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        top_n: int = 5,
    ) -> dict[str, Any]:
        matched: list[Product] = []
        for p in self._catalog.products:
            if category and p.category != category:
                continue
            if not p.matches_price(budget_cny):  # matches_price: price<=budget；None 不限
                continue
            if tags and not (set(tags) & set(p.tags)):
                continue
            matched.append(p)
        matched.sort(key=lambda p: p.price_cny)  # 预算内按价升序，展示"最省到最贵"
        return {
            "ok": True,
            "tool": "filter_products",
            "matched_count": len(matched),
            "total_count": len(self._catalog),
            "category": category,
            "budget_cny": budget_cny,
            "products": [_product_summary(p) for p in matched[:top_n]],
        }

    # ---- 工具 3：参数对比（只对齐共有规格键）----
    def compare_products(
        self, product_ids: list[str], dims: list[str] | None = None
    ) -> dict[str, Any]:
        products = self._catalog.get_many(product_ids[:3])
        if len(products) < 2:
            return {"ok": False, "error": f"对比至少需 2 个在库商品，现有 {len(products)} 个"}
        # 共有键：全部两两交集（简化为取首个商品与其余商品的共同键再迭代收缩）
        common: set[str] = set(products[0].specs)
        for p in products[1:]:
            common &= set(p.specs)
        dims = list(common) if dims is None else [d for d in dims if d in common]
        dims = sorted(dims)
        rows = [
            {
                "product_id": p.id,
                "name": p.name,
                "price_cny": p.price_cny,
                "dims": {d: p.specs.get(d, "") for d in dims},
            }
            for p in products
        ]
        return {
            "ok": True,
            "tool": "compare_products",
            "product_ids": [p.id for p in products],
            "dims": dims,
            "rows": rows,
        }
