# -*- coding: utf-8 -*-
"""检索层数据契约。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedProduct(BaseModel):
    """检索层返回的候选商品（RRF 融合 + 重排后，去重到 SKU 粒度）。"""

    product_id: str
    name: str
    brand: str
    category: str
    price_cny: int
    rrf_score: float = Field(default=0.0, description="RRF 融合分（跨路可比的排名分）")
    snippet: str = Field(default="", description="命中原文摘录（引用溯源，防幻觉）")
    ranks: dict[str, int] = Field(default_factory=dict, description="各路名次：dense/bm25/rerank")
