# -*- coding: utf-8 -*-
"""商品知识库数据契约（pydantic v2）。

字段设计考量：
- specs 用 dict[str, str] 而非扁平字段：不同品类（手机/笔记本/平板）的规格键天然不同，
  统一到一张扁平表会塞满 None；dict 保留品类自由度，比价时取两品**共有的键**即可。
- price_cny 用 int（元）；不要浮点价格，避免"xx.9 元"类陷阱，比价/筛选确定性。
- category 限定三类，保证"参数对比工具"能在同类内做有意义的行对齐。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProductCategory = Literal["phone", "laptop", "tablet"]

# 各类别的"可选规格键"说明文档用途（不做运行时强约束，仅给种子数据作者参考）
CATEGORY_SPEC_KEYS: dict[str, list[str]] = {
    "phone": ["屏幕", "处理器", "运行内存", "存储", "后置相机", "电池容量", "充电功率", "重量", "操作系统", "5G"],
    "laptop": ["屏幕尺寸", "处理器", "内存", "存储", "显卡", "续航", "重量", "操作系统", "触控屏"],
    "tablet": ["屏幕尺寸", "处理器", "内存", "存储", "续航", "重量", "手写笔支持"],
}


class Product(BaseModel):
    """一件可被检索/比价/筛选的 SKU。"""

    id: str = Field(description="稳定唯一键，如 'xiaomi-14'；被引用溯源 [P:xxx]")
    brand: str
    model: str
    name: str = Field(description="展示名，如 '小米 14'")
    category: ProductCategory
    price_cny: int
    summary: str = Field(description="一句话卖点（进语料，供语义检索）")
    highlights: list[str] = Field(default_factory=list, description="核心卖点短句（进语料）")
    specs: dict[str, str] = Field(default_factory=dict, description="结构化规格，键值均为展示串")
    tags: list[str] = Field(default_factory=list, description="人群/场景标签：拍照/游戏/办公/轻薄/长续航/性价比")

    # ---- 供 Agent 工具 / 比价用的派生字段 ----
    @property
    def display_price(self) -> str:
        return f"¥{self.price_cny}"

    def spec_keys_common(self, other: "Product") -> list[str]:
        """两品共有的规格键（比价只对齐共有维度，避免缺列）。"""
        return [k for k in self.specs if k in other.specs]

    def matches_price(self, budget_cny: int | None) -> bool:
        """预算筛选：budget 为空视为不限；price<=budget 通过。"""
        return budget_cny is None or self.price_cny <= budget_cny


class ProductRef(BaseModel):
    """检索命中在 Agent 侧的最小呈现单元（去重到 SKU 粒度）。"""

    product_id: str
    name: str
    brand: str
    price_cny: int
    score: float = Field(description="RRF 融合后该品最佳命中分（可观测）")
    matched_snippet: str = Field(default="", description="命中块原文摘录（引用溯源用）")
