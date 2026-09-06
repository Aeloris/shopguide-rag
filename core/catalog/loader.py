# -*- coding: utf-8 -*-
"""商品知识库加载：从 fixtures/catalog/products.json 校验并索引。

设计：
- 唯一数据源是 JSON（由 scripts/make_catalog.py 确定性生成并提交）。
- 加载时用 pydantic 逐条校验 → 坏数据启动即炸（fail fast），不会在检索/比价中途爆。
- 维护 id→Product 的索引与 list，供 Agent 三工具（检索/比价/筛选）共享同一个内存目录。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from config.settings import Settings, get_settings
from core.catalog.schemas import Product


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"商品知识库不存在：{path}\n请先运行 uv run python scripts/make_catalog.py 生成。"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"商品知识库 {path} 应为非空 JSON 数组")
    return data


class Catalog:
    """内存商品目录。全项目（retriever / agent 工具 / app）共享同一实例。"""

    def __init__(self, products: Iterable[Product]) -> None:
        items = list(products)
        if not items:
            raise ValueError("Catalog 不能为空")
        ids = [p.id for p in items]
        if len(set(ids)) != len(ids):
            dup = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"商品 id 重复：{sorted(dup)}")
        self._products = items
        self._by_id = {p.id: p for p in items}

    # ---- 查询 ----
    @property
    def products(self) -> list[Product]:
        return list(self._products)

    def get(self, product_id: str) -> Product | None:
        return self._by_id.get(product_id)

    def get_many(self, product_ids: Iterable[str]) -> list[Product]:
        """保序取回（未知 id 静默跳过；查重由调用方负责）。"""
        out = []
        for pid in product_ids:
            p = self._by_id.get(pid)
            if p:
                out.append(p)
        return out

    def by_category(self, category: str) -> list[Product]:
        return [p for p in self._products if p.category == category]

    def __len__(self) -> int:
        return len(self._products)


def load_catalog(settings: Settings | None = None) -> Catalog:
    s = settings or get_settings()
    path = s.repo_root / s.catalog.file
    raw = _load_raw(path)
    products = [Product(**r) for r in raw]
    return Catalog(products)
