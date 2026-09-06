# -*- coding: utf-8 -*-
"""商品入库流水线：Catalog → 每 SKU 渲染 Markdown → 向量化 → 写入 ProductVectorStore。

设计：
- 幂等整包重建：reset() 清空集合 → 全部 SKU 重写。语料=13 商品量级极小，
  每次入库全量重建最简单可靠（无需增量/upsert 冲突处理）。
- 每个商品一条文档：payload 含 product_id/name/brand/category/price_cny/text。
  检索命中即拿 product_id 溯源，text 供 BM25 与摘录。
"""
from __future__ import annotations

from core.catalog.docs import product_to_markdown
from core.catalog.loader import Catalog
from core.embeddings.base import EmbeddingProvider
from core.vector_store import ProductVectorStore


class Ingester:
    def __init__(
        self,
        store: ProductVectorStore,
        embedding: EmbeddingProvider,
    ) -> None:
        self._store = store
        self._embedding = embedding

    async def rebuild(self, catalog: Catalog) -> int:
        """整包重建索引，返回入库商品数。"""
        docs = []
        texts: list[str] = []
        for p in catalog.products:
            text = product_to_markdown(p)
            docs.append(
                (
                    p.id,
                    text,
                    {
                        "name": p.name,
                        "brand": p.brand,
                        "category": p.category,
                        "price_cny": p.price_cny,
                    },
                )
            )
            texts.append(text)
        vectors = await self._embedding.embed(texts)
        self._store.reset()
        n = self._store.upsert_docs(docs, vectors)
        # 清掉旧 BM25/全量缓存（key 不变但底层集合变了）
        return n
