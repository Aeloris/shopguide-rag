# -*- coding: utf-8 -*-
"""入库流水线与向量库封装：幂等整包重建 / payload 结构 / 检索打分。"""
from __future__ import annotations

import asyncio

import pytest

from config.settings import get_settings
from core.catalog.loader import Catalog
from core.embeddings.mock_embedding import MockEmbedding
from core.ingest import Ingester
from core.vector_store import ProductVectorStore, product_id_to_uint64


def test_product_id_hash_stable_and_bounded() -> None:
    a = product_id_to_uint64("xiaomi-14")
    b = product_id_to_uint64("xiaomi-14")
    assert a == b and 0 <= a < 2**64
    assert product_id_to_uint64("redmi-k70") != a


def test_upsert_length_mismatch_raises(fresh_store: ProductVectorStore) -> None:
    with pytest.raises(ValueError, match="数量不一致"):
        fresh_store.upsert_docs([("a", "text", {})], [[1.0], [2.0]])


def test_rebuild_is_idempotent_and_has_full_payload(
    catalog: Catalog, fresh_store: ProductVectorStore
) -> None:
    emb = MockEmbedding(dimension=get_settings().embedding.dimension)
    n1 = asyncio.run(Ingester(fresh_store, emb).rebuild(catalog))
    assert n1 == len(catalog) == 13
    assert fresh_store.count() == 13
    # 整包重建第二次：清空后重写，数量不变、无残留
    n2 = asyncio.run(Ingester(fresh_store, emb).rebuild(catalog))
    assert n2 == 13 and fresh_store.count() == 13

    docs = fresh_store.all_docs()
    assert len(docs) == 13
    d = next(x for x in docs if x["product_id"] == "xiaomi-14")
    for k in ("name", "brand", "category", "price_cny", "text"):
        assert k in d, f"payload 缺 {k}"
    assert d["price_cny"] == 3999
    assert d["brand"] == "小米"
    assert d["category"] == "phone"
    assert "## 规格参数" in d["text"]  # Markdown 文档含规格节


def test_search_returns_scored_payloads(
    catalog: Catalog, fresh_store: ProductVectorStore
) -> None:
    s = get_settings()
    emb = MockEmbedding(dimension=s.embedding.dimension)
    asyncio.run(Ingester(fresh_store, emb).rebuild(catalog))

    async def _go() -> list[dict]:
        vec = (await emb.embed(["手机 快充 直屏"]))[0]
        return fresh_store.search(vec, top_k=3)

    hits = asyncio.run(_go())
    assert len(hits) == 3
    for h in hits:
        assert h.get("_dense_rank") == hits.index(h) + 1
        assert "_score" in h and h["_score"] >= 0.0
        assert h["product_id"]
