# -*- coding: utf-8 -*-
"""核心链路冒烟：Catalog → Ingest → Retriever（内存 Qdrant + Mock embedding）。

用法：uv run python scripts/smoke_core.py
通过则打印检索命中，供快速人肉验证；非正式评测（评测走 evals/run.py）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import get_settings
from core.catalog.loader import load_catalog
from core.embeddings.mock_embedding import MockEmbedding
from core.ingest import Ingester
from core.retriever import Retriever
from core.vector_store import ProductVectorStore


async def main() -> None:
    s = get_settings()
    catalog = load_catalog(s)
    store = ProductVectorStore(
        collection="smoke", dimension=s.embedding.dimension, path=":memory:"
    )
    emb = MockEmbedding(dimension=s.embedding.dimension)
    await Ingester(store, emb).rebuild(catalog)
    print(f"入库商品数: {store.count()}")
    retriever = Retriever(store, emb, s)
    for q in ["预算3000以内的安卓直屏手机", "适合办公的轻薄笔记本", "512g 拍人像好的手机"]:
        hits = await retriever.search(q)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  [{h.ranks.get('rerank','?')}] {h.product_id} ¥{h.price_cny} :: {h.snippet[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
