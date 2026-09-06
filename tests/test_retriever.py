# -*- coding: utf-8 -*-
"""检索链路 e2e：Catalog → Ingest → Retriever(BM25+Dense→RRF→重排) 的确定性行为。"""
from __future__ import annotations

import asyncio

import pytest

from config.settings import get_settings
from core.embeddings.mock_embedding import MockEmbedding
from core.retriever import Retriever
from core.retriever.schemas import RetrievedProduct
from core.vector_store import ProductVectorStore


def _search(retriever: Retriever, query: str) -> list[RetrievedProduct]:
    return asyncio.run(retriever.search(query))


def test_empty_store_returns_empty() -> None:
    s = get_settings()
    store = ProductVectorStore(
        collection="empty", dimension=s.embedding.dimension, path=":memory:"
    )
    retriever = Retriever(store, MockEmbedding(dimension=s.embedding.dimension), s)
    assert _search(retriever, "手机") == []


def test_result_shape_is_sku_deduplicated(
    built_retriever: Retriever,
) -> None:
    hits = _search(built_retriever, "预算3000以内的安卓直屏手机")
    assert 0 < len(hits) <= get_settings().retrieval.final_top_n
    ids = [h.product_id for h in hits]
    assert len(ids) == len(set(ids)), "应按 SKU 粒度去重"
    for h in hits:
        assert h.name and h.brand and h.snippet
        assert isinstance(h.price_cny, int) and h.price_cny > 0
        assert h.ranks.get("rerank") is not None


def test_rrf_score_descends_with_output_rank(
    built_retriever: Retriever,
) -> None:
    hits = _search(built_retriever, "512g 拍人像好的手机")
    scores = [h.rrf_score for h in hits]
    assert scores == sorted(scores, reverse=True)  # 输出序=名次序 → RRF 分严格递减


def test_budget_phone_query_recalls_redmi(
    built_retriever: Retriever,
) -> None:
    """预算敏感问法：红米 K70（2499）应居首，而非更贵的旗舰。"""
    hits = _search(built_retriever, "预算3000以内的安卓直屏手机")
    assert hits[0].product_id == "redmi-k70"
    assert "redmi-k70" in [h.product_id for h in hits[:5]]


def test_office_light_laptop_query(built_retriever: Retriever) -> None:
    hits = _search(built_retriever, "适合办公的轻薄笔记本")
    top = [h.product_id for h in hits]
    assert top[0] == "macbook-air-m3"
    # 前二应是笔记本；轻薄+办公语义由词法重排撑住（见 smoke）
    assert "thinkpad-x1-carbon" in top[:3]


def test_photo_phone_query(built_retriever: Retriever) -> None:
    hits = _search(built_retriever, "512g 拍人像好的手机")
    top3 = [h.product_id for h in hits[:3]]
    assert "vivo-x100" in top3 or "xiaomi-14" in top3
    # 问"手机"不应被笔记本/平板顶到最前
    assert hits[0].category == "phone"


def test_dense_path_and_bm25_path_both_present(built_retriever: Retriever) -> None:
    """返回序是 RRF 融合结果：候选数=融合后候选而非单路 top_k 直出。"""
    hits = _search(built_retriever, "适合办公的轻薄笔记本")
    assert len(hits) == get_settings().retrieval.final_top_n
