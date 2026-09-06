# -*- coding: utf-8 -*-
"""评测 harness：指标算术（stub 检索器）+ 数据集完整性（gold 都真实可达）。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.catalog.loader import Catalog
from evals.dataset import RETRIEVAL_CASES, RetrievalCase
from evals.retrieval_eval import evaluate_retriever


class _StubRetriever:
    """只实现 .search() 的假检索器：query → 带 product_id 的对象列表。"""

    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self._mapping = mapping

    async def search(self, query: str):
        return [SimpleNamespace(product_id=pid) for pid in self._mapping.get(query, [])]


def test_metrics_arithmetic() -> None:
    retriever = _StubRetriever(
        {
            "q1": ["a", "x", "b", "c", "d"],   # a 首中 rank1
            "q2": ["x", "y", "a"],             # a 首中 rank3
            "q3": ["x", "y"],                  # gold zz 全落空
        }
    )
    cases = [
        RetrievalCase(question="q1", gold_product_ids=["a", "b"]),
        RetrievalCase(question="q2", gold_product_ids=["a"]),
        RetrievalCase(question="q3", gold_product_ids=["zz"]),
    ]
    report = asyncio.run(
        evaluate_retriever(retriever, cases, top_k=5, recall_threshold=0.5, mrr_threshold=0.4)
    )
    # safe_mean round 到 4 位：recall=2/3→0.6667，mrr=(1+1/3)/3=4/9→0.4444
    assert report.avg_recall_at_k == pytest.approx(round((1 + 1 + 0) / 3, 4))
    assert report.avg_mrr_at_k == pytest.approx(round((1 + 1 / 3 + 0) / 3, 4))
    assert report.n_cases == 3 and report.n_passed == 2
    assert report.gate_ok  # recall 0.667≥0.5 且 mrr 0.444≥0.4


def test_gold_hit_in_topk_rank_tracking() -> None:
    """全落空 → 该例 recall/mrr=0；给定正阈值(0.5)时 passed=False。"""
    retriever = _StubRetriever({"q": ["x", "y"]})
    cases = [RetrievalCase(question="q", gold_product_ids=["miss"])]
    report = asyncio.run(
        evaluate_retriever(retriever, cases, top_k=5, recall_threshold=0.5, mrr_threshold=0.5)
    )
    c = report.cases[0]
    assert (c.recall_at_k, c.mrr_at_k) == (0.0, 0.0)
    assert not c.passed  # 0.0 < 0.5


def test_empty_gold_raises() -> None:
    """gold 空 → 指标无意义，harness 直接拒绝（dataset 生成时就保证非空）。"""
    retriever = _StubRetriever({"q": ["x"]})
    cases = [RetrievalCase(question="q", gold_product_ids=[])]
    with pytest.raises(ValueError, match="缺 gold"):
        asyncio.run(evaluate_retriever(retriever, cases, top_k=5))


def test_dataset_gold_all_reachable(catalog: Catalog) -> None:
    """每条 gold 的 SKU 必须真实存在：gold 指向不存在的 id 会让门禁永远失败。"""
    known = {p.id for p in catalog.products}
    assert len(RETRIEVAL_CASES) >= 10
    for case in RETRIEVAL_CASES:
        assert case.gold_product_ids, f"{case.question!r} 缺 gold"
        missing = set(case.gold_product_ids) - known
        assert not missing, f"{case.question!r} 引用不存在 SKU：{sorted(missing)}"
