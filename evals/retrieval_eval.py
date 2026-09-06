# -*- coding: utf-8 -*-
"""检索评测 harness：对 Retriever 跑 gold 集 → 每例 Recall@k / MRR@k + 汇总。

纯调用方逻辑：指标本身在 metrics.py（确定性代码计数），本模块只负责
"把检索 top-k 结果喂进指标、逐例归档、聚合均值"，不引入任何引擎依赖以外的 I/O。
检索是 async（embed 可异步），本模块提供 async 入口；同步环境用 asyncio.run 包。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.retriever import Retriever
from evals.dataset import RetrievalCase
from evals.metrics import mrr_at_k, recall_at_k, safe_mean


@dataclass
class RetrievalCaseResult:
    """单条 gold 的检索结果（供报告表格/失败定位）。"""

    question: str
    gold_product_ids: list[str]
    hit_product_ids: list[str]
    recall_at_k: float
    mrr_at_k: float
    passed: bool


@dataclass
class RetrievalReport:
    """聚合汇总（均值为 None 表示无有效样例，理论上不发生：gold 非空校验在 dataset）。"""

    cases: list[RetrievalCaseResult]
    avg_recall_at_k: float | None
    avg_mrr_at_k: float | None
    threshold_recall: float | None = None
    threshold_mrr: float | None = None

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.cases if c.passed)

    @property
    def gate_ok(self) -> bool:
        """门禁判定：在给定阈值下均值达标。无阈值(传入 None)则只算不判。"""
        ok_r = self.avg_recall_at_k is not None and (
            self.threshold_recall is None or self.avg_recall_at_k >= self.threshold_recall
        )
        ok_m = self.avg_mrr_at_k is not None and (
            self.threshold_mrr is None or self.avg_mrr_at_k >= self.threshold_mrr
        )
        return bool(ok_r and ok_m)


async def evaluate_retriever(
    retriever: Retriever,
    cases: list[RetrievalCase],
    *,
    top_k: int,
    recall_threshold: float | None = None,
    mrr_threshold: float | None = None,
) -> RetrievalReport:
    """对每条 case 检索并把 top_k 的 product_id 喂给指标；返回逐例 + 聚合。"""
    results: list[RetrievalCaseResult] = []
    for case in cases:
        hits = await retriever.search(case.question)
        hit_ids = [h.product_id for h in hits[:top_k]]
        # 指标要求 gold 非空：dataset 保证非空，这里仍防御（空 gold 记 0 但跳过聚合？）
        if not case.gold_product_ids:
            raise ValueError(f"RetrievalCase 缺 gold：{case.question!r}")
        r = recall_at_k(hit_ids, case.gold_product_ids, k=top_k)
        m = mrr_at_k(hit_ids, case.gold_product_ids, k=top_k)
        results.append(
            RetrievalCaseResult(
                question=case.question,
                gold_product_ids=list(case.gold_product_ids),
                hit_product_ids=hit_ids,
                recall_at_k=r,
                mrr_at_k=m,
                passed=(recall_threshold is None or r >= recall_threshold),
            )
        )
    recalls = [c.recall_at_k for c in results]
    mrrs = [c.mrr_at_k for c in results]
    return RetrievalReport(
        cases=results,
        avg_recall_at_k=safe_mean(recalls),
        avg_mrr_at_k=safe_mean(mrrs),
        threshold_recall=recall_threshold,
        threshold_mrr=mrr_threshold,
    )
