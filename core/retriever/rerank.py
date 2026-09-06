# -*- coding: utf-8 -*-
"""重排（rerank）：把 RRF 融合后的候选再精排。

现状诚实口径：
- provider=mock：词法重叠分（query 与文档命中 token 覆盖），离线确定性，验证链路；
- provider=dashscope/gte-rerank 等真模型：阶段 B 预留（CrossEncoder 风格）。
重排是可选阶段：小语料下 RRF 后直接取 top_n 也够用，但保留该阶段以对齐
「RRF 融合 → CrossEncoder 精排」的生产 RAG 范式。
"""
from __future__ import annotations

from dataclasses import dataclass

from core.retriever.tokens import tokenize


@dataclass
class RerankItem:
    doc_index: int
    score: float


class MockReranker:
    """词法重排：按 query 命中 token 覆盖度给候选项打分（确定性，离线用）。"""

    def __init__(self, top_n: int = 8) -> None:
        self.top_n = top_n

    def rerank(self, query: str, candidates: list[int], texts: list[str]) -> list[RerankItem]:
        """candidates = RRF 后的 doc_index 列表（已按融合分降序）；texts 为该库全文。"""
        q = set(tokenize(query))
        scored: list[RerankItem] = []
        for idx in candidates:
            doc_tokens = set(tokenize(texts[idx]))
            overlap = len(q & doc_tokens)
            scored.append(RerankItem(doc_index=idx, score=float(overlap)))
        # 先按重排分降序，再按原 RRF 顺序兜底 → 稳定
        rank_of = {idx: i for i, idx in enumerate(candidates)}
        scored.sort(key=lambda it: (-it.score, rank_of[it.doc_index]))
        return scored[: self.top_n]
