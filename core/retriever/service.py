# -*- coding: utf-8 -*-
"""混合检索服务：BM25(词法) + Dense(向量) → RRF 融合 → 重排 → 商品候选。

链路（每查询一次）：
1. 惰性重建 BM25Index：BM25 用词法召回，Dense 用向量召回（各自 top_k）；
2. RRF 按"名次"而非"分数"融合两路（量纲不同不可直接相加，见 fusion.py）；
3. MockReranker 词法精排 top_n（阶段 B 可换真 CrossEncoder）；
4. 输出去重后的商品级候选，附命中摘录（溯源用）。
"""
from __future__ import annotations

from config.settings import Settings, get_settings
from core.embeddings.base import EmbeddingProvider
from core.embeddings.mock_embedding import MockEmbedding
from core.retriever.bm25 import BM25Index
from core.retriever.fusion import rrf_fuse
from core.retriever.rerank import MockReranker
from core.retriever.schemas import RetrievedProduct
from core.retriever.tokens import tokenize
from core.vector_store import ProductVectorStore


def _snippet_for(text: str, query: str, limit: int = 160) -> str:
    """摘录命中原文：优先取含 query token 的那一行，否则取开头。"""
    q_tokens = set(tokenize(query))
    for line in text.splitlines():
        if q_tokens & set(tokenize(line)):
            line = line.strip()
            return line if len(line) <= limit else line[:limit] + "…"
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[:limit] + "…"


class Retriever:
    def __init__(
        self,
        store: ProductVectorStore,
        embedding: EmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._embedding = embedding or MockEmbedding(dimension=store.dimension)
        self._s = settings or get_settings()
        r = self._s.retrieval
        self._dense_top_k = r.dense_top_k
        self._bm25_top_k = r.bm25_top_k
        self._rrf_k = r.rrf_k
        self._rerank_top_n = r.rerank_top_n
        self._final_top_n = r.final_top_n
        from core.retriever.rerank import MockReranker

        self._reranker = MockReranker(top_n=r.rerank_top_n)

    def _docs(self) -> list[dict]:
        """库内全部 payload（product_id/text/…）；排序稳定 → BM25 doc_index 可复现。

        不缓存：入库整包重建后必须读到新数据（语料=13 SKU，全量代价可忽略）。
        """
        return sorted(self._store.all_docs(), key=lambda d: d["product_id"])

    def _bm25(self) -> BM25Index:
        return BM25Index([d["text"] for d in self._docs()])

    async def search(self, query: str) -> list[RetrievedProduct]:
        """返回按质量排序的商品候选（已去重到 SKU 粒度）。"""
        docs = self._docs()
        if not docs:
            return []
        pid_to_idx = {d["product_id"]: i for i, d in enumerate(docs)}

        # 1) Dense 路
        q_vec = (await self._embedding.embed([query]))[0]
        dense_hits = self._store.search(q_vec, top_k=self._dense_top_k)
        dense_ranked = [pid_to_idx[h["product_id"]] for h in dense_hits if h["product_id"] in pid_to_idx]

        # 2) BM25 路
        bm25_hits = self._bm25().search(query, top_k=self._bm25_top_k)
        bm25_ranked = [i for i, _ in bm25_hits]

        # 3) RRF 融合（只喂非空路）
        fused = rrf_fuse(dense_ranked, bm25_ranked, k=self._rrf_k) if (dense_ranked or bm25_ranked) else []

        # 4) 重排：把融合序的 doc_index 再精排（Mock=词法）；无重排器则退回融合序
        reranked = self._reranker.rerank(query, fused, [d["text"] for d in docs])
        ordered = [it.doc_index for it in reranked] if reranked else fused
        if not ordered:
            return []

        # 5) 商品级输出 + 溯源摘录（rank 是最终序，供评估/调试可观测）
        texts_by_idx = [d["text"] for d in docs]
        out: list[RetrievedProduct] = []
        for rank, idx in enumerate(ordered[: self._final_top_n], start=1):
            d = docs[idx]
            out.append(
                RetrievedProduct(
                    product_id=d["product_id"],
                    name=d.get("name", d["product_id"]),
                    brand=d.get("brand", ""),
                    category=d.get("category", ""),
                    price_cny=d.get("price_cny", 0),
                    rrf_score=round(1.0 / (self._rrf_k + rank), 6),
                    snippet=_snippet_for(texts_by_idx[idx], query),
                    ranks={"rerank": rank},
                )
            )
        return out
