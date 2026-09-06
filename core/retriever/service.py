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


# ---------------- 域外门禁（真商品图 e2e 曝光：检索无相关度地板 → 域外 query 也硬推 3C）-----
#
# 判定：问句与库内商品**零词法重叠**（len>=2 的 token：CJK 双字 / 整段型号）且**顶配向量分低于
# 地板** → 用户要的东西这家店根本不卖（洗洁精/抽纸…）→ 直接判无匹配，不给 5 个凑数的 3C。
# 词法零重叠是主信号（对封闭目录稳定、embedding 无关）；向量分是真 embedding 时的语义兜底——
# 放行"零词面但语义强、实属店内"的问句，避免真向量下误拒。mock 向量无语义，离线靠词法零重叠撑住。
# 诚实边界（README 同述）：机械键盘/游戏耳机/显示器等"外设配件"与库内笔记本共享规格词
# （键盘/散热/屏），词法重叠判定认为它们在语域内 → 不会被本门禁拦截 —— 真店靠"库存品类"判定，
# 属下一增量，此处不虚标。


def _catalog_vocab(doc_texts: list[str]) -> set[str]:
    """语料词表：所有 doc 中 len>=2 的检索 token（CJK 双字 / 整段英数型号）。"""
    vocab: set[str] = set()
    for text in doc_texts:
        for t in tokenize(text):
            if len(t) >= 2:
                vocab.add(t)
    return vocab


def has_catalog_overlap(query: str, vocab: set[str]) -> bool:
    """问句与语料共享任一 len>=2 token（品类/规格词面可达）→ 属店内语域候选。"""
    for t in tokenize(query):
        if len(t) >= 2 and t in vocab:
            return True
    return False


def is_out_of_catalog(
    query: str, doc_texts: list[str], top_dense_score: float, dense_floor: float
) -> bool:
    """真·域外（本店不卖）判定；空库不参与（空结果本就由上层处理）。"""
    if not doc_texts:
        return False
    if has_catalog_overlap(query, _catalog_vocab(doc_texts)):
        return False
    return top_dense_score < dense_floor


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
        self._dense_floor = r.dense_match_floor
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
        # 域外门禁：零词法重叠 + 顶配向量分 < 地板 → 本店不卖这类商品 → 无匹配（不给凑数 3C）
        doc_texts = [d["text"] for d in docs]
        top_dense = dense_hits[0]["_score"] if dense_hits else 0.0
        if is_out_of_catalog(query, doc_texts, top_dense, self._dense_floor):
            return []
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
