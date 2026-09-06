# -*- coding: utf-8 -*-
"""商品级文档向量库（Qdrant 三态封装）。

与 bid 引擎同一设计哲学：qdrant-client 本地模式(path/:memory:)的 API 与正式
Qdrant 服务完全一致 → 无 Docker 可离线入库/检索；上生产把构造从 path 换成 url，
上层代码零改动（config.vector_db.provider 切换）。

检索原子 = 商品：一份 Product 渲染成 Markdown 文档（core.catalog.docs），
作为一个点入库。payload 存结构化元数据 + 全文 → 命中即拿原文做溯源/引用。
点 id 用 product_id 的内容哈希转 uint64 → 同 SKU 幂等 upsert。
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from qdrant_client import QdrantClient, models


def product_id_to_uint64(product_id: str) -> int:
    """把商品 id 稳定映射为 Qdrant 点 id（uint64）。"""
    return int(hashlib.sha1(product_id.encode("utf-8")).hexdigest()[:16], 16)


class ProductVectorStore:
    def __init__(
        self,
        collection: str,
        dimension: int,
        *,
        path: str | Path | None = None,
        url: str | None = None,
    ) -> None:
        if url:
            self._client = QdrantClient(url=url)  # 生产：连 Qdrant 服务器
        elif path == ":memory:" or path is None:
            self._client = QdrantClient(path=":memory:")  # 测试：内存
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))  # 本地持久化：无 docker
        self.collection = collection
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self.collection):
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimension, distance=models.Distance.COSINE
                ),
            )

    def delete_collection(self) -> None:
        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)

    def reset(self) -> None:
        """清空并按当前维度重建集合（整包重建语义）。"""
        self.delete_collection()
        self._ensure_collection()

    def upsert_docs(
        self, docs: Iterable[tuple[str, str, dict[str, Any]]], vectors: list[list[float]]
    ) -> int:
        """批量写入商品文档。

        参数：docs = [(product_id, text, extra_meta)]，vectors 与之同序。
        payload = {"product_id", "text", **extra_meta}。点 id 由 product_id 哈希派生。
        """
        items = list(docs)
        if len(items) != len(vectors):
            raise ValueError(f"docs({len(items)}) 与 vectors({len(vectors)}) 数量不一致")
        points = [
            models.PointStruct(
                id=product_id_to_uint64(pid),
                vector=vec,
                payload={"product_id": pid, "text": text, **(meta or {})},
            )
            for (pid, text, meta), vec in zip(items, vectors)
        ]
        self._client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """按向量召回，返回按相似度降序的 payload（附 score 与向量内名次）。"""
        res = self._client.query_points(
            collection_name=self.collection, query=vector, limit=top_k
        )
        out: list[dict[str, Any]] = []
        for rank, hit in enumerate(res.points, start=1):
            payload = dict(hit.payload or {})
            payload["_score"] = float(hit.score)
            payload["_dense_rank"] = rank
            out.append(payload)
        return out

    def count(self) -> int:
        return int(self._client.count(collection_name=self.collection, exact=True).count)

    def all_docs(self) -> list[dict[str, Any]]:
        """全量取出 payload（BM25 需要全量重建词法索引；语料=13 SKU，量级极小）。"""
        out: list[dict[str, Any]] = []
        points, _ = self._client.scroll(
            collection_name=self.collection, limit=10_000, with_vectors=False
        )
        for p in points:
            out.append(dict(p.payload or {}))
        return out
