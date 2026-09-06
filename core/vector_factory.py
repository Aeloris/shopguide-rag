# -*- coding: utf-8 -*-
"""按 settings.vector_db.provider 构造 ProductVectorStore：
- qdrant_local：无 Docker，path 指向 data/qdrant（默认），或 ":memory:"（测试注入）；
- qdrant_server：阶段 B，url 连 Docker 里的 Qdrant 容器。
测试与业务共用同一工厂，测试用 tmp_path 保证隔离、不污染 ./data。
"""
from __future__ import annotations

from config.settings import Settings, get_settings
from core.vector_store import ProductVectorStore


def build_store(
    settings: Settings | None = None, *, dimension: int | None = None, path_override: str | None = None
) -> ProductVectorStore:
    s = settings or get_settings()
    dim = dimension if dimension is not None else s.embedding.dimension
    if s.vector_db.provider == "qdrant_local":
        p = path_override if path_override is not None else str(s.repo_root / s.vector_db.path)
        return ProductVectorStore(
            collection=s.vector_db.collection, dimension=dim, path=p
        )
    if s.vector_db.provider == "qdrant_server":
        return ProductVectorStore(
            collection=s.vector_db.collection, dimension=dim, url=s.vector_db.url
        )
    raise ValueError(f"未知 vector_db.provider：{s.vector_db.provider}（可选 qdrant_local|qdrant_server）")
