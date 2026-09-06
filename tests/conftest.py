# -*- coding: utf-8 -*-
"""pytest 公共夹具 + Windows GBK 控制台兜底（否则中文 / ¥ 打印即崩）。

Windows 下 Python 默认 stdout 是 cp936(GBK)，¥(U+00A5) 与部分中文打印会抛
UnicodeEncodeError。这里在 import 期把 stdout/stderr 重配为 UTF-8(errors=replace)，
保证测试输出与断言失败信息在中文环境可读、不中断套件。
"""
from __future__ import annotations

import asyncio
import sys

import pytest

from config.settings import get_settings
from core.agent import AgentRuntime
from core.catalog.loader import Catalog, load_catalog
from core.embeddings.mock_embedding import MockEmbedding
from core.ingest import Ingester
from core.retriever import Retriever
from core.vector_store import ProductVectorStore


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


_force_utf8_streams()


@pytest.fixture(scope="session")
def catalog() -> Catalog:
    """读 fixtures/catalog/products.json 的种子目录（13 SKU，确定性）。"""
    return load_catalog(get_settings())


@pytest.fixture
def fresh_store() -> ProductVectorStore:
    """每次独立的空内存向量库。"""
    s = get_settings()
    return ProductVectorStore(
        collection="core-tests", dimension=s.embedding.dimension, path=":memory:"
    )


@pytest.fixture
def built_retriever(catalog) -> Retriever:
    """整包入库后的内存检索器（独立实例，mock embedding + Qdrant 内存，确定性）。"""
    s = get_settings()
    store = ProductVectorStore(
        collection="core-tests", dimension=s.embedding.dimension, path=":memory:"
    )
    emb = MockEmbedding(dimension=s.embedding.dimension)
    asyncio.run(Ingester(store, emb).rebuild(catalog))
    return Retriever(store, emb, s)


@pytest.fixture(scope="session")
def runtime() -> AgentRuntime:
    """离线 Agent 运行时（Catalog + 内存向量库 + LangGraph 图，只读复用，确定性）。"""
    return asyncio.run(AgentRuntime.build(get_settings()))
