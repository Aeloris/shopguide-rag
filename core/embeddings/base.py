# -*- coding: utf-8 -*-
"""Embedding Provider 协议（与 bid 引擎同哲学）。

业务代码只依赖 `embed(texts) -> list[list[float]]`：
- provider=mock → MockEmbedding（确定性伪向量，离线/测试用）；
- provider=dashscope → DashScopeEmbedding（阶段 B，需 key + 真接入）。
切 provider 业务代码零改动。
"""
from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
