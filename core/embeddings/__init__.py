# -*- coding: utf-8 -*-
"""Embedding 工厂：按 settings.embedding.provider 产出实现。

离线默认 mock；dashscope 为阶段 B 预留（未接 key 启动即 fail-fast，不会默默降级）。
"""
from __future__ import annotations

from config.settings import Settings, get_settings
from core.embeddings.base import EmbeddingProvider
from core.embeddings.mock_embedding import MockEmbedding


class DashScopeEmbedding:
    """阶段 B 预留：真实文本向量（text-embedding-v3，需 DASHSCOPE_API_KEY）。

    当前未接入 → 构造即抛，防止误以为已接真模型。
    """

    dimension: int

    def __init__(self, dimension: int = 1024, *, base_url: str = "", api_key: str | None = None) -> None:
        self.dimension = dimension
        if not api_key:
            raise RuntimeError(
                "DashScopeEmbedding 尚未接入：阶段 B 需先实现真 embedding 调用并配置 "
                "DASHSCOPE_API_KEY。离线请保持 embedding.provider=mock。"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    if s.embedding.provider == "mock":
        return MockEmbedding(dimension=s.embedding.dimension)
    if s.embedding.provider == "dashscope":
        from config.settings import load_dotenv  # noqa: F401  # 已在上层加载

        import os

        return DashScopeEmbedding(
            dimension=s.embedding.dimension,
            base_url=s.embedding.base_url,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
    raise ValueError(f"未知 embedding.provider：{s.embedding.provider}（可选 mock|dashscope）")


__all__ = ["DashScopeEmbedding", "EmbeddingProvider", "MockEmbedding", "get_embedding_provider"]
