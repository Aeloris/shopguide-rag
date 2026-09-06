# -*- coding: utf-8 -*-
"""Embedding 工厂：按 settings.embedding.provider 产出实现。

离线默认 mock；dashscope 为真文本向量（text-embedding-v3，OpenAI 兼容 /embeddings），
需 DASHSCOPE_API_KEY —— 无 key 构造即 fail-fast，不会默默降级。
"""
from __future__ import annotations

import os

import httpx

from config.settings import Settings, get_settings
from core.embeddings.base import EmbeddingProvider
from core.embeddings.mock_embedding import MockEmbedding

# DashScope text-embedding-v3 单请求上限（实测 >10 返回 400 InvalidParameter）
_MAX_EMBED_BATCH = 10


class DashScopeEmbedding:
    """真文本向量：阿里云百炼 text-embedding-v3（OpenAI 兼容接口）。

    - 输出维度与 vector_db 集合一致（config.embedding.dimension，v3 默认 1024）；
    - 同批 input 一次请求，返回序与输入序一致（data.index 对齐回填）；
    - key 只从环境变量读，绝不落代码/配置。
    """

    dimension: int

    def __init__(
        self,
        dimension: int = 1024,
        *,
        model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: str | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        self.dimension = dimension
        self._model = model
        self._base = (base_url or "").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_sec
        if not self._api_key:
            raise RuntimeError(
                "embedding.provider=dashscope 但环境变量 DASHSCOPE_API_KEY 未设置。\n"
                "解决办法：复制 .env.example 为 .env 填入阿里云百炼 key；"
                "或保持 embedding.provider=mock。"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # 端点单请求上限 _MAX_EMBED_BATCH 条 → 分批，返回序与输入序一致
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for i in range(0, len(texts), _MAX_EMBED_BATCH):
                chunk = texts[i : i + _MAX_EMBED_BATCH]
                payload: dict = {
                    "model": self._model,
                    "input": chunk,
                    "encoding_format": "float",
                }
                # text-embedding-v3 支持 dimensions≤1024 控制输出维；等于默认(1024)时省略即可
                if 0 < self.dimension < 1024:
                    payload["dimensions"] = self.dimension
                headers = {"Authorization": f"Bearer {self._api_key}"}
                resp = await client.post(
                    f"{self._base}/embeddings", headers=headers, json=payload
                )
                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DashScope embedding HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                data = resp.json().get("data", [])
                ordered = [
                    d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))
                ]
                if len(ordered) != len(chunk):
                    raise RuntimeError(
                        f"DashScope embedding 返回 {len(ordered)} 条，输入 {len(chunk)} 条"
                    )
                vectors.extend(ordered)
        return vectors


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    s = settings or get_settings()
    if s.embedding.provider == "mock":
        return MockEmbedding(dimension=s.embedding.dimension)
    if s.embedding.provider == "dashscope":
        return DashScopeEmbedding(
            dimension=s.embedding.dimension,
            model=s.embedding.model,
            base_url=s.embedding.base_url,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            timeout_sec=s.llm.timeout_sec,
        )
    raise ValueError(f"未知 embedding.provider：{s.embedding.provider}（可选 mock|dashscope）")


__all__ = ["DashScopeEmbedding", "EmbeddingProvider", "MockEmbedding", "get_embedding_provider"]
