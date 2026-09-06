# -*- coding: utf-8 -*-
"""Agent 运行时门面：一次构建（Catalog+检索器+图），供多次 ask。

对外契约只有两个：
    AgentRuntime.build(settings) -> 自举 Catalog/向量库/Ingest/Retriever/LangGraph 图
    await runtime.ask(query, *, images) -> AgentReply（结构化回执，渲染层再转文本）

多模态：images 非空时先经 VisionProvider 抽成文本语义并入 query（v1 mock 固定返回；
阶段 B 换 anthropic 视觉）。history（多轮记忆）由上层会话存储承接，v1 单轮无状态作答。
"""
from __future__ import annotations

from dataclasses import dataclass

from config.settings import Settings, get_settings
from core.agent.graph import build_ctx, build_graph
from core.agent.schemas import AgentReply
from core.catalog.loader import Catalog, load_catalog
from core.embeddings import EmbeddingProvider, get_embedding_provider
from core.ingest import Ingester
from core.retriever import Retriever
from core.vector_factory import build_store
from llm.vision import VisionExtraction, get_vision_provider


@dataclass
class AgentRuntime:
    settings: Settings
    catalog: Catalog
    retriever: Retriever
    graph: object  # CompiledStateGraph

    @classmethod
    async def build(
        cls,
        settings: Settings | None = None,
        *,
        path: str = ":memory:",
        embedding: EmbeddingProvider | None = None,
    ) -> "AgentRuntime":
        """自举 Agent 运行时。

        向量库由 core.vector_factory.build_store 按 settings.vector_db.provider 构造：
        - qdrant_local：path=:memory:（测试）或本地 data/qdrant（离线可持久化）；
        - qdrant_server：阶段 B，连 Docker 里 Qdrant 容器（url 由 config 给），
          path 参数被忽略 —— 同一集合名 shopguide_products 入库/检索。
        """
        s = settings or get_settings()
        catalog = load_catalog(s)
        store = build_store(s, dimension=s.embedding.dimension, path_override=path)
        emb = embedding or get_embedding_provider(s)
        await Ingester(store, emb).rebuild(catalog)
        retriever = Retriever(store, emb, s)

        # 决定器：llm.provider=mock → 确定性规则；=anthropic → 真 Claude 选动作（回落规则）
        if s.llm.provider == "anthropic":
            from llm.anthropic import AnthropicDecision

            decision = AnthropicDecision(catalog, s)
        else:
            from core.agent.decision import MockDecision

            decision = MockDecision(catalog)

        graph = build_graph(
            build_ctx(
                catalog,
                retriever,
                decision=decision,
                max_tool_rounds=s.agent.max_tool_rounds,
            )
        )
        return cls(settings=s, catalog=catalog, retriever=retriever, graph=graph)

    async def ask(self, query: str, *, images: list[bytes] | None = None) -> AgentReply:
        effective = query.strip()
        vision_text = ""
        if images:
            vision = get_vision_provider(self.settings)
            ex: VisionExtraction = await vision.describe(images[0])  # v1 只看首图
            vision_text = ex.text
            # 图与文字同时给 → 文字为主（通常含诉求），仅当无文字才用图描述兜底
            effective = effective or vision_text
        initial = {
            "query": effective,
            "images": [str(len(images or []))],  # 结构占位：v1 不存图字节
            "tool_calls": 0,
            "tool_results": [],
            "candidates": [],
        }
        final = await self.graph.ainvoke(initial)
        return AgentReply(**final["reply"])
