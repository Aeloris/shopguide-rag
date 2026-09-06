# -*- coding: utf-8 -*-
"""FastAPI 应用工厂：离线即可起（uv run uvicorn app.main:app --port 8000）。

- lifespan 一次性构建 AgentRuntime + 会话存储（注入 runtime 可跳过重建 → 测试快）；
- /health 健康检查 + /api/{chat,search,products} 三组路由。
模式说明：默认 mode=offline（mock LLM/视觉/embedding + 内存向量库），无 key 无 Docker
可跑通全部端点；阶段 B 切 config.mode=dev 换真服务，业务代码零改动。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from app.routers import chat, products, search
from app.schemas import ProductListItem
from config.settings import Settings, get_settings
from core.agent import AgentRuntime
from core.store.session_store import SessionStore, get_session_store


def create_app(
    *,
    runtime: AgentRuntime | None = None,
    session_store: SessionStore | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    s = settings or get_settings()
    built_runtime = runtime
    built_store = session_store

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal built_runtime, built_store
        if built_runtime is None:  # 生产路径：启动时构建一次，供所有请求复用
            built_runtime = await AgentRuntime.build(s)
        if built_store is None:
            built_store = get_session_store(s)
        app.state.runtime = built_runtime
        app.state.session_store = built_store
        yield

    app = FastAPI(title="shopguide-rag", version="0.1.0", lifespan=lifespan)
    app.state.settings = s

    app.include_router(chat.router, prefix="/api", tags=["chat"])
    app.include_router(search.router, prefix="/api", tags=["search"])
    app.include_router(products.router, prefix="/api", tags=["products"])

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        runtime = getattr(app.state, "runtime", None)
        return {
            "status": "ok",
            "mode": s.mode,
            "catalog_size": len(runtime.catalog) if runtime else None,
        }

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "shopguide-rag",
            "docs": "/docs",
            "health": "/health",
            "api": "/api/chat  /api/search  /api/products",
        }

    return app


app = create_app()
