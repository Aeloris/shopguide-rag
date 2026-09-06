# -*- coding: utf-8 -*-
"""FastAPI 依赖注入：从 request.app.state 取共享运行时（lifespan 里一次性构建）。"""
from __future__ import annotations

from fastapi import Request

from core.agent import AgentRuntime
from core.store.cache import Cache
from core.store.session_store import SessionStore


def get_runtime(request: Request) -> AgentRuntime:
    return request.app.state.runtime


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_cache(request: Request) -> Cache:
    return request.app.state.cache
