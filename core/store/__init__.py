# -*- coding: utf-8 -*-
"""会话/状态存储与缓存抽象。

- session store：memory | file | postgres(JSONB) —— 同一 SessionStore 接口；
- cache：off | memory | redis —— 同一 Cache 接口，off 用 NullCache（读写空操作）。
离线默认都是本地实现；dev 切真服务时业务代码零改动。
"""
from core.store.cache import (
    Cache,
    InMemoryCache,
    NullCache,
    RedisCache,
    get_cache,
)
from core.store.session_store import (
    FileSessionStore,
    InMemorySessionStore,
    PostgresSessionStore,
    SessionStore,
    get_session_store,
)

__all__ = [
    "Cache",
    "FileSessionStore",
    "InMemoryCache",
    "InMemorySessionStore",
    "NullCache",
    "PostgresSessionStore",
    "RedisCache",
    "SessionStore",
    "get_cache",
    "get_session_store",
]
