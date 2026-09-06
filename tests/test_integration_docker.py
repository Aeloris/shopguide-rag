# -*- coding: utf-8 -*-
"""阶段 B 真服务集成验证（PG JSONB 会话 + Redis 缓存 + Qdrant server 检索）。

跑法（先起服务、再设连接串与开关）：
  cd deploy && docker compose up -d          # postgres:16 / redis:7 / qdrant
  export SHOPGUIDE_DOCKER_INT=1
  export DATABASE_URL=postgresql://shopguide:shopguide_dev@localhost:5432/shopguide
  export REDIS_URL=redis://localhost:6379/0
  uv run pytest tests/test_integration_docker.py -q

未设 SHOPGUIDE_DOCKER_INT 或任一服务连不上 → 整文件 skip（不进默认离线计数）。
"""
from __future__ import annotations

import asyncio
import os
import time
import urllib.request

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from config.settings import REPO_ROOT, Settings
from core.agent import AgentRuntime
from core.store.cache import RedisCache
from core.store.session_store import PostgresSessionStore

DEV = REPO_ROOT / "config" / "config.dev.yaml"


def _services_alive() -> bool:
    s = Settings.from_yaml(DEV)
    try:
        urllib.request.urlopen(s.vector_db.url + "/healthz", timeout=3)
    except Exception:  # noqa: BLE001
        return False
    try:
        import psycopg

        psycopg.connect(
            os.environ.get(s.store.dsn_env, ""), connect_timeout=3, autocommit=True
        ).close()
    except Exception:  # noqa: BLE001
        return False
    try:
        from redis import Redis

        Redis.from_url(os.environ.get(s.cache.dsn_env, "")).ping()
    except Exception:  # noqa: BLE001
        return False
    return True


pytestmark = pytest.mark.skipif(
    os.getenv("SHOPGUIDE_DOCKER_INT") != "1" or not _services_alive(),
    reason="阶段 B 集成测试：需 SHOPGUIDE_DOCKER_INT=1 且 PG/Redis/Qdrant 可达",
)


def _dev() -> Settings:
    return Settings.from_yaml(DEV)


@pytest.fixture(scope="module")
def dev_runtime() -> AgentRuntime:
    """连 Qdrant server 的运行时（mock embedding，入库 13 SKU，确定性）。"""
    return asyncio.run(AgentRuntime.build(_dev()))


def test_postgres_session_roundtrip_and_reopen() -> None:
    s = _dev()
    dsn = os.environ[s.store.dsn_env]
    store = PostgresSessionStore(dsn)
    sid = store.create_session()
    assert store.exists(sid)

    store.append_message(sid, "user", {"content": "预算 3000 手机"})
    store.append_message(
        sid,
        "assistant",
        {"reply": {"answerable": True, "recommendations": [{"product_id": "redmi-k70"}]}},
    )
    msgs = store.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "预算 3000 手机"
    assert msgs[1]["reply"]["recommendations"][0]["product_id"] == "redmi-k70"

    # 新连接实例读同一库 → 跨进程/重启可查
    store2 = PostgresSessionStore(dsn)
    assert store2.get_messages(sid) == msgs

    assert not store.exists("nope0000000000")
    assert store.get_messages("nope0000000000") == []


def test_redis_cache_roundtrip_and_ttl() -> None:
    s = _dev()
    from redis import Redis

    cache = RedisCache(Redis.from_url(os.environ[s.cache.dsn_env], decode_responses=True))
    cache.set("k", "v1")
    assert cache.get("k") == "v1"
    cache.set("k", "v2")
    assert cache.get("k") == "v2"
    cache.set("exp", "x", ttl_sec=1)
    assert cache.get("exp") == "x"
    time.sleep(1.1)
    assert cache.get("exp") is None


def test_qdrant_server_runtime_search(dev_runtime: AgentRuntime) -> None:
    hits = asyncio.run(dev_runtime.retriever.search("适合办公的轻薄笔记本"))
    assert hits and hits[0].product_id == "macbook-air-m3"
    # dense 路真实打到 Qdrant server（集合 shopguide_products 应有 13 条）
    assert dev_runtime.retriever._store.count() == 13


def test_app_chat_persists_in_pg_and_search_cached() -> None:
    s = _dev()
    runtime = asyncio.run(AgentRuntime.build(s))  # 独立运行时（不影响 dev_runtime）
    app = create_app(runtime=runtime, settings=s)  # store=PG、cache=Redis 由 lifespan 构建
    with TestClient(app) as c:
        r1 = c.post("/api/chat", json={"message": "预算 3000 内的手机"})
        assert r1.status_code == 200 and r1.json()["reply"]["answerable"] is True
        sid = r1.json()["session_id"]

        r2 = c.post("/api/chat", json={"message": "那拍照更好的呢", "session_id": sid})
        assert r2.status_code == 200 and r2.json()["session_id"] == sid

        # 两轮对话落 PG（4 条：user/assistant × 2）
        msgs = app.state.session_store.get_messages(sid)
        assert len(msgs) == 4

        # /api/search 走 Redis：同 query 二次请求数据一致
        q = {"query": "适合办公的轻薄笔记本"}
        assert c.post("/api/search", json=q).json() == c.post("/api/search", json=q).json()
