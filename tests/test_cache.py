# -*- coding: utf-8 -*-
"""缓存层离线单测：NullCache / InMemoryCache / RedisCache 降级 / provider 工厂。"""
from __future__ import annotations

import pytest

from config.settings import CacheConfig, Settings
from core.store.cache import (
    InMemoryCache,
    NullCache,
    RedisCache,
    get_cache,
)


def test_null_cache_is_noop() -> None:
    c = NullCache()
    assert c.get("k") is None
    assert c.set("k", "v") is None  # 写不报错
    assert c.get("k") is None  # 读恒 miss


def test_inmemory_roundtrip_miss_and_overwrite() -> None:
    c = InMemoryCache()
    assert c.get("missing") is None
    c.set("a", "1")
    assert c.get("a") == "1"
    c.set("a", "2")
    assert c.get("a") == "2"


def test_inmemory_ttl_expiry() -> None:
    c = InMemoryCache(default_ttl_sec=300)
    c.set("x", "v", ttl_sec=0)  # ttl=0 → 立即过期
    assert c.get("x") is None
    c.set("y", "v")  # 走默认 300s → 未过期
    assert c.get("y") == "v"


def test_inmemory_uses_default_ttl_from_config() -> None:
    s = Settings.from_yaml().model_copy(
        update={"cache": CacheConfig(provider="memory", ttl_sec=7)}
    )
    c = get_cache(s)
    assert isinstance(c, InMemoryCache)
    assert c._default_ttl == 7


class _FlakyClient:
    """get/set 必抛 —— 模拟 Redis 掉线。"""

    def get(self, *a, **k):  # noqa: N802 - redis client 命名
        raise ConnectionError("redis down")

    def set(self, *a, **k):
        raise ConnectionError("redis down")


def test_redis_cache_degrades_to_miss_on_error() -> None:
    """Redis 掉线时读 miss、写 no-op、不抛 —— 缓存降级，主检索链路不受影响。"""
    c = RedisCache(_FlakyClient())
    assert c.get("k") is None
    c.set("k", "v")  # 不抛


def test_get_cache_off_and_unknown_provider() -> None:
    assert isinstance(get_cache(Settings.from_yaml()), NullCache)  # 默认 off
    bad = Settings.from_yaml().model_copy(
        update={"cache": CacheConfig(provider="etcd")}
    )
    with pytest.raises(ValueError, match="etcd"):
        get_cache(bad)


def test_get_cache_redis_without_dsn_fails_fast(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    s = Settings.from_yaml().model_copy(
        update={"cache": CacheConfig(provider="redis")}
    )
    with pytest.raises(ValueError, match="REDIS_URL"):
        get_cache(s)
