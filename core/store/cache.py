# -*- coding: utf-8 -*-
"""检索结果缓存抽象（off | memory | redis），存的是已序列化的 JSON 字符串。

与 session store 的分工：会话存储是"必达、要落库"的（失败快速可见 → 500）；
缓存是"可有可无、只为省热点计算"的 —— 因此 Redis 掉线时静默降级为 miss（返回 None /
不写），检索照常走全量计算，不因缓存把主链路打挂。接口对齐让调用方无需分支：
provider=off 给 NullCache，读写全是空操作。
"""
from __future__ import annotations

import os
import threading
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, *, ttl_sec: int | None = None) -> None: ...


class NullCache:
    """provider=off：读写均为空操作（调用方不用 if provider != off 分支）。"""

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str, *, ttl_sec: int | None = None) -> None:
        return None


class InMemoryCache:
    """进程内 TTL 缓存（dict + 单调时钟过期；单进程够用，多进程由 Redis 承担）。"""

    def __init__(self, default_ttl_sec: int = 300) -> None:
        self._default_ttl = default_ttl_sec
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if now >= expires_at:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, *, ttl_sec: int | None = None) -> None:
        ttl = self._default_ttl if ttl_sec is None else ttl_sec
        with self._lock:
            self._data[key] = (time.monotonic() + ttl, value)


class RedisCache:
    """Redis 缓存（阶段 B：连 redis:7 容器）。

    掉线容忍：get/set 任何异常都当作 miss / 不写 —— 缓存降级，主检索链路不受影响。
    """

    def __init__(self, client, default_ttl_sec: int = 300) -> None:
        self._client = client
        self._default_ttl = default_ttl_sec

    def get(self, key: str) -> str | None:
        try:
            return self._client.get(key)  # miss → None
        except Exception:  # noqa: BLE001 - 缓存降级为 miss
            return None

    def set(self, key: str, value: str, *, ttl_sec: int | None = None) -> None:
        ttl = self._default_ttl if ttl_sec is None else ttl_sec
        try:
            self._client.set(key, value, ex=ttl)
        except Exception:  # noqa: BLE001 - 写失败不阻断主链路
            pass


def get_cache(settings=None) -> Cache:
    """按 settings.cache.provider 产出 off|memory|redis；redis 缺 DSN 时快速失败。"""
    from config.settings import Settings, get_settings

    s: Settings = settings if settings is not None else get_settings()
    ttl = s.cache.ttl_sec
    provider = s.cache.provider
    if provider == "off":
        return NullCache()
    if provider == "memory":
        return InMemoryCache(default_ttl_sec=ttl)
    if provider == "redis":
        dsn = os.getenv(s.cache.dsn_env)
        if not dsn:
            raise ValueError(
                f"cache.provider=redis 但环境变量 {s.cache.dsn_env} 未设置。\n"
                "解决办法：复制 .env.example 为 .env 填 REDIS_URL；"
                "或把 config 的 cache.provider 保持 off/memory。"
            )
        from redis import Redis

        return RedisCache(
            Redis.from_url(dsn, decode_responses=True), default_ttl_sec=ttl
        )
    raise ValueError(f"未知 cache.provider：{provider}（可选 off|memory|redis）")
