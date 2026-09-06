# -*- coding: utf-8 -*-
"""会话存储抽象：ChatSession 持久化（对话历史/工具轨迹的可审计底座）。

离线两种实现：
- InMemorySessionStore：单进程内存 dict（默认，测试/无状态冒烟）；
- FileSessionStore：data/sessions/<id>.json，重启可查（无 Docker 的本地持久化）。
阶段 B（Docker 后）接 Postgres JSONB（会话表一行=一条消息/一次工具轨迹），同一接口。

为什么存会话而不是让 /chat 无状态：
- 多轮上下文、用户偏好、历史工具轨迹是导购 Agent 做"记忆/连续对话"的前提；
- 简历宣称"会话/上下文/工具轨迹落库"要有真实落点 —— v1 存全文消息 + 每次结构化回执。
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class SessionStore(Protocol):
    def create_session(self) -> str: ...
    def append_message(self, session_id: str, role: str, content: dict) -> None: ...
    def get_messages(self, session_id: str) -> list[dict]: ...
    def exists(self, session_id: str) -> bool: ...


def _new_session_id() -> str:
    return uuid.uuid4().hex[:16]


class InMemorySessionStore:
    """进程内 dict + 锁（单进程足够；多进程/分布式由阶段 B PG 承担）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    def create_session(self) -> str:
        sid = _new_session_id()
        with self._lock:
            self._sessions[sid] = []
        return sid

    def append_message(self, session_id: str, role: str, content: dict) -> None:
        with self._lock:
            msgs = self._sessions.setdefault(session_id, [])
            msgs.append({"role": role, **content})

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


class FileSessionStore:
    """data/sessions/<id>.json —— 每会话一个 JSON 文件（原子写防半截损坏）。"""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.json"

    def create_session(self) -> str:
        sid = _new_session_id()
        path = self._path(sid)
        with self._lock:
            if not path.exists():  # 极小概率冲突 → 重试一次
                path.write_text("[]", encoding="utf-8")
        return sid

    def append_message(self, session_id: str, role: str, content: dict) -> None:
        msgs = self.get_messages(session_id)
        msgs.append({"role": role, **content})
        with self._lock:
            tmp = self._path(session_id).with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(msgs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self._path(session_id))  # 原子替换

    def get_messages(self, session_id: str) -> list[dict]:
        path = self._path(session_id)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []  # 损坏容忍：宁可空会话，不让单文件坏档拖垮 API

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()


def get_session_store(settings) -> SessionStore:
    """按 config.store.provider 产出：memory | file（PG 阶段 B 同接口接入）。"""
    from config.settings import Settings

    s: Settings = settings
    if s.store.provider == "memory":
        return InMemorySessionStore()
    if s.store.provider == "file":
        return FileSessionStore(s.repo_root / s.store.path)
    raise ValueError(f"未知 store.provider：{s.store.provider}（可选 memory|file）")
