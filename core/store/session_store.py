# -*- coding: utf-8 -*-
"""会话存储抽象：ChatSession 持久化（对话历史/工具轨迹的可审计底座）。

离线两种实现：
- InMemorySessionStore：单进程内存 dict（默认，测试/无状态冒烟）；
- FileSessionStore：data/sessions/<id>.json，重启可查（无 Docker 的本地持久化）。
阶段 B 已落地 Postgres JSONB（PostgresSessionStore：sessions 头表 + session_messages 明细，
一行一条消息/工具轨迹），与 File 同一接口 —— 业务代码零感知切换。

为什么存会话而不是让 /chat 无状态：
- 多轮上下文、用户偏好、历史工具轨迹是导购 Agent 做"记忆/连续对话"的前提；
- 简历宣称"会话/上下文/工具轨迹落库"要有真实落点 —— v1 存全文消息 + 每次结构化回执。
"""
from __future__ import annotations

import json
import os
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


class PostgresSessionStore:
    """Postgres JSONB 会话存储（阶段 B：连 postgres:16 容器，DSN 从环境变量读）。

    建模：sessions 头表（会话存在性）+ session_messages 明细表 —— JSONB 一行一条消息/
    工具轨迹，IDENTITY seq 保序 → create/get/exists 语义与 File 完全一致，同一接口。
    append 只插入不回读 → 无 read-modify-write 丢更新；FK 自愈（头行缺失自动补建）。

    连接与健壮性：
    - psycopg3 惰性连接：首次调用建连，并 CREATE TABLE IF NOT EXISTS 一次性建 schema；
    - 连接级错误（PG 重启/断连）自动重连重试一次，应用无需重启；
    - 失败快速可见：PG 不可达/约束错误直接抛错（HTTP 500），不做"静默丢会话"。
    """
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._conn = None
        self._schema_ready = False
        self._lock = threading.Lock()

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def _ready(self):
        """返回已建连且 schema 就绪的连接（连接被关则重建并重建 schema）。"""
        if self._conn is None or self._conn.closed:
            self._conn = self._connect()
            self._schema_ready = False
        if not self._schema_ready:
            with self._lock:
                if not self._schema_ready:
                    self._conn.execute(
                        "CREATE TABLE IF NOT EXISTS sessions ("
                        " session_id TEXT PRIMARY KEY,"
                        " created_at TIMESTAMPTZ NOT NULL DEFAULT now());"
                        "CREATE TABLE IF NOT EXISTS session_messages ("
                        " seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,"
                        " session_id TEXT NOT NULL REFERENCES sessions(session_id)"
                        "   ON DELETE CASCADE,"
                        " role TEXT NOT NULL,"
                        " content JSONB NOT NULL,"
                        " created_at TIMESTAMPTZ NOT NULL DEFAULT now());"
                        "CREATE INDEX IF NOT EXISTS idx_session_messages_sid"
                        " ON session_messages(session_id, seq);"
                    )
                    self._schema_ready = True
        return self._conn

    def _call(self, fn):
        """执行一次会话操作；遇连接级错误自动重建连接并重试一次。"""
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 按类型细分
            import psycopg

            if not isinstance(exc, psycopg.OperationalError):
                raise  # 语法/约束/业务类错误：直接暴露，利于排查
            with self._lock:
                self._conn = None
                self._schema_ready = False
            return fn()

    # ---- SessionStore 接口（与 InMemory/File 一致）----
    def create_session(self) -> str:
        def op():
            sid = _new_session_id()
            self._ready().execute(
                "INSERT INTO sessions(session_id) VALUES (%s)"
                " ON CONFLICT (session_id) DO NOTHING",
                (sid,),
            )
            return sid

        return self._call(op)

    def append_message(self, session_id: str, role: str, content: dict) -> None:
        def op():
            conn = self._ready()
            # 头行自愈：即使会话未经 create_session 也能落（兼容 File 的 auto-vivify 语义）
            conn.execute(
                "INSERT INTO sessions(session_id) VALUES (%s)"
                " ON CONFLICT (session_id) DO NOTHING",
                (session_id,),
            )
            conn.execute(
                "INSERT INTO session_messages(session_id, role, content)"
                " VALUES (%s, %s, %s::jsonb)",
                (
                    session_id,
                    role,
                    json.dumps({"role": role, **content}, ensure_ascii=False),
                ),
            )

        self._call(op)

    def get_messages(self, session_id: str) -> list[dict]:
        def op():
            conn = self._ready()
            rows = conn.execute(
                "SELECT content::text FROM session_messages"
                " WHERE session_id = %s ORDER BY seq",
                (session_id,),
            ).fetchall()
            return [json.loads(r[0]) for r in rows]

        return self._call(op)

    def exists(self, session_id: str) -> bool:
        def op():
            conn = self._ready()
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = %s", (session_id,)
            ).fetchone()
            return row is not None

        return self._call(op)


def get_session_store(settings) -> SessionStore:
    """按 config.store.provider 产出：memory | file | postgres（阶段 B JSONB）。"""
    from config.settings import Settings

    s: Settings = settings
    if s.store.provider == "memory":
        return InMemorySessionStore()
    if s.store.provider == "file":
        return FileSessionStore(s.repo_root / s.store.path)
    if s.store.provider == "postgres":
        dsn = os.getenv(s.store.dsn_env)
        if not dsn:
            raise ValueError(
                f"store.provider=postgres 但环境变量 {s.store.dsn_env} 未设置。\n"
                "解决办法：复制 .env.example 为 .env 填 DATABASE_URL；"
                "Postgres 默认账号见 deploy/.env.docker（docker compose 起来后）。"
            )
        return PostgresSessionStore(dsn)
    raise ValueError(
        f"未知 store.provider：{s.store.provider}（可选 memory|file|postgres）"
    )
