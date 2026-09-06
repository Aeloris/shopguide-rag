# -*- coding: utf-8 -*-
"""会话存储：memory / file 两态 + 损坏容忍 + provider 工厂。"""
from __future__ import annotations

from config.settings import Settings
from core.store.session_store import FileSessionStore, InMemorySessionStore, get_session_store


def test_memory_store_roundtrip() -> None:
    store = InMemorySessionStore()
    sid = store.create_session()
    assert store.exists(sid)
    store.append_message(sid, "user", {"content": "你好"})
    store.append_message(sid, "assistant", {"reply": {"answerable": True}})
    msgs = store.get_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "你好" and msgs[1]["reply"]["answerable"] is True


def test_memory_unknown_session_empty() -> None:
    store = InMemorySessionStore()
    assert not store.exists("nope")
    assert store.get_messages("nope") == []


def test_file_store_persists_across_instances(tmp_path) -> None:
    a = FileSessionStore(tmp_path)
    sid = a.create_session()
    a.append_message(sid, "user", {"content": "预算3000手机"})
    # 新实例读同一目录 → 数据仍在（重启可查）
    b = FileSessionStore(tmp_path)
    assert b.exists(sid)
    assert b.get_messages(sid) == [{"role": "user", "content": "预算3000手机"}]


def test_file_store_tolerates_corruption(tmp_path) -> None:
    store = FileSessionStore(tmp_path)
    sid = store.create_session()
    # 人为写坏 JSON → get 返回空会话而非抛异常拖垮 API
    store._path(sid).write_text("{ not valid json", encoding="utf-8")
    assert store.get_messages(sid) == []


def test_factory_postgres_without_dsn_fails_fast(monkeypatch) -> None:
    """provider=postgres 但 DATABASE_URL 缺失 → 启动期就报清晰错误，不等到运行期。"""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    s = Settings.from_yaml()
    s.store.provider = "postgres"
    try:
        get_session_store(s)
    except ValueError as e:
        assert "DATABASE_URL" in str(e)
    else:  # pragma: no cover
        raise AssertionError("缺 DSN 应抛 ValueError")


def test_factory_unknown_provider_raises() -> None:
    s = Settings.from_yaml()
    s.store.provider = "hdf5"  # 从未支持的 provider → 报错防误配
    try:
        get_session_store(s)
    except ValueError as e:
        assert "hdf5" in str(e)
    else:  # pragma: no cover
        raise AssertionError("未知 provider 应抛 ValueError")
