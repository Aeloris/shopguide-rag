# -*- coding: utf-8 -*-
"""API 集成（TestClient + 注入运行时会话共享实例）：health/products/search/chat。"""
from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from core.agent import AgentRuntime


@pytest.fixture(scope="module")
def client(runtime: AgentRuntime):
    app = create_app(runtime=runtime)  # 注入 runtime → lifespan 跳过重建（快）
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mode"] == "offline" and body["catalog_size"] == 13


def test_products_list_filters(client: TestClient) -> None:
    assert len(client.get("/api/products").json()) == 13
    phones = client.get("/api/products", params={"category": "phone"}).json()
    assert len(phones) == 6 and all(p["category"] == "phone" for p in phones)
    cheap = client.get("/api/products", params={"max_price": 3000}).json()
    assert all(p["price_cny"] <= 3000 for p in cheap)
    assert {p["id"] for p in cheap} == {"redmi-k70"}


def test_products_detail(client: TestClient) -> None:
    r = client.get("/api/products/xiaomi-14")
    assert r.status_code == 200
    d = r.json()
    assert d["price_cny"] == 3999 and "处理器" in d["specs"]
    assert client.get("/api/products/__nope__").status_code == 404


def test_search_endpoint(client: TestClient) -> None:
    r = client.post("/api/search", json={"query": "适合办公的轻薄笔记本"})
    assert r.status_code == 200
    hits = r.json()
    assert hits and hits[0]["product_id"] == "macbook-air-m3"
    assert hits[0]["snippet"] and hits[0]["category"] == "laptop"


def test_chat_budget_reply_grounded(client: TestClient) -> None:
    r = client.post("/api/chat", json={"message": "预算3000以内的安卓直屏手机"})
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"]
    reply = body["reply"]
    assert reply["answerable"] is True and reply["refused"] is False
    # 预算由代码判价 → 回执每个推荐都 ≤3000 且是手机
    assert "filter_products" in reply["tool_trace"]
    for rec in reply["recommendations"]:
        assert rec["price_cny"] <= 3000 and rec["category"] == "phone"


def test_chat_refusal_and_session_persist(client: TestClient) -> None:
    r = client.post("/api/chat", json={"message": "帮我下单买一台 iPhone 15"})
    assert r.status_code == 200
    reply = r.json()["reply"]
    assert reply["refused"] is True and reply["refusal_kind"] == "trade"

    # 同一会话续接 → 落库 4 条（两轮 × user/assistant）
    sid = r.json()["session_id"]
    r2 = client.post("/api/chat", json={"message": "那 3000 内有什么手机", "session_id": sid})
    assert r2.status_code == 200 and r2.json()["session_id"] == sid
    store = client.app.state.session_store
    msgs = store.get_messages(sid)
    assert len(msgs) == 4
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


def test_chat_unknown_session_404(client: TestClient) -> None:
    r = client.post("/api/chat", json={"message": "hi", "session_id": "deadbeef00000000"})
    assert r.status_code == 404


def test_chat_empty_payload_422(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": "  "}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_chat_image_only_uses_vision(client: TestClient) -> None:
    img = base64.b64encode(b"fake png bytes").decode()
    r = client.post("/api/chat", json={"images": [img]})
    assert r.status_code == 200
    reply = r.json()["reply"]
    assert reply["answerable"] is True and reply["tool_trace"][0] == "search_products"


def test_chat_bad_base64_400(client: TestClient) -> None:
    r = client.post("/api/chat", json={"message": "", "images": ["!!!not-base64!!!"]})
    assert r.status_code == 400
