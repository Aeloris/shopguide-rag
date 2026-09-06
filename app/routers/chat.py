# -*- coding: utf-8 -*-
"""POST /api/chat —— 导购对话主入口（文字 + 可选图片，返回结构化回执并落会话）。"""
from __future__ import annotations

import asyncio
import base64
import binascii
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.deps import get_runtime, get_session_store
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


def _decode_images(images: list[str]) -> list[bytes]:
    """base64 → bytes；容忍 data:image/...;base64, 前缀。只解码不校验内容（v1 mock 视觉）。"""
    out: list[bytes] = []
    for item in images:
        s = item.split(",", 1)[1] if item.startswith("data:") else item
        try:
            out.append(base64.b64decode(s))
        except (binascii.Error, ValueError) as e:  # pragma: no cover - 防御
            raise HTTPException(status_code=400, detail="images 含非法 base64") from e
    return out


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    runtime = get_runtime(request)
    store = get_session_store(request)

    if not payload.message.strip() and not payload.images:
        raise HTTPException(status_code=422, detail="message 与 images 至少提供一个")

    # SessionStore 是同步接口（PG 走 psycopg 阻塞驱动）→ 放线程池，别占事件循环
    sid = payload.session_id
    if sid is not None and not await asyncio.to_thread(store.exists, sid):
        raise HTTPException(status_code=404, detail=f"会话不存在：{sid}")
    sid = sid or await asyncio.to_thread(store.create_session)

    image_bytes = _decode_images(payload.images) if payload.images else None
    reply = await runtime.ask(payload.message, images=image_bytes)

    # 会话落库：用户输入 + 结构化回执（多轮/工具轨迹可审计）
    user_meta: dict[str, Any] = {"content": payload.message}
    if image_bytes:
        user_meta["images"] = len(image_bytes)
    await asyncio.to_thread(store.append_message, sid, "user", user_meta)
    await asyncio.to_thread(store.append_message, sid, "assistant", {"reply": reply.model_dump()})

    return ChatResponse(session_id=sid, reply=reply)
