# -*- coding: utf-8 -*-
"""POST /api/search —— 检索链路调试口（绕过 Agent，直接看混合检索 top-k）。

热点缓存：query+top_k 做 key（sha1）存 JSON 字符串。provider=off 时 NullCache 读写
空操作 → 无缓存也不多一次分支；provider=redis 时掉线自动降级 miss，主链路不受影响。
"""
from __future__ import annotations

import hashlib
import json

from fastapi import APIRouter, Request

from app.deps import get_cache, get_runtime
from app.schemas import SearchRequest
from core.retriever.schemas import RetrievedProduct

router = APIRouter()


@router.post("/search", response_model=list[RetrievedProduct])
async def search(payload: SearchRequest, request: Request):
    runtime = get_runtime(request)
    cache = get_cache(request)
    query = payload.query
    requested = payload.top_k

    key = "search:" + hashlib.sha1(f"{query}|{requested}".encode("utf-8")).hexdigest()
    hit = cache.get(key)
    if hit is not None:
        return [RetrievedProduct(**item) for item in json.loads(hit)]

    hits = await runtime.retriever.search(query)
    limit = requested or len(hits)
    top = hits[:limit]
    cache.set(key, json.dumps([h.model_dump() for h in top], ensure_ascii=False))
    return top
