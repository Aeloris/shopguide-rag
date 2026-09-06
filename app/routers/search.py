# -*- coding: utf-8 -*-
"""POST /api/search —— 检索链路调试口（绕过 Agent，直接看混合检索 top-k）。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.deps import get_runtime
from app.schemas import SearchRequest
from core.retriever.schemas import RetrievedProduct

router = APIRouter()


@router.post("/search", response_model=list[RetrievedProduct])
async def search(payload: SearchRequest, request: Request):
    runtime = get_runtime(request)
    hits = await runtime.retriever.search(payload.query)
    top_k = payload.top_k or len(hits)
    return hits[:top_k]
