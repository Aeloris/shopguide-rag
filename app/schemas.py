# -*- coding: utf-8 -*-
"""API 层数据契约（请求/响应的进出边界；引擎层模型原样透传 AgentReply/检索命中）。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.agent import AgentReply
from core.catalog.schemas import ProductCategory


class ChatRequest(BaseModel):
    """一次导购对话请求：文字与图至少给一个（v1 图文同传以文字为主）。"""

    message: str = Field(default="", description="用户文字（可为空，仅传图时由视觉兜底）")
    session_id: str | None = Field(default=None, description="续接会话；缺省则新开会话")
    images: list[str] | None = Field(
        default=None, description="图片 base64 列表（v1 mock 不解析内容，仅多模态占位）"
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: AgentReply


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="检索 query（走混合检索链路）")
    top_k: int | None = Field(default=None, ge=1, le=20)


class ProductListItem(BaseModel):
    id: str
    brand: str
    model: str
    name: str
    category: ProductCategory
    price_cny: int
    summary: str
    tags: list[str]


class ProductDetail(ProductListItem):
    highlights: list[str]
    specs: dict[str, str]
