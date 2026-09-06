# -*- coding: utf-8 -*-
"""GET /api/products —— 商品目录查询（调试/前端兜底；过滤走代码，不走检索）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from app.deps import get_runtime
from app.schemas import ProductDetail, ProductListItem

router = APIRouter()


def _brief(p) -> ProductListItem:
    return ProductListItem(
        id=p.id, brand=p.brand, model=p.model, name=p.name,
        category=p.category, price_cny=p.price_cny, summary=p.summary, tags=p.tags,
    )


@router.get("/products", response_model=list[ProductListItem])
async def list_products(
    request: Request,
    category: str | None = Query(default=None),
    brand: str | None = Query(default=None),
    max_price: int | None = Query(default=None, ge=0),
):
    runtime = get_runtime(request)
    out = []
    for p in runtime.catalog.products:
        if category and p.category != category:
            continue
        if brand and p.brand != brand:
            continue
        if max_price is not None and p.price_cny > max_price:
            continue
        out.append(_brief(p))
    return out


@router.get("/products/{product_id}", response_model=ProductDetail)
async def get_product(product_id: str, request: Request):
    runtime = get_runtime(request)
    p = runtime.catalog.get(product_id)
    if p is None:
        raise HTTPException(status_code=404, detail=f"商品不存在：{product_id}")
    b = _brief(p)
    return ProductDetail(
        **b.model_dump(), highlights=p.highlights, specs=p.specs
    )
