# -*- coding: utf-8 -*-
"""商品知识库：种子数据完整性 / Catalog 查询语义 / Product 派生逻辑。"""
from __future__ import annotations

import pytest

from config.settings import Settings
from core.catalog.loader import Catalog, load_catalog
from core.catalog.schemas import Product

_CATEGORY_EXPECTED = {"phone": 6, "laptop": 4, "tablet": 3}


def test_seed_counts(catalog: Catalog) -> None:
    """种子库 13 SKU，三品类配额与数据设计一致。"""
    assert len(catalog) == 13
    counts = {c: 0 for c in _CATEGORY_EXPECTED}
    for p in catalog.products:
        counts[p.category] += 1
    assert counts == _CATEGORY_EXPECTED


def test_products_shape(catalog: Catalog) -> None:
    """每条 SKU 结构完整：价格整型正数、名称齐全、有规格、display_price 用 ¥。"""
    for p in catalog.products:
        assert isinstance(p.price_cny, int) and p.price_cny > 0
        assert p.id and p.brand and p.model and p.name and p.summary
        assert p.specs, f"{p.id} 缺规格参数"
        assert p.display_price == f"¥{p.price_cny}"


def test_catalog_duplicate_id_raises() -> None:
    def _mk(i: str, cat: str) -> Product:
        return Product(
            id=i, brand="B", model="M", name="N", category=cat,  # type: ignore[arg-type]
            price_cny=100, summary="s",
        )

    with pytest.raises(ValueError, match="重复"):
        Catalog([_mk("dup", "phone"), _mk("dup", "laptop")])


def test_catalog_empty_raises() -> None:
    with pytest.raises(ValueError):
        Catalog([])


def test_get_get_many_by_category(catalog: Catalog) -> None:
    assert catalog.get("xiaomi-14") is not None
    assert catalog.get("xiaomi-14").brand == "小米"  # type: ignore[union-attr]
    assert catalog.get("__nope__") is None
    # get_many 保序、未知 id 静默跳过
    many = catalog.get_many(["redmi-k70", "__nope__", "xiaomi-14"])
    assert [p.id for p in many] == ["redmi-k70", "xiaomi-14"]
    phones = catalog.by_category("phone")
    assert len(phones) == _CATEGORY_EXPECTED["phone"]
    assert all(p.category == "phone" for p in phones)


def test_missing_catalog_file_raises() -> None:
    s = Settings.from_yaml()
    s.catalog.file = "./fixtures/catalog/__no_such__.json"
    with pytest.raises(FileNotFoundError):
        load_catalog(s)


def test_matches_price_and_common_specs(catalog: Catalog) -> None:
    a = catalog.get("xiaomi-14")
    b = catalog.get("redmi-k70")
    assert a is not None and b is not None
    common = a.spec_keys_common(b)
    assert "处理器" in common and "屏幕" in common and "电池容量" in common
    # 预算筛选语义：price <= budget；budget=None 视为不限
    assert not a.matches_price(3000)   # 3999 超预算
    assert a.matches_price(4000)       # 刚好够
    assert a.matches_price(None)
    assert b.matches_price(2499)       # 边界：price == budget 通过
    assert not b.matches_price(2000)
