# -*- coding: utf-8 -*-
"""Agent 工具单测：预算筛选(代码判价) / 参数对比(共有键对齐) / 问句解析纯函数。"""
from __future__ import annotations

from core.agent.tools import GuideTools, _category_from_query, parse_budget_cny
from core.catalog.loader import Catalog


def _tools(catalog: Catalog) -> GuideTools:
    # 只测同步方法（filter/compare 不依赖 retriever），search 单独在 e2e 里验
    return GuideTools(catalog, None)  # type: ignore[arg-type]


def test_filter_by_budget_and_category(catalog: Catalog) -> None:
    res = _tools(catalog).filter_products(budget_cny=3000, category="phone", top_n=10)
    assert res["ok"]
    assert res["matched_count"] == 1  # 3000 内手机只有 redmi-k70(2499)
    assert res["products"][0]["product_id"] == "redmi-k70"
    assert res["products"][0]["price_cny"] <= 3000


def test_filter_unlimited_category_counts_all(catalog: Catalog) -> None:
    phones = _tools(catalog).filter_products(category="phone", top_n=50)
    assert phones["matched_count"] == len(catalog.by_category("phone")) == 6
    # 预算 None = 不限：matches_price 语义
    assert all(p["category"] == "phone" for p in phones["products"])


def test_filter_sorted_ascending(catalog: Catalog) -> None:
    laptops = _tools(catalog).filter_products(category="laptop", top_n=50)
    prices = [p["price_cny"] for p in laptops["products"]]
    assert prices == sorted(prices)


def test_compare_only_common_dims(catalog: Catalog) -> None:
    # 手机 x 平板：共有规格键只有 处理器/存储/重量 —— 缺列不硬补
    res = _tools(catalog).compare_products(["xiaomi-14", "ipad-air-5"])
    assert res["ok"]
    assert set(res["dims"]) == {"处理器", "存储", "重量"}
    assert len(res["rows"]) == 2
    for row in res["rows"]:
        assert set(row["dims"].keys()) == set(res["dims"])


def test_compare_requires_two_products(catalog: Catalog) -> None:
    res = _tools(catalog).compare_products(["xiaomi-14"])
    assert not res["ok"] and "至少需 2 个" in res["error"]


def test_parse_budget_and_category() -> None:
    assert parse_budget_cny("预算3000以内的手机") == 3000
    assert parse_budget_cny("不超过 5000 块") == 5000
    assert parse_budget_cny("随便看看轻薄本") is None
    assert _category_from_query("想买个办公笔记本") == "laptop"
    assert _category_from_query("安卓直屏手机") == "phone"
    assert _category_from_query("看剧用的平板") == "tablet"
    assert _category_from_query("什么值得买") is None
