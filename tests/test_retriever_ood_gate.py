# -*- coding: utf-8 -*-
"""域外门禁回归（真商品图 e2e 曝光缺陷的修复验证）。

背景：用户提供真实商品照片（柠檬青柠洗洁精，3C 域外）走 Qwen-VL → 检索时发现
`search_products` 无相关度地板 —— 域外 query 也会凑数返回 top-5 个 3C，Agent 照单推荐。
修复：问句与语料**零词法重叠** 且 顶配向量分低于 `retrieval.dense_match_floor` → 判店外无匹配。

本文件锁三类行为：
1. 店外品类（洗洁精/洗发水/抽纸…）→ 检索零候选（否则 finalize 凑数推 3C）；
2. 域内改写问法（带品类/规格词面重叠）→ 照常召回，不误伤；
3. Agent 层：洗洁精类 query → no_match 拒答；域内 query 照常 answerable。
"""
from __future__ import annotations

import asyncio

from core.retriever import Retriever
from core.retriever.service import has_catalog_overlap, is_out_of_catalog


def _search(retriever: Retriever, query: str):
    return asyncio.run(retriever.search(query))


# 用户真实照片曝光的店外品类（Qwen-VL 对那张图抽出"柠檬洗洁精除油清新"）+ 典型日用品
_OOD_QUERIES = ["柠檬洗洁精除油清新", "洗洁精", "洗发水", "抽纸", "酸奶", "洗衣液"]

# 域内改写问法（语义量表同款口径：无型号字面，但带品类/规格词）——门禁不可误伤
_IN_DOMAIN_QUERIES = [
    "适合办公的轻薄笔记本",
    "512g 拍人像好的手机",
    "预算3000以内的安卓直屏手机",
    "打游戏帧率高又轻便的本",
]


def test_foreign_goods_queries_now_return_no_match(built_retriever: Retriever) -> None:
    """店外品类 → 零候选；否则 finalize 会把凑数的 3C 当推荐（本缺陷）。"""
    for q in _OOD_QUERIES:
        assert _search(built_retriever, q) == [], f"域外问句应无匹配: {q!r}"


def test_in_domain_paraphrase_queries_untouched(built_retriever: Retriever) -> None:
    """域内改写问法（品类/规格词面可达）必须照常召回，不误伤。"""
    for q in _IN_DOMAIN_QUERIES:
        assert len(_search(built_retriever, q)) > 0, f"域内问句不应被误拒: {q!r}"


def test_official_gold_queries_still_recall(built_retriever: Retriever) -> None:
    """离线 gold 全词面可达 → 门禁不影响 Recall（门禁只在零词法重叠时才可能触发）。"""
    for q in ["预算3000以内的安卓直屏手机", "适合办公的轻薄笔记本"]:
        assert len(_search(built_retriever, q)) > 0


def test_has_catalog_overlap_ignores_single_chars() -> None:
    """重叠只认 len>=2 的 token（CJK 双字/整段型号）：单字如"机/买"太泛，不能当语域证据。"""
    vocab = _catalog_vocab_of(["联想小新Pro16 轻薄本 2.5K 高刷屏 大电池 办公"])
    assert has_catalog_overlap("轻薄本办公", vocab) is True
    assert has_catalog_overlap("柠檬洗洁精", vocab) is False


def test_is_out_of_catalog_semantics() -> None:
    """纯函数语义：词法零重叠(主信号) ∧ 顶配向量分低(真向量兜底) → 店外。"""
    docs = ["联想小新Pro16 轻薄本 2.5K 高刷屏 大电池 办公"]
    # 零词法重叠 + 顶配分低 → 店外
    assert is_out_of_catalog("柠檬洗洁精", docs, top_dense_score=0.30, dense_floor=0.45) is True
    # 有词法重叠 → 永不判店外（离线 mock 向量无语义，就靠这条保证不误伤）
    assert is_out_of_catalog("轻薄本办公", docs, top_dense_score=0.0, dense_floor=0.45) is False
    # 零词法重叠但向量分高（真 embedding 语义兜底：实属店内的极迂回问法）→ 不误拒
    assert is_out_of_catalog("柠檬洗洁精", docs, top_dense_score=0.6, dense_floor=0.45) is False
    # 空库不参与（上层空结果自会处理，不误标）
    assert is_out_of_catalog("柠檬洗洁精", [], 0.30, 0.45) is False


def _catalog_vocab_of(docs: list[str]) -> set[str]:
    from core.retriever.service import _catalog_vocab

    return _catalog_vocab(docs)


# ---- Agent 层（离线确定性）：搜空 → candidates 空 → finalize no_match ----

def test_agent_refuses_foreign_goods_query(runtime) -> None:
    """图→文本侧的洗洁精 query 到 Agent：应 no_match 拒答，绝不推 3C。"""
    reply = asyncio.run(runtime.ask("柠檬洗洁精除油清新"))
    assert reply.refused is True
    assert reply.refusal_kind == "no_match"
    assert reply.recommendations == []


def test_agent_still_answers_in_domain_query(runtime) -> None:
    """同一条域内问法 Agent 仍正常作答（门禁不过度）。"""
    reply = asyncio.run(runtime.ask("适合办公的轻薄笔记本"))
    assert reply.answerable is True
    assert reply.refused is False
    assert len(reply.recommendations) > 0
