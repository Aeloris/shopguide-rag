# -*- coding: utf-8 -*-
"""检索原语：中文分词(tokenize) / BM25 / RRF 融合 —— 纯函数确定性。"""
from __future__ import annotations

from core.retriever.bm25 import BM25Index
from core.retriever.fusion import rrf_fuse
from core.retriever.tokens import tokenize


def test_tokenize_ascii_model_kept_whole() -> None:
    """英数型号串整段保留（型号必须整段匹配，不能切碎）。"""
    toks = tokenize("支持 5G 与 Mate60Pro / X100")
    for whole in ("5g", "mate60pro", "x100"):
        assert whole in toks


def test_tokenize_chinese_yields_singles_and_bigrams() -> None:
    toks = tokenize("直屏手机")
    for t in ("直", "屏", "手", "机", "直屏", "屏手", "手机"):
        assert t in toks


def test_tokenize_deterministic_and_lowercased() -> None:
    assert tokenize("Mate 60 Pro") == tokenize("mate 60 pro")
    assert tokenize("X100 长焦") == tokenize("x100 长焦")


def test_tokenize_empty() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []


def test_bm25_ranks_repeated_term_doc_first() -> None:
    idx = BM25Index(
        ["红米 手机 手机 快充", "苹果 电脑 笔记本 办公", "手机上的一条无关句子"]
    )
    top = idx.search("手机", top_k=3)
    assert top and top[0][0] == 0  # tf 最高的 doc0 排第一
    assert all(score > 0 for _, score in top)


def test_bm25_deterministic() -> None:
    docs = ["小米 14 徕卡 影像", "Redmi K70 性价比", "vivo X100 蔡司 长焦"]
    a = BM25Index(docs).search("影像 徕卡", top_k=3)
    b = BM25Index(docs).search("影像 徕卡", top_k=3)
    assert a == b and a[0][0] == 0


def test_bm25_empty_query_returns_empty() -> None:
    idx = BM25Index(["随便 一段 文本"])
    assert idx.search("") == []
    assert idx.search("   ") == []


def test_rrf_fuse_common_high_rank_wins() -> None:
    # 1 两路都在前列 → 总分最高；3 第二路第 1 名压过 2（两路都靠后）
    assert rrf_fuse([1, 2, 3], [3, 1, 2]) == [1, 3, 2]


def test_rrf_fuse_member_in_one_list_only_kept() -> None:
    assert rrf_fuse([1, 2], []) == [1, 2]
    assert rrf_fuse([], [2, 1]) == [2, 1]


def test_rrf_fuse_empty_inputs() -> None:
    assert rrf_fuse([], []) == []
