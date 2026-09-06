# -*- coding: utf-8 -*-
"""语义量表入口：隔离「Dense 语义路」做真/假 embedding 对照。

背景：分词是字+双字 n-gram、语料仅 13 SKU —— 在 top-5 下任何带共同词的改写问法
BM25 词法路都能召回，整库"混合召回"饱和（mock 混合=1.0）。所以**在 Recall 层面比
混合结果拉不开，也测不到 embedding 的价值**。embedding 换真向量真正影响的是
**Dense 路**：离线 mock 的 Dense 向量是确定性伪向量（无语义 ≈ 噪声），接 text-embedding-v3
后才是有语义的查询/文档向量。

本量表因此只测 Dense 路（store.search 顶真实向量集合），同题对照：

    real_dense = text-embedding-v3 查询向量 → qdrant_server 真向量集合检索
    mock_dense = MockEmbedding 查询向量  → Qdrant :memory: 伪向量集合检索

gold 是改写问法（无型号/参数字面，语义才可判），所以：
    · mock_dense（无语义）≈ 近随机，Recall@5/MRR@5 应低；
    · real_dense（语义）应把 gold 顶进 top-k。
最终线上混合（BM25+语义）召回也一并列出作上下文（语义路变真后混合不变差）。

诚实口径：
- 语料仅 13 SKU、gold 按人工语义判断标注 → 本表是"Dense 语义路在受控语料上
  从伪向量(≈随机)切到 text-embedding-v3 的实测增益"，非通用能力宣称。
- 真侧数字为网络实测（模型/端点版本会浮动）→ 作测量不作防回退门禁；阈值仅作
  "语义路确实工作"的下限自检。不参与离线 pytest / evals.run 门禁。
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from config.settings import Settings, get_settings
from core.catalog.loader import load_catalog
from core.embeddings.mock_embedding import MockEmbedding
from core.embeddings import get_embedding_provider
from core.ingest import Ingester
from core.retriever import Retriever
from core.vector_factory import build_store
from core.vector_store import ProductVectorStore
from evals.dataset_semantic import SEMANTIC_CASES, SEMANTIC_RATIONALE
from evals.metrics import mrr_at_k, recall_at_k

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_CONFIG = "config/config.live.yaml"
# Dense 语义路下限自检（非回归门禁；真向量接入后该路应显著高于伪向量基线）
REAL_DENSE_RECALL_REQ = 0.7
REAL_DENSE_MRR_REQ = 0.5


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def _fmt(x: float | None, nd: int = 3) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _clamp_txt(t: str, n: int = 24) -> str:
    return t if len(t) <= n else t[: n - 1] + "…"


async def _measure_dense(store, embedding, cases, *, top_k: int):
    """只走 Dense 路：查询向量 → store.search 顶真实向量集合 → 每例 top_k 的 product_id。"""
    rows = []
    for c in cases:
        q_vec = (await embedding.embed([c.question]))[0]
        hits = store.search(q_vec, top_k=top_k)
        hit_ids = [h["product_id"] for h in hits]
        rows.append(
            (c, hit_ids, recall_at_k(hit_ids, c.gold_product_ids, k=top_k),
             mrr_at_k(hit_ids, c.gold_product_ids, k=top_k))
        )
    return rows


async def _measure_hybrid(retriever, cases, *, top_k: int):
    rows = []
    for c in cases:
        hits = await retriever.search(c.question)
        hit_ids = [h.product_id for h in hits[:top_k]]
        rows.append(
            (c, hit_ids, recall_at_k(hit_ids, c.gold_product_ids, k=top_k),
             mrr_at_k(hit_ids, c.gold_product_ids, k=top_k))
        )
    return rows


def _avg(rows, metric: int) -> float:
    return sum(r[metric] for r in rows) / len(rows) if rows else 0.0


def _render(s, md, rd, mh, rh, ok: bool) -> str:
    k = s.eval.top_k
    lines = [
        "# shopguide-rag 语义量表（Dense 语义路：真向量 vs 伪向量，对照口径）",
        "",
        f"- top_k：{k}｜语料：13 SKU（3C）｜gold 单 SKU×{len(SEMANTIC_CASES)} 题",
        f"- **real_dense**：`text-embedding-v3` 查询向量 → qdrant_server 真向量集合",
        f"- **mock_dense**：`MockEmbedding`（确定性伪向量≈噪声，无语义）→ Qdrant :memory:",
        f"- 混合列=线上最终检索（BM25 + Dense → RRF，语义路已真）",
        f"- 问题集刻意**不带**型号/品牌/参数字面（改写问法，仅语义可判，见各题 rationale）。",
        "",
        "| # | 问题(截断) | gold | mock_dense 命中 | mock R@k/MRR | real_dense 命中 | real R@k/MRR |",
        "|---|-----------|------|-----------------|--------------|-----------------|--------------|",
    ]
    for i, ((cm, md_h, md_r, md_m), (_, rd_h, rd_r, rd_m)) in enumerate(
        zip(md, rd), start=1
    ):
        hit_mark = "✅" if any(g in rd_h for g in cm.gold_product_ids) else "❌"
        lines.append(
            f"| {i} | {_clamp_txt(cm.question)} | {cm.gold_product_ids[0]} | "
            f"{'、'.join(md_h) or '无'} | {_fmt(md_r)}/{_fmt(md_m)} | "
            f"{'、'.join(rd_h) or '无'} {hit_mark} | {_fmt(rd_r)}/{_fmt(rd_m)} |"
        )
    lines += [
        "",
        "**Dense 语义路汇总（本量表核心指标）：**",
        "",
        f"- mock_dense（伪向量）平均 Recall@{k} = {_fmt(_avg(md, 2))} / MRR@{k} = {_fmt(_avg(md, 3))}",
        f"- real_dense（text-embedding-v3）平均 Recall@{k} = {_fmt(_avg(rd, 2))} / MRR@{k} = {_fmt(_avg(rd, 3))}",
        f"- **语义路增益（real−mock）Recall：{_fmt(_avg(rd, 2) - _avg(md, 2))}｜"
        f"MRR：{_fmt(_avg(rd, 3) - _avg(md, 3))}**",
        "",
        "**线上最终混合召回（BM25+Dense 已真，作上下文）：**",
        "",
        f"- mock 混合（词法）平均 Recall@{k} = {_fmt(_avg(mh, 2))} / MRR@{k} = {_fmt(_avg(mh, 3))}",
        f"- real 混合（语义+词法）平均 Recall@{k} = {_fmt(_avg(rh, 2))} / MRR@{k} = {_fmt(_avg(rh, 3))}",
        "",
        "### 逐题语义映射（为什么该召回 gold）",
        "",
    ]
    for c in SEMANTIC_CASES:
        lines.append(f"- {c.question} → **{c.gold_product_ids[0]}**：{SEMANTIC_RATIONALE.get(c.question, '')}")
    lines += [
        "",
        "**诚实口径**：语料仅 13 SKU、gold 按人工语义判断标注 —— 本表是“Dense 语义路从伪向量",
        f"（≈随机）切到 text-embedding-v3 的实测增益”，非通用能力宣称。真侧为网络实测、随版本",
        f"浮动。下限自检：real_dense Recall@{k} ≥ {REAL_DENSE_RECALL_REQ} 且 MRR ≥ "
        f"{REAL_DENSE_MRR_REQ} → **{'PASS ✅' if ok else 'FAIL ❌'}**。",
        "",
    ]
    return "\n".join(lines) + "\n"


async def main() -> int:
    cfg_path = os.getenv("SHOPGUIDE_SEMANTIC_CONFIG") or str(REPO_ROOT / DEFAULT_LIVE_CONFIG)
    s = Settings.from_yaml(cfg_path)
    if s.embedding.provider != "dashscope" or s.vector_db.provider != "qdrant_server":
        print(
            "[semantic] ✗ 真侧需要 config.live.yaml：embedding.provider=dashscope + "
            "vector_db.provider=qdrant_server。当前不满足，不跑。"
        )
        return 2

    cases = SEMANTIC_CASES
    k = s.eval.top_k
    catalog = load_catalog(s)

    # 真侧：dashscope 真向量 → qdrant_server 集合（reset 全量重建）
    emb_real = get_embedding_provider(s)
    store_real = build_store(s, dimension=s.embedding.dimension)  # qdrant_server
    await Ingester(store_real, emb_real).rebuild(catalog)
    retriever_real = Retriever(store_real, emb_real, s)

    # 假侧（离线确定性）：MockEmbedding 伪向量 → Qdrant :memory:（本地内存集合，隔离）
    emb_mock = MockEmbedding(dimension=s.embedding.dimension)
    store_mock = ProductVectorStore(
        collection="semantic_mock_baseline", dimension=s.embedding.dimension, path=":memory:"
    )
    await Ingester(store_mock, emb_mock).rebuild(catalog)
    retriever_mock = Retriever(store_mock, emb_mock, s)

    # 先跑真侧；构建（联网/连库）失败 → 报清晰错误并退出 2
    try:
        rd = await _measure_dense(store_real, emb_real, cases, top_k=k)
        rh = await _measure_hybrid(retriever_real, cases, top_k=k)
    except Exception as e:  # noqa: BLE001 — 人类可读错误
        print("[semantic] ✗ 真侧失败（需 DASHSCOPE_API_KEY + qdrant server 在跑）：")
        print(f"  {type(e).__name__}: {e}")
        return 2

    md = await _measure_dense(store_mock, emb_mock, cases, top_k=k)
    mh = await _measure_hybrid(retriever_mock, cases, top_k=k)

    ok = _avg(rd, 2) >= REAL_DENSE_RECALL_REQ and _avg(rd, 3) >= REAL_DENSE_MRR_REQ
    md_txt = _render(s, md, rd, mh, rh, ok)
    out_dir = s.repo_root / s.eval.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_report.semantic.md"
    path.write_text(md_txt, encoding="utf-8")
    print(md_txt)
    print(f"[semantic] 报告已写：{path.relative_to(s.repo_root)}")
    print(
        f"[semantic] Dense 语义路 Recall@{k}={_avg(rd, 2):.3f}（mock={_avg(md, 2):.3f}），"
        f"MRR@{k}={_avg(rd, 3):.3f}（mock={_avg(md, 3):.3f}）"
    )
    print(f"[semantic] 语义路下限自检：{'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    _force_utf8_streams()
    raise SystemExit(asyncio.run(main()))
