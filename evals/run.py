# -*- coding: utf-8 -*-
"""检索评测入口：离线构建管线 → 跑 gold 集 → 写 Markdown 报告 + 门禁退出码。

用法：uv run python -m evals.run
- 报告写到 config.eval.output_dir/eval_report.md（默认 ./data/eval/eval_report.md）。
- 门禁（config.yaml eval.thresholds）：Recall@K / MRR@K 均值低于阈值 → 退出码 1。
  阈值是"离线 mock 实测回填的防回退基线"（见 README 诚实口径），不是能力宣称。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config.settings import Settings, get_settings
from core.catalog.loader import load_catalog
from core.embeddings import get_embedding_provider
from core.ingest import Ingester
from core.retriever import Retriever
from core.vector_store import ProductVectorStore
from evals.dataset import RETRIEVAL_CASES
from evals.retrieval_eval import RetrievalReport, evaluate_retriever


def _force_utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


async def _run_eval(s: Settings) -> RetrievalReport:
    catalog = load_catalog(s)
    store = ProductVectorStore(
        collection="eval", dimension=s.embedding.dimension, path=":memory:"
    )
    emb = get_embedding_provider(s)  # mock（config.mode=offline 默认）
    await Ingester(store, emb).rebuild(catalog)
    retriever = Retriever(store, emb, s)
    t = s.eval.thresholds
    return await evaluate_retriever(
        retriever,
        RETRIEVAL_CASES,
        top_k=s.eval.top_k,
        recall_threshold=t.hybrid_recall_at_k,
        mrr_threshold=t.hybrid_mrr_at_k,
    )


def _render_markdown(report: RetrievalReport, s: Settings) -> str:
    rows = []
    for i, c in enumerate(report.cases, start=1):
        mark = "✅" if c.passed else "❌"
        gold = "、".join(c.gold_product_ids)
        hits = "、".join(c.hit_product_ids) if c.hit_product_ids else "（无命中）"
        rows.append(
            f"| {i} | {c.question} | {gold} | {hits} | {c.recall_at_k:.2f} | "
            f"{c.mrr_at_k:.2f} | {mark} |"
        )
    t = s.eval.thresholds
    ra = report.avg_recall_at_k
    rm = report.avg_mrr_at_k
    recall_line = (
        f"{ra:.3f}" if ra is not None else "-"
    )
    mrr_line = f"{rm:.3f}" if rm is not None else "-"
    return "\n".join(
        [
            "# 检索评测报告（离线 mock 口径）",
            "",
            f"- 引擎模式：`{s.mode}`（MockEmbedding + Qdrant 内存，确定性）",
            f"- 样本：{report.n_cases} 条 gold（字段级 Recall/MRR 逐例见下表）",
            f"- top_k：{s.eval.top_k}",
            "",
            "## 汇总",
            "",
            f"- **Recall@{s.eval.top_k} = {recall_line}**（门禁 ≥ {t.hybrid_recall_at_k}）",
            f"- **MRR@{s.eval.top_k} = {mrr_line}**（门禁 ≥ {t.hybrid_mrr_at_k}）",
            f"- 达标样例：{report.n_passed}/{report.n_cases}",
            "",
            "## 门禁结论",
            "",
            f"**{'PASS ✅' if report.gate_ok else 'FAIL ❌'}**",
            "",
            "## 逐例明细",
            "",
            "| # | 问题 | gold | top-k 命中 | Recall | MRR | 达标 |",
            "|---|------|------|-----------|--------|-----|------|",
            *rows,
            "",
        ]
    )


async def main() -> int:
    s = get_settings()
    report = await _run_eval(s)
    out_dir = s.repo_root / s.eval.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_report.md"
    path.write_text(_render_markdown(report, s), encoding="utf-8")
    print(_render_markdown(report, s))
    print(f"[eval] 报告已写：{path.relative_to(s.repo_root)}")
    print(
        f"[eval] 结论：{'PASS' if report.gate_ok else 'FAIL'} "
        f"（Recall@5={report.avg_recall_at_k} MRR@5={report.avg_mrr_at_k}）"
    )
    return 0 if report.gate_ok else 1


if __name__ == "__main__":
    _force_utf8_streams()
    raise SystemExit(asyncio.run(main()))
