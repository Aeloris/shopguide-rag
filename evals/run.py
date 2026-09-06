# -*- coding: utf-8 -*-
"""全量评测入口：检索门禁 + Agent 门禁 → Markdown 报告 + 退出码。

用法：uv run python -m evals.run
- 报告写到 config.eval.output_dir/eval_report.md（默认 ./data/eval/eval_report.md）。
- 离线 mock 确定性；config.yaml eval.thresholds 是实测回填的防回退基线（README 诚实口径）。

评测块：
  1) 检索：12 条 gold Q&A → Recall@K / MRR@K（召回是否把"该召回的"保住）
  2) Agent：同批可答问法 → grounded_rate / bad_refusal_rate；对抗不可答集 → 拦截率
任一指标低于阈值 → 退出码 1（防引擎回退）。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from config.settings import Settings, get_settings
from core.agent import AgentRuntime
from evals.agent_eval import AgentEvalReport, evaluate_agent
from evals.dataset import RETRIEVAL_CASES, UNANSWERABLE_CASES
from evals.retrieval_eval import RetrievalReport, evaluate_retriever

UNANSWERABLE_REFUSAL_REQ = 1.0  # 对抗不可答必须全拦（代码硬规则，恒 1.0）


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


async def _eval_all(s: Settings) -> tuple[RetrievalReport, AgentEvalReport]:
    rt = await AgentRuntime.build(s)  # 一次构建，检索/agent 共用
    t = s.eval.thresholds
    retrieval = await evaluate_retriever(
        rt.retriever,
        RETRIEVAL_CASES,
        top_k=s.eval.top_k,
        recall_threshold=t.hybrid_recall_at_k,
        mrr_threshold=t.hybrid_mrr_at_k,
    )
    agent = await evaluate_agent(
        rt,
        answerable_queries=[c.question for c in RETRIEVAL_CASES],
        unanswerable_cases=UNANSWERABLE_CASES,
    )
    return retrieval, agent


def _render_markdown(retrieval: RetrievalReport, agent: AgentEvalReport, s: Settings) -> str:
    t = s.eval.thresholds
    lines = [
        "# shopguide-rag 评测报告（离线 mock 口径）",
        "",
        f"- 引擎模式：`{s.mode}`（MockEmbedding + Qdrant 内存 + 确定性规则 Agent，可回放）",
        f"- 检索 top_k：{s.eval.top_k}",
        "",
        "## ① 检索门禁（词面可达 gold，防召回回退）",
        "",
        f"- 样本：{retrieval.n_cases} 条 gold｜达标 {retrieval.n_passed}",
        f"- **Recall@{s.eval.top_k} = {_fmt(retrieval.avg_recall_at_k)}**（阈值 ≥ {t.hybrid_recall_at_k}）",
        f"- **MRR@{s.eval.top_k} = {_fmt(retrieval.avg_mrr_at_k)}**（阈值 ≥ {t.hybrid_mrr_at_k}）",
        "",
        "| 问题 | gold | 命中 | Recall | MRR | 达标 |",
        "|------|------|------|--------|-----|------|",
    ]
    for c in retrieval.cases:
        lines.append(
            f"| {c.question} | {'、'.join(c.gold_product_ids)} | "
            f"{'、'.join(c.hit_product_ids) or '无命中'} | {c.recall_at_k:.2f} | "
            f"{c.mrr_at_k:.2f} | {'✅' if c.passed else '❌'} |"
        )
    lines += [
        "",
        "## ② Agent 门禁（回答有据 / 好例不误拒 / 不可答全拦）",
        "",
        f"- 预期可答：{agent.n_answerable}｜对抗不可答：{agent.n_unanswerable}",
        f"- **grounded_rate = {_fmt(agent.grounded_rate, 4)}**（有据：引用非空且全在库｜阈值 ≥ {t.grounded_rate_min}）",
        f"- **bad_refusal_rate = {_fmt(agent.bad_refusal_rate, 4)}**（好例被正确服务，无误拒｜阈值 ≥ {t.bad_refusal_rate_min}）",
        f"- **unanswerable_refusal_rate = {_fmt(agent.unanswerable_refusal_rate, 4)}**（对抗不可答被拦｜要求 = {UNANSWERABLE_REFUSAL_REQ}）",
        "",
        "### 对抗不可答明细",
        "",
        "| 问题 | 判定 | 原因 |",
        "|------|------|------|",
    ]
    for c in agent.cases:
        if c.expected_answerable:
            continue
        served_mark = "❌ 未拒绝" if c.served else "✅ 拒绝"
        lines.append(f"| {c.query} | {served_mark} | {c.reply.refusal_reason or c.reply.refusal_kind} |")
    return "\n".join(lines) + "\n"


def _gate_ok(retrieval: RetrievalReport, agent: AgentEvalReport, s: Settings) -> bool:
    t = s.eval.thresholds
    checks = [
        (retrieval.avg_recall_at_k, t.hybrid_recall_at_k, "检索 Recall"),
        (retrieval.avg_mrr_at_k, t.hybrid_mrr_at_k, "检索 MRR"),
        (agent.grounded_rate, t.grounded_rate_min, "Agent grounded_rate"),
        (agent.bad_refusal_rate, t.bad_refusal_rate_min, "Agent bad_refusal_rate"),
        (agent.unanswerable_refusal_rate, UNANSWERABLE_REFUSAL_REQ, "不可答拦截率"),
    ]
    ok = True
    for value, req, name in checks:
        if value is None or value < req:
            ok = False
            print(f"[gate] ✗ {name}: {value} < {req}")
    return ok


async def main() -> int:
    s = get_settings()
    retrieval, agent = await _eval_all(s)
    md = _render_markdown(retrieval, agent, s)
    out_dir = s.repo_root / s.eval.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "eval_report.md"
    path.write_text(md, encoding="utf-8")
    print(md)
    ok = _gate_ok(retrieval, agent, s)
    print(f"[eval] 报告已写：{path.relative_to(s.repo_root)}")
    print(f"[eval] 结论：{'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    _force_utf8_streams()
    raise SystemExit(asyncio.run(main()))
