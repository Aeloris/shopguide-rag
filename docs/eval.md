# 评测：离线 mock 口径的回归门禁

## 为什么数字能信又"不能信"（诚实边界，README 同源）

- **能信**：全部指标是确定性代码计数（`evals/metrics.py`），同一份代码同一份语料逐次回放
  结果不变 → 适合做"防引擎回退"的回归护栏。测试与评测都离线、无 key、可复现。
- **不能信**：它**不是**"我的检索/理解能力有多强"的量表。mock 向量无语义、Agent 决定器是
  规则 → 召回靠词法、Agent 验证的是结构与防幻觉。语义天花板要用真 embedding + 真 LLM
  另开量表（阶段 B），本仓库绝不把 mock 数值迁移到真服务宣称。

## 数据源

- **检索 gold**（`evals/dataset.py`，12 条）：问法刻意带目标 SKU 的区分性词面
  （型号/参数/品牌），保证词法可达 → Recall/MRR 可回归。若某条 gold 无法召回，说明链路改了。
- **对抗不可答**（4 条）：交易 / 实时补贴行情 / 未来发布 / 具体型号真机性能 ——
  必须拒绝。另有 `no_match`（预算内无匹配）在流程测试覆盖。

## 双门禁（`uv run python -m evals.run`，报告 `data/eval/eval_report.md`）

| 指标 | 含义 | 实测 | 阈值 |
|---|---|---|---|
| 检索 Recall@5 | gold 目标进没进 top5（去重命中/gold 总数） | 1.000 | ≥ 0.9 |
| 检索 MRR@5 | 第一条相关目标的倒排名次 | 1.000 | ≥ 0.9 |
| grounded_rate | 可答问法中"作答且引用非空且全在库"占比 | 1.0000 | ≥ 1.0 |
| bad_refusal_rate | 可答问法被正确服务（未误拒）占比 | 1.0000 | ≥ 1.0 |
| 不可答拦截率 | 对抗集被拒绝占比 | 1.0 | = 1.0 |

任一指标低于阈值 → 进程退出码 1（CI/本地可作门禁）。阈值存在 `config.yaml eval.thresholds`，
是**实测回填**的防回退基线，不是能力宣称。

## 为什么要测"能拒绝"（bad_refusal / 拦截率）

面试时被问"LLM 幻觉怎么防"——除了回答白名单、引用溯源，还要证明**当该拒绝时系统会拒绝**。
对抗集 + 拦截率把这条也变成可回归的测试，而不是口头承诺。

## 运行

```bash
uv run python -m evals.run          # 全量：检索 + Agent，写报告 + 退出码
uv run pytest tests/test_retrieval_eval.py   # 指标算术 + gold 完整性
uv run pytest tests/test_agent_eval.py       # Agent 指标算术
```
