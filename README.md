# shopguide-rag — 基于多模态 RAG 的电商智能导购 Agent

> 一个**离线可跑、测试可复现、面试经得起拆**的多模态 RAG 导购 Agent。
> 混合检索（BM25 + Dense → RRF → 重排）召回商品，LangGraph 工具 Agent 负责
> 检索/预算筛选/参数对比，回答只引用检索命中（grounding 由白名单构造保证），
> 不可答问题按硬规则拒绝——绝不硬答、不凭空引入商品。

- 语言/工程：Python 3.12 · uv · pydantic v2 · FastAPI
- 检索：Qdrant（稠密向量） + 自研中文 BM25 → **RRF 融合** → 词法重排（真 CrossEncoder 阶段 B 可换）
- Agent：**LangGraph** 有界工具循环（3 工具，代码判价/对齐规格，模型不算算术）
- 多模态：视觉 Provider 抽象（mock 离线 / Claude 阶段 B 可切换）
- 评测：gold 检索集 + 对抗不可答集 → 门禁退出码（**诚实口径，见下**）

**当前阶段 A 全部完成**：`uv run pytest` 67 全绿、`uv run python -m evals.run` 门禁全 PASS，
无 Docker、无任何 API key 即可复现。阶段 B（Docker + 真视觉/真服务）见 [docs/architecture.md](docs/architecture.md)。

---

## 快速开始（离线，~1 分钟）

```bash
uv sync                    # 安装依赖（含 langgraph）
uv run pytest              # 67 tests 全绿（离线确定性）
uv run python -m evals.run # 评测门禁：PASS 退出码 0，报告在 data/eval/eval_report.md
uv run uvicorn app.main:app --port 8000   # 起 API（离线即可）
```

冒烟：

```bash
curl -s localhost:8000/health
# {"status":"ok","mode":"offline","catalog_size":13}

curl -s -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"预算3000以内的安卓直屏手机"}'
# reply.recommendations 全部 price<=3000 且 category=phone（代码判价）
# reply.tool_trace == ["search_products","filter_products"]

curl -s -X POST localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message":"帮我下单买一台 iPhone 15"}'
# reply.refused=true, refusal_kind="trade" —— 不做交易，宁可拒绝不硬答

curl -s -X POST localhost:8000/api/search -H 'content-type: application/json' \
  -d '{"query":"适合办公的轻薄笔记本"}'       # 混合检索调试口
curl -s 'localhost:8000/api/products?category=phone&max_price=3000'
```

OpenAPI 文档：`http://localhost:8000/docs`

## 评测结果（本评测集口径，非能力宣称）

离线 mock（MockEmbedding + Qdrant 内存 + 确定性规则 Agent）逐次回放：

| 门禁 | 实测 | 阈值(config.yaml) |
|---|---|---|
| 检索 Recall@5 | 1.000（12/12 gold） | ≥ 0.9 |
| 检索 MRR@5 | 1.000 | ≥ 0.9 |
| Agent grounded_rate（回答有据） | 1.0000 | ≥ 1.0 |
| Agent bad_refusal_rate（好例不误拒） | 1.0000 | ≥ 1.0 |
| 对抗不可答拦截率 | 1.0（4/4） | = 1.0 |

**诚实声明**：以上数字只对"13 个 3C SKU + 12 条词面可达 gold + 4 条对抗不可答"的
**离线语料**成立，是**防引擎回退的回归门禁**，不是"我的检索/理解能力有多强"。原因：
mock 向量无语义、决定器是确定性规则——召回主要靠词法、Agent 主要验证链路与防幻觉结构。
真正的语义上限要用真 embedding + 真 LLM 评测（阶段 B 后另开量表），本仓库不虚标。

## 目录

```
config/       pydantic 分节强类型配置 + config.yaml（provider/top_k/门禁阈值）
core/catalog  商品知识库：Product schema + 13 SKU 种子 + loader
core/ingest   Catalog → Markdown → MockEmbedding → Qdrant 幂等整包重建
core/retriever 自研中文分词 + BM25 + Dense → RRF(k=60) → 词法重排
core/agent    LangGraph 状态机：decide→run_tool↺ / finalize / refuse + 三工具
core/store    会话存储抽象：InMemory / File（PG/Redis 阶段 B 同接口）
llm/          Provider 抽象：MockVision(离线) / anthropic 阶段 B 占位
app/          FastAPI：/health + /api/{chat,search,products}
evals/        gold 检索集 + 对抗集 + 检索/Agent harness + run.py 门禁
fixtures/     catalog/products.json(13 SKU) + vision/sample.json
scripts/      make_catalog.py（确定性生成种子）/ smoke_core.py
tests/        67 个离线确定性测试
docs/         架构 / 检索 / Agent / 评测 四篇
```

## 为什么这些设计能防"面试暴雷"（三个诚实点）

1. **能算的绝不让模型算**：预算筛选用 `price <= budget` 的代码判定，参数对比只对齐两品
   共有规格键——模型/规则不做算术，比价结果可验证、可单测。
2. **回答只引用检索命中**：结构化回执里的每个 `product_id` 必须来自工具产出的
   candidates 白名单（grounding 由构造保证），不可答走独立 refuse 出口（交易/实时行情/
   未来/真机实测/库内无匹配五类硬规则）。
3. **离线 mock 是"真"的**：Agent 决定器默认是确定性规则（扮演听话 LLM），mock 向量无语义
   ——README 与代码注释都写明边界；接真 LLM/真 embedding 只换 provider，图与工具不动。

## 明确不做（v1 边界）

- 不做 ES/双引擎；不做真实价格爬虫与实时比价；不做下单/支付；
- 不做大规模并发压测宣称；不做视觉真 OCR（阶段 B 接 Claude 视觉）。

## License

MIT © 2026 qiaosheng
