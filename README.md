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

**进度**：
- 阶段 A（离线底座）完成：`uv run pytest` **84 全绿**、`uv run python -m evals.run` 门禁全 PASS，
  无 Docker、无任何 API key 即可复现。
- 阶段 B（Docker 真服务**数据平面**）落地：Postgres 16 JSONB 会话 + Redis 缓存 + Qdrant server 检索，
  集成测试 4 条在真容器上全绿（见"真服务（Docker）"）。
- 阶段 B（**真 LLM 决定器**，live）：LangGraph 调度走真实 LLM —— Anthropic 兼容 Messages API，
  端点/模型全可配（本机演示端点=DeepSeek `deepseek-chat`）。**视觉与语义 embedding 未接**（当前端点
  无图片能力、缺 DashScope key）→ 保持 mock，接入点已就绪（诚实口径，取舍见
  [docs/architecture.md](docs/architecture.md)）。

---

## 快速开始（离线，~1 分钟）

```bash
uv sync                    # 安装依赖（含 langgraph）
uv run pytest              # 84 tests 全绿（离线确定性；集成测试无容器自动跳过）
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

## 真服务（Docker，阶段 B 数据平面已落地）

`deploy/docker-compose.yml` 起三个依赖容器，dev 配置下会话/缓存/向量库走真服务：

```bash
cd deploy && docker compose up -d   # postgres:16 + redis:7 + qdrant（首次拉镜像走 DaoCloud 加速）
docker compose ps                   # 等三个都 (healthy)
cd ..
cp .env.example .env                # 填 DATABASE_URL / REDIS_URL（默认值见 deploy/.env.docker）
export SHOPGUIDE_CONFIG=config/config.dev.yaml   # Windows PowerShell: $env:SHOPGUIDE_CONFIG="..."
uv run uvicorn app.main:app --port 8000          # 此即 dev 真服务
```

> **诚实边界**：dev 模式只把"数据平面"切真（Qdrant server 检索 / Postgres JSONB 会话 / Redis
> 缓存），LLM/视觉/embedding 仍是 mock —— 检索走词法召回、不宣称语义；接真 Claude / 真
> embedding 只需改 provider 并填 key，图与工具代码不动。

集成验证（需容器在跑；探测不可达会自动跳过，不进离线计数）：

```bash
SHOPGUIDE_DOCKER_INT=1 uv run pytest tests/test_integration_docker.py -q   # 4 passed
```

容器数据都在命名卷，落 Docker 数据根（本机已挪到 `D:\develop`，不占 C 盘）。
停止：`docker compose down`（加 `-v` 连数据卷一起删）。

### 真 LLM 决定器（live，可选）

数据平面之上再切**真实 LLM** 做工具调度（`config/config.live.yaml`；需 `.env` 里
`ANTHROPIC_API_KEY`，走中转再加 `ANTHROPIC_BASE_URL`）：

```bash
export SHOPGUIDE_CONFIG=config/config.live.yaml   # PowerShell: $env:SHOPGUIDE_CONFIG="..."
uv run uvicorn app.main:app --port 8000
```

> **边界**：真 LLM 只决定"下一步调哪个工具"——不可答拒绝仍是代码硬规则、预算/比参仍是代码做，
> grounding 由白名单构造保证；视觉/语义 embedding 未接时保持 mock（词法召回，不宣称语义）。
> 本机演示端点=DeepSeek（`api.deepseek.com/anthropic`，模型 `deepseek-chat`）；切官方 Claude 只需
> 去掉 `ANTHROPIC_BASE_URL` 并把 config 的 `llm.model` 改回 Claude 型号，视觉同理把
> `vision.provider` 切 anthropic。

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
config/       pydantic 分节强类型配置 + config.yaml（offline）/ config.dev.yaml（真服务）
core/store    会话存储：InMemory / File / Postgres JSONB；缓存：off / memory / Redis（同接口）
llm/          Provider 抽象：MockVision(离线) / anthropic 占位（需 key）
app/          FastAPI：/health + /api/{chat,search,products}（search 带热点缓存）
evals/        gold 检索集 + 对抗集 + 检索/Agent harness + run.py 门禁
fixtures/     catalog/products.json(13 SKU) + vision/sample.json
scripts/      make_catalog.py（确定性生成种子）/ smoke_core.py
deploy/       docker-compose.yml(postgres16/redis7/qdrant) + .env.docker
tests/        84 离线确定性测试 + 4 Docker 集成测试（无容器自动跳过）
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
- 不做大规模并发压测宣称；不做视觉真 OCR / 真 LLM / 真 embedding（provider 已留好，
  需真实 API key —— 未接前离线与 dev 都是 mock，不虚标语义）。

## License

MIT © 2026 qiaosheng
