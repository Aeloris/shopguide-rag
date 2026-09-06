# shopguide-rag

**Multimodal RAG shopping-guide agent — hybrid retrieval + tool-calling agent, fully runnable offline.**

基于多模态 RAG 的电商智能导购 Agent：对一份固定 3C 商品目录（13 SKU：手机 / 笔记本 / 平板）做结构化导购。
混合检索（BM25 + Dense → RRF 融合 → 词法重排）召回商品，LangGraph 有界工具 Agent 负责检索 / 预算筛选 / 参数对比；
回答**只引用检索命中的商品**（grounding 由候选白名单构造保证），不可答问题按硬规则拒绝 —— 绝不硬答、不凭空引入商品。

无需任何 API key、无需 Docker，离线即可复现全部测试与评测门禁。

<picture>
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white" alt="Python 3.11 | 3.12">
  <img src="https://img.shields.io/badge/tests-101%20passed-2ea44f" alt="tests: 101 passed">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="license: MIT">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/LangGraph-agent-4B32C3" alt="LangGraph">
</picture>

---

## 核心特性

- **混合检索**：自研中文 BM25（字 + 双字 n-gram）+ Dense 向量（Qdrant）→ **RRF(k=60) 融合** → 词法重排，检索阶段 B 可换真 CrossEncoder。
- **LangGraph 有界工具循环**：`search_products` / `filter_products` / `compare_products` 三个工具，≤ 上限轮次；
  **能算的绝不让模型算** —— 预算筛选用 `price <= budget` 的代码判定、参数对比只对齐两品共有规格键，结果可验证、可单测。
- **grounding 由构造保证**：结构化回执里每个 `product_id` 必须来自工具产出的候选白名单，回答层绝不凭空引入商品。
- **五类硬规则拒答**：交易 / 实时行情 / 未来发布 / 真机实测 / 库内无匹配 —— 词面规则先于检索与模型，宁可拒绝不硬答。
- **两道 no_match 门禁**（真图 e2e 暴露并修复的真实缺陷，live / offline 同逻辑）：
  - **域外门禁**：问句与库内商品零词法重叠且顶配向量分低于门禁阈值 → 判"店外无匹配"，检索返空 → `no_match` 拒答。
  - **库外数字代际门禁**：点名了"在售手机系列之外的数字代际"（iPhone 17 / 小米 15 / …）→ 调工具前直接 `no_match`，
    不拿同品牌相近代冒充命中（在库代际从目录自校准）。
- **多模态视觉**：视觉 Provider 抽象 —— 离线 mock / live 真视觉 **Qwen-VL**（同一把 DashScope key），Anthropic 视觉可切。
- **Streamlit 演示前端**：三 Tab 可视化演示，直接 import 引擎、离线可跑。
- **可复现评测门禁**：gold 检索集 + 对抗不可答集，门禁失败退出码非 0。

## 快速开始（离线，约 1 分钟）

环境：Python 3.11/3.12 + [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                     # 安装依赖（含 langgraph / streamlit）
uv run pytest               # 101 tests 全绿（集成测试无容器自动跳过）
uv run python -m evals.run  # 评测门禁 PASS（退出码 0），报告在 data/eval/eval_report.md
uv run uvicorn app.main:app --port 8000   # 起 API
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
# reply.refused=true, refusal_kind="trade" —— 不做交易
```

## Streamlit 演示 UI

```bash
uv run streamlit run scripts/demo_ui.py   # 打开 http://localhost:8501
```

- **💬 导购对话**：聊天框 + 侧边栏示例按钮；可答给商品卡（引用只来自检索命中白名单）、不可答给拒答 banner，附工具轨迹；可传图走多模态路径。
- **📦 商品库**：13 SKU 一览 + 单款规格 / 卖点。
- **🔍 混合检索调试**：`retriever.search` 可视化（BM25 + Dense → RRF → 重排，top-5）。

UI 右下角如实显示当前 provider（默认 `offline · mock`），切换 live 自动生效，不虚标运行形态。

## API

| Endpoint | 说明 |
|---|---|
| `GET /health` | 状态 / 运行模式 / 目录大小 |
| `POST /api/chat` | 一次导购对话。请求 `{message, session_id?, images?(base64)}` → 结构化回执 `AgentReply`：`answerable / refused / refusal_kind / refusal_reason / recommendations / comparison / filter_note / tool_trace` |
| `POST /api/search` | 混合检索调试口，返回 top-k 命中（含 rrf_score / snippet） |
| `GET /api/products` | 商品列表，支持 `category` / `max_price` 过滤 |
| `GET /docs` | OpenAPI 文档 |

## 运行模式

| 模式 | 配置 | 向量 | 会话 | 缓存 | LLM / Embedding / 视觉 |
|---|---|---|---|---|---|
| **offline**（默认） | `config/config.yaml` | Qdrant `:memory:` | InMemory | off | mock（确定性，无 key） |
| **dev**（Docker 数据平面） | `config/config.dev.yaml` | Qdrant server | Postgres 16 JSONB | Redis | mock |
| **live**（真链，可选） | `config/config.live.yaml` | Qdrant server | Postgres JSONB | Redis | anthropic / dashscope（真 key） |

- 用环境变量 `SHOPGUIDE_CONFIG` 选配置（如 `config/config.live.yaml`），缺省走 offline。
- Provider 是插拔的：LLM 走 Anthropic Messages API（端点 / 模型全可配，兼容 OpenAI 兼容的 anthropic 端点）；语义向量与视觉走 DashScope（`text-embedding-v3` / `Qwen-VL`）。真 key 只放 `.env`（已 gitignore），缺 key 时对应 provider fail-fast。
- **Docker 真服务**：`deploy/docker-compose.yml` 起 postgres:16 + redis:7 + qdrant；`SHOPGUIDE_DOCKER_INT=1 uv run pytest tests/test_integration_docker.py` 在真容器上跑 4 条集成测试（容器不可达自动跳过，不进离线计数）。

## 架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  入口层  FastAPI (/chat /search /products) · Streamlit demo_ui       │
├──────────────────────────────────────────────────────────────────────┤
│  决定层  AgentDecision（Mock 规则 或 真 LLM 决定器）                    │
│          ① 不可答硬规则(五类) → ② 库外数字代际门禁 → ③ 工具规划          │
├──────────────────────────────────────────────────────────────────────┤
│  工具层  search_products ── filter_products(代码判价) ── compare(对齐) │
│          └ 候选白名单 → 回执 grounding 校验                            │
├──────────────────────────────────────────────────────────────────────┤
│  检索层  BM25(词法) ─┐  RRF(k=60) 融合 → 词法重排 → top-5               │
│          Dense(Qdrant) ┘  （含域外门禁：零词法重叠∧低向量分 → 返空）      │
├──────────────────────────────────────────────────────────────────────┤
│  数据层  fixtures 13 SKU → ingest → Qdrant  :memory: / server          │
│  存储层  SessionStore(InMemory/File/Postgres) · Cache(off/memory/Redis)│
└──────────────────────────────────────────────────────────────────────┘
```

```
config/       pydantic 分节强类型配置 + config.yaml / config.dev.yaml / config.live.yaml
core/catalog  商品知识库：Product schema + 13 SKU 种子 + loader
core/ingest   Catalog → Markdown → Embedding → Qdrant 幂等整包重建
core/retriever 自研中文分词 + BM25 + Dense → RRF(k=60) → 词法重排 + 域外门禁
core/agent    LangGraph 状态机：decide → run_tool ↺ / finalize / refuse + 三工具
core/store    会话存储 InMemory / File / Postgres JSONB；缓存 off / memory / Redis（同接口）
llm/          Provider 抽象：Mock / Anthropic / DashScope(Qwen-VL) + 视觉
app/          FastAPI：/health + /api/{chat,search,products}
evals/        gold 检索集 + 对抗集 + harness + run.py 门禁 + 语义量表(run_semantic)
fixtures/     catalog/products.json(13 SKU) + vision/sample.json
scripts/      make_catalog.py / smoke_core.py / demo_ui.py（Streamlit 演示）
deploy/       docker-compose.yml(postgres16/redis7/qdrant) + .env.docker
tests/        101 离线确定性测试 + 4 Docker 集成测试（无容器自动跳过）
docs/         架构 / 检索 / Agent / 评测 四篇
```

## 评测

离线（mock embedding + Qdrant 内存 + 确定性规则 Agent）逐次回放，作为**防引擎回退的回归门禁**：

| 门禁 | 实测 | 阈值 |
|---|---|---|
| 检索 Recall@5 | 1.000（12/12 gold） | ≥ 0.9 |
| 检索 MRR@5 | 1.000 | ≥ 0.9 |
| Agent grounded_rate（回答有据） | 1.0000 | ≥ 1.0 |
| Agent bad_refusal_rate（好例不误拒） | 1.0000 | ≥ 1.0 |
| 对抗不可答拦截率 | 1.0（4/4） | = 1.0 |

Dense 语义路的上限用独立**语义量表**（live，真 embedding）实测 —— 隔离只测"换 embedding 真正影响"的那条路
（13 SKU 改写问法、top-5；需容器 + DashScope key，不并入离线门禁）：

| Dense 语义路 | Recall@5 | MRR@5 |
|---|---|---|
| 离线 mock（确定性伪向量） | 0.846 | 0.531 |
| 真 `text-embedding-v3` | **1.000** | **0.949** |

- 门禁：`uv run pytest`（101）+ `uv run python -m evals.run`（PASS 退出码 0）。
- 语义量表：`uv run python -m evals.run_semantic`（live），对照报告 `data/eval/eval_report.semantic.md`。

## 真视觉 e2e（live，Qwen-VL）

`vision.provider=dashscope` 时，图片以 base64 data-URI 发给 Qwen-VL（支持 png/jpeg/webp/**avif**/heic），抽一句"用户要找的商品需求"并入检索 query。用真实商品照片走完整链路已实测：

- **域外负例**：一张柠檬洗洁精商品宣传图（AVIF，3C 域外）→ 抽出"洗洁精"需求 → 域外门禁判店外无匹配 → Agent `no_match` 拒答，不再凑数推荐 3C。
- **域内正例**：一张紫色 iPhone 商品图（AVIF，背面无型号字样）→ 抽出"紫色 iPhone 手机"需求 → 召回 top-1 = `iphone-15`（目录在售的唯一 iPhone）。**不把 iphone-15 说成"图上这台必是 15"**：Qwen-VL 对该图只能判"15/15 Plus、无法完全确定"，型号级匹配以文字点名为准（图为目录外代际如 iPhone 17 时走库外数字代际门禁）。
- 对纯色空图模型诚实拒识（"无具体商品信息"，不幻觉商品）。

## 局限（Limitations）

- 目录是**固定 13 SKU 离线语料**（6 手机 + 4 笔记本 + 3 平板），非真实电商库存；测试 / 评测数字只对该语料成立。
- 不做真实价格爬虫 / 实时比价 / 下单支付 / 并发压测宣称；不做视觉真 OCR。
- offline 模式 LLM / 视觉 / embedding 均为 mock：决定器是确定性规则、伪向量无语义（词法召回），README 与代码注释均如实标注。
- **域外门禁以词法重叠为主信号**：机械键盘 / 游戏耳机 / 显示器等与库内笔记本共享规格词（键盘 / 散热 / 屏）的外设不被拦截 —— 真实零售场景以"库存品类"判定，属边界（回归见 `tests/test_retriever_ood_gate.py`）。
- **库外数字代际门禁只覆盖 6 个数字代际清晰的在售手机系列**；笔记本 / 平板命名（M3 / X1 Carbon / Pro16 / Air5 / MatePad）无稳定代际规则，不在门禁内（回归见 `tests/test_agent_out_of_stock.py`）。

## 文档

- [architecture.md](docs/architecture.md) — 架构与取舍 · [retrieval.md](docs/retrieval.md) — 检索链路
- [agent.md](docs/agent.md) — 工具 Agent 与拒答 · [eval.md](docs/eval.md) — 评测与门禁

## License

[MIT](LICENSE) © 2026 qiaosheng
