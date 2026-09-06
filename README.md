# shopguide-rag — 基于多模态 RAG 的电商智能导购 Agent

> 一个**离线可跑、测试可复现、面试经得起拆**的多模态 RAG 导购 Agent。
> 混合检索（BM25 + Dense → RRF → 重排）召回商品，LangGraph 工具 Agent 负责
> 检索/预算筛选/参数对比，回答只引用检索命中（grounding 由白名单构造保证），
> 不可答问题按硬规则拒绝——绝不硬答、不凭空引入商品。

- 语言/工程：Python 3.12 · uv · pydantic v2 · FastAPI
- 检索：Qdrant（稠密向量） + 自研中文 BM25 → **RRF 融合** → 词法重排（真 CrossEncoder 阶段 B 可换）
- Agent：**LangGraph** 有界工具循环（3 工具，代码判价/对齐规格，模型不算算术）
- 多模态：视觉 Provider 抽象（mock 离线 / live 真视觉 Qwen-VL，同支持 Anthropic 视觉可切）
- 评测：gold 检索集 + 对抗不可答集 → 门禁退出码（**诚实口径，见下**）

**进度**：
- 阶段 A（离线底座）完成：`uv run pytest` **101 全绿**、`uv run python -m evals.run` 门禁全 PASS，
  无 Docker、无任何 API key 即可复现。
- 阶段 B（Docker 真服务**数据平面**）落地：Postgres 16 JSONB 会话 + Redis 缓存 + Qdrant server 检索，
  集成测试 4 条在真容器上全绿（见"真服务（Docker）"）。
- 阶段 B（**真 LLM 决定器 + 真语义 embedding + 真视觉 Qwen-VL**，live）：LangGraph 调度走真实 LLM
  —— Anthropic 兼容 Messages API（端点/模型全可配，本机演示端点=DeepSeek `deepseek-chat`）；Dense
  语义路已切 DashScope `text-embedding-v3` 真向量（语义量表实测见下）；视觉已接同一把百炼 key 的
  **Qwen-VL**（OpenAI 兼容，连通性实测通过，见下）。诚实口径与取舍见
  [docs/architecture.md](docs/architecture.md)。

---

## 快速开始（离线，~1 分钟）

```bash
uv sync                    # 安装依赖（含 langgraph）
uv run pytest              # 101 tests 全绿（离线确定性；集成测试无容器自动跳过）
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

## Streamlit 演示（可视化前端，可选）

直接 import 引擎、不依赖 API/容器的演示界面，**离线 mock 即可全跑**：

```bash
uv sync                                  # 首次（拉取 streamlit 依赖）
uv run streamlit run scripts/demo_ui.py  # 打开 http://localhost:8501
```

三个 Tab：
- **💬 导购对话**：聊天框 / 侧边栏示例按钮 → 可答给商品卡（引用只来自检索命中白名单）、
  不可答给拒答 banner（交易/实时行情/未来/真机实测/库内无匹配五类）+ 工具轨迹；
  折叠区可传图走多模态路径（offline 空文本 → Vision mock 需求抽取）。
- **📦 商品库**：13 SKU 一览 + 单款规格/卖点（诚实口径：仅本离线语料）。
- **🔍 混合检索调试**：`POST /api/search` 的可视化（BM25+Dense→RRF→重排，top-5）。

UI 右下角如实显示当前 provider：默认 `mode=offline · llm/vision/embedding=mock`；设
`SHOPGUIDE_CONFIG=config/config.live.yaml`（且 `.env` 有真 key）即切 live 真链（真 LLM /
真语义 embedding / 真 Qwen-VL 视觉）。示例按钮只用词面可达的域内查询，不搬语义量表金句。

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
> grounding 由白名单构造保证。live 下 **语义 embedding 与视觉都已真**（都是同一把 DashScope key：
> `text-embedding-v3` 语义向量 + `Qwen-VL` 看图），仅 LLM 演示端点仍是 DeepSeek（本机无官方
> Claude key）。本机 LLM 端点=DeepSeek（`api.deepseek.com/anthropic`，模型 `deepseek-chat`）；
> 切官方 Claude 只需去掉 `ANTHROPIC_BASE_URL` 并把 config 的 `llm.model` 改回 Claude 型号。

### 真语义 embedding 与语义量表（live，3c 已落地）

Dense 语义路从"确定性伪向量"切到 **DashScope text-embedding-v3** 真向量（`embedding.provider=dashscope`
＋ `.env` 的 `DASHSCOPE_API_KEY`）；`build` 时 qdrant_server 集合全量重建为真向量。语义上限用独立
对照量表实测（`uv run python -m evals.run_semantic`，需容器 + key；**不并入离线门禁**），隔离只测
Dense 语义路（换 embedding 真正影响的那条路，13 SKU 改写问法、top-5）：

| Dense 语义路 | Recall@5 | MRR@5 |
|---|---|---|
| 离线 mock（确定性伪向量≈噪声） | 0.846 | 0.531 |
| **真 text-embedding-v3（live 实测）** | **1.000** | **0.949** |

对照报告（逐题命中 + "为什么该召回"rationale）：`data/eval/eval_report.semantic.md`。
**诚实口径**：语料仅 13 SKU、gold 按人工语义判断标注 —— 该增益是"Dense 路接真向量"的功能验证，
非通用语义能力宣称；线上混合（BM25+语义）召回不因换真向量变差。

### 真视觉（Qwen-VL，live，3b 已接）

`vision.provider=dashscope`（同一把 `DASHSCOPE_API_KEY`，OpenAI 兼容 `/chat/completions`）：
`llm/dashscope_vision.py` 把图片 base64 data-URI 发给 Qwen-VL（支持 png/jpeg/webp/avif/heic，
`image/avif` 实测被端点接受），抽一句"用户要找的商品需求"并入检索 query。连通性已实测
（鉴权/载荷/解析/AVIF）：对**纯色空图模型诚实拒识**（输出"无具体商品信息、无法识别数码商品
需求"，不幻觉商品）。

**真实照片 e2e（用户实拍商品图走完整链路，正/负两例都实测）**：
- **域外负例**：柠檬青柠洗洁精的商品宣传图（AVIF，3C 域外）→ Qwen-VL 诚实抽出"洗洁精"需求
  → 域外门禁判**店外无匹配** → Agent `no_match` 拒答（`answerable=False`），不再凑数推荐 3C
  （该图暴露并修复了"检索无相关度地板"缺陷，见下"域外门禁"）。
- **域内正例**：一张紫色 iPhone 手机商品图（AVIF）→ Qwen-VL 抽出"紫色iPhone手机，追求时尚外观
  与高性能体验" → 检索召回 **top-1 = `iphone-15`**（目录在售的唯一 iPhone），其余候选同品类无跨
  品类幻觉；域外门禁不误伤域内图。**诚实备注**：该图背面无型号字样，Qwen-VL 判"15/15 Plus、
  无法完全确定"，仓库不把 iphone-15 说成"图上这台必是 15"的强断言；若该机型实为目录外的代际
  （如 iPhone 17），属下"库外数字代际门禁"场景、以用户文字点名判定（图上无型号字样时不虚构）。

### 域外门禁（no_match 拒答落地，live/offline 同逻辑）

真商品图 e2e 曝光：`search_products` 原来**无相关度地板** —— 域外 query（洗洁精/抽纸…）也会
凑数返回 top-5 个 3C，`candidates` 永不为空，声明里的 `no_match` 拒答实际不可达。修复（
`core/retriever/service.py` + `retrieval.dense_match_floor`）：问句与库内商品**零词法重叠**
（len≥2 token：CJK 双字/整段型号）**且** 顶配向量分低于门禁阈值 → 判"店外无匹配"，检索返空 →
Agent 走 `no_match` 拒答。**诚实边界**：机械键盘/游戏耳机/显示器等"外设配件"与库内笔记本共享
规格词（键盘/散热/屏）→ 词法判定视其语域内，不会被本门禁拦截 —— 真店靠"库存品类"判定，
属下一增量，此处不虚标。回归见 `tests/test_retriever_ood_gate.py`（洗洁精/洗发水/抽纸→拒答，
域内改写问法不误伤，离线门禁 Recall@5 不回退）。

### 库外数字代际门禁（文字点名判定，live/offline 同逻辑）

真机 iPhone 17 图 e2e 曝光第二类"冒充命中"：用户图/文字点名 **iPhone 17**，而目录 iPhone 只有
15 —— 修前会拿同品牌相近代 `iphone-15` 冒充命中返回。修复（`core/agent/decision.py`
`find_out_of_stock_phone`，Mock 与 Anthropic 决定器都**先于模型**走代码规则）：问句点名了
"在售手机系列之外的数字代际"（iPhone 17 / 小米 15 / Mate 70 / Redmi K80 / vivo X200 /
OnePlus 13，任一越库即拒）→ `no_match` 拒答，理由点明该型号与在售代际（在库代际从目录
自校准，不硬编码具体数字）。**诚实边界**：只覆盖 6 个"数字代际清晰"的在售手机系列；
笔记本/平板（M3 / X1 Carbon / Pro16 / Air5 / MatePad 13.2）无稳定数字代际规则，未纳入
（仍按最近在售）。回归见 `tests/test_agent_out_of_stock.py`。

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
真正的语义上限在 live 用真 embedding 另开的**语义量表**实测（见上，`data/eval/eval_report.semantic.md`），
本仓库不把 mock 数字搬到语义宣称，不虚标。

## 目录

```
config/       pydantic 分节强类型配置 + config.yaml（provider/top_k/门禁阈值）
core/catalog  商品知识库：Product schema + 13 SKU 种子 + loader
core/ingest   Catalog → Markdown → MockEmbedding → Qdrant 幂等整包重建
core/retriever 自研中文分词 + BM25 + Dense → RRF(k=60) → 词法重排
core/agent    LangGraph 状态机：decide→run_tool↺ / finalize / refuse + 三工具
config/       pydantic 分节强类型配置 + config.yaml（offline）/ config.dev.yaml（真服务）
core/store    会话存储：InMemory / File / Postgres JSONB；缓存：off / memory / Redis（同接口）
llm/          Provider 抽象：视觉 mock / anthropic / dashscope(Qwen-VL) + LLM 决定器 + 视觉
app/          FastAPI：/health + /api/{chat,search,products}（search 带热点缓存）
evals/        gold 检索集 + 对抗集 + 检索/Agent harness + run.py 门禁 + 语义量表(run_semantic)
fixtures/     catalog/products.json(13 SKU) + vision/sample.json
scripts/      make_catalog.py（确定性生成种子）/ smoke_core.py / demo_ui.py（Streamlit 演示前端）
deploy/       docker-compose.yml(postgres16/redis7/qdrant) + .env.docker
tests/        101 离线确定性测试 + 4 Docker 集成测试（无容器自动跳过）
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
- 不做大规模并发压测宣称；不做视觉真 OCR（provider 已留好，真视觉待带图端点/官方 key）。
  真 LLM 决定器与真语义 embedding 已在 live 接入（.env 供 key）；离线与 dev 保持 mock，不虚标。

## License

MIT © 2026 qiaosheng
