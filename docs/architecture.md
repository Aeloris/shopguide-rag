# 架构设计

## 分层与数据流

```
用户 ──文字/图片──▶ FastAPI (/api/chat)
                     │  (图片先走 VisionProvider → 文本兜底)
                     ▼
              LangGraph Agent（core/agent）
   decide ──tool──▶ run_tool（检索/预算筛选/参数对比）──▶ decide（有界 ≤3 轮）
      │ final                    │ refuse
      ▼                          ▼
  finalize（白名单收束）       refuse（拒绝出口）
      └──────── 结构化 AgentReply（渲染层转文本）────────┘
                     │
                     ├─▶ SessionStore 落库（多轮/工具轨迹可审计）
                     ▼
              Qdrant（商品向量） ◀─ Ingest（13 SKU Markdown，整包重建）

 混合检索：BM25(自研中文分词) + Dense(Mock/Qdrant) → RRF(k=60) → 词法重排 → top5
```

## 三阶段演进（先离线可跑 → Docker 真服务数据平面 → 真 LLM/embedding）

硬约束决定了顺序：装 Docker 前要能跑 → **真服务可切换的代码先落**；Docker 就绪后先切
**数据平面**（PG/Redis/Qdrant server，无 key 也能验）；最后等有真实 API key 再接 provider。
同一套工厂，业务代码零改动：

| 能力 | 离线默认（config.yaml） | 真服务数据平面（config.dev.yaml，✅） | 真 LLM/视觉/embedding（config.live.yaml） |
|---|---|---|---|
| 向量库 | Qdrant `:memory:` / 本地 path | Qdrant server（✅ 集成测试过） | — |
| 会话存储 | InMemory / File | Postgres 16 JSONB（✅） | — |
| 缓存 | off（NullCache） | Redis 7（✅） | — |
| LLM 决定器 | MockDecision（确定性规则） | — | ✅ 真决定器：Anthropic 兼容 Messages（端点/模型可配；演示端点=DeepSeek `deepseek-chat`） |
| 视觉 | MockVision（fixture 固定返回） | — | ✅ 真视觉 Qwen-VL（`provider=dashscope`，OpenAI 兼容，`llm/dashscope_vision.py`；连通实测通过，真商品图 e2e 待补） |
| Embedding | MockEmbedding（确定性伪向量） | — | ✅ 真语义向量 DashScope `text-embedding-v3`（Dense 路，qdrant_server 真向量集合；含单请求≤10 分批） |

切换点都收敛在工厂：向量库 `vector_factory.build_store`（`vector_db.provider`）、会话
`get_session_store`、缓存 `get_cache`、视觉 `get_vision_provider`。**config.dev.yaml 仍全 mock
（词法召回、不宣称语义）**；live 把 LLM 决定器 / Embedding / 视觉（Qwen-VL）切真后，语义上限由
**独立语义量表**（evals/run_semantic.py，真/伪向量对照）实测，不把 mock 数字搬到真服务宣称、
也不并入离线门禁。

## 关键取舍（面试可展开）

1. **单 Qdrant，不上 ES**：语料量级与语义召回靠向量足够；上双引擎是过度设计，
   面试被问"为什么不用 ES"可以答：候选集合小、真实场景可平滑扩展，工程上先做对再做重。
2. **一份 Product = 一份 Markdown 文档入库**：结构化 spec 仍在 pydantic 层（供比价/筛选），
   入库文本带标题组织——检索命中即拿 SKU 溯源，引用能说清出处。
3. **LangGraph 只做编排，不是"巨型 LLM"**：图里多数节点是确定性代码；"哪步调什么工具"
   由决定器输出（离线=规则，真 LLM=同接口换实现）；预算算术/规格对齐绝不给模型算。
4. **能算的绝不让模型算**：全项目统一的诚实原则，预算、Recall/MRR、grounding 都是代码计数。

## 运行方式

- 离线（无 Docker）：`uv run pytest` · `uv run python -m evals.run` ·
  `uv run uvicorn app.main:app --port 8000`（mode=offline，默认）。测试/评测与业务共用同一批
  provider 工厂，注入 `path=":memory:"` 保证隔离、不污染 `./data`。
- 真服务数据平面（Docker）：`cd deploy && docker compose up -d` → `.env` 填 DATABASE_URL/
  REDIS_URL → `SHOPGUIDE_CONFIG=config/config.dev.yaml uv run uvicorn app.main:app --port 8000`。
- 集成测试（需容器在跑）：`SHOPGUIDE_DOCKER_INT=1 uv run pytest tests/test_integration_docker.py`
  —— 探测 PG/Redis/Qdrant 不可达即 skip，不进默认离线计数。
- 语义量表（live，需容器 + DASHSCOPE_API_KEY）：
  `SHOPGUIDE_SEMANTIC_CONFIG=config/config.live.yaml uv run python -m evals.run_semantic`
  —— Dense 语义路真/伪向量对照，报告 data/eval/eval_report.semantic.md，不进离线计数。
- live 全服务：`SHOPGUIDE_CONFIG=config/config.live.yaml uv run uvicorn app.main:app --port 8000`

## 阶段 B 进度（Roadmap）

- [x] 1. `deploy/docker-compose.yml`：postgres:16 + redis:7 + qdrant（含健康检查；镜像经
      DaoCloud 加速，已实测 `up -d` 三服务 healthy）
- [x] 2. Postgres JSONB SessionStore + Redis Cache + 向量切 qdrant server（`config.dev.yaml`
      / `.env.docker`）；集成测试 4 条在真服务上全绿：会话跨连接回读、Redis TTL 过期、
      qdrant server 检索召回、app 双轮对话落 PG + `/api/search` 走 Redis
- [x] 3a. 真 LLM 决定器（`config.live.yaml`，`llm.provider=anthropic`）：LangGraph 调度走真实
      LLM —— Anthropic 兼容 Messages API（httpx 直连，端点/模型全可配）。实测 `deepseek-chat`
      正确选 compare/filter/final；预算对话 e2e 召回 redmi-k70(≤3000) 且真调 filter 工具；
      交易问题仍在问 LLM 前被代码硬规则拦截（refuse=trade，0 工具轮）
- [x] 3b. 真视觉已接 Qwen-VL：`llm/dashscope_vision.py`（OpenAI 兼容 /chat/completions，图片
      base64 data-URI；支持 png/jpeg/webp/**avif**/heic）+ vision 工厂 `dashscope` 分支 +
      `config.live.yaml`（`qwen-vl-max`）。连通性实测通过（鉴权/载荷/解析/AVIF）；对纯色空图
      模型诚实拒识（"无具体商品信息"，不幻觉）。**真实照片 e2e 已录**：用户实拍柠檬青柠洗洁精
      商品宣传图（AVIF，3C 域外）→ Qwen-VL 诚实抽出"洗洁精" → 域外门禁判店外无匹配 →
      Agent `no_match` 拒答（不再凑数推 3C）。该图曝光并修复了"检索无相关度地板"缺陷（见 3d）。
      域内真实商品图的正例 e2e 仍待一张手机/笔记本照片后补录（AnthropicVision 同就绪，切官方
      Claude 只需 provider 改回 anthropic）
- [x] 3c. 真 embedding DashScope：`embedding.provider=dashscope` 切 text-embedding-v3，
      qdrant_server 集合全量重建为真向量；真联调暴露并修复"单请求≤10 条分批"边界（补回归测试）
- [x] 4. 语义量表已开（evals/run_semantic.py，隔离 Dense 语义路、真/伪向量对照）：
      text-embedding-v3 Recall@5=1.000/MRR@5=0.949 vs mock 伪向量 0.846/0.531（13 SKU top-5）
      —— 报告 data/eval/eval_report.semantic.md；mock 数值不迁移到真服务宣称
- [x] 5. **域外门禁**（真商品图 e2e 曝光缺陷的修复）：检索原来无相关度地板，域外 query 也会
      凑数返回 top-5 个 3C（`candidates` 永不为空 → `no_match` 拒答不可达）。修复：问句与库内
      商品零词法重叠（len≥2 token）且 顶配向量分 < `retrieval.dense_match_floor`(0.45) →
      判店外无匹配 → 检索返空 → Agent `no_match` 拒答。词法零重叠是主信号（对封闭目录稳定、
      embedding 无关）；向量分是真 embedding 时的语义兜底（防"零词面但语义强、实属店内"误拒）。
      离线门禁 Recall@5=1.000 不回退；回归 `tests/test_retriever_ood_gate.py`。
      **诚实边界**：机械键盘/游戏耳机/显示器等"外设配件"与库内笔记本共享规格词（键盘/散热/屏）
      → 不被本门禁拦截 —— 真店靠"库存品类"判定，属下一增量。
