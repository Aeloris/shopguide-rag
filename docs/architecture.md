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
| 视觉 | MockVision（fixture 固定返回） | — | 待带视觉端点/官方 Claude（AnthropicVision 已实现，`vision.provider=anthropic` 即接） |
| Embedding | MockEmbedding（确定性伪向量） | — | 待 DASHSCOPE_API_KEY（DashScopeEmbedding 已实现，`embedding.provider=dashscope` 即接） |

切换点都收敛在工厂：向量库 `vector_factory.build_store`（`vector_db.provider`）、会话
`get_session_store`、缓存 `get_cache`。dev 缺真 key 的部分维持 mock —— 检索是词法召回、
**不宣称语义**；真实语义评测待真 embedding 后另开量表（不把 mock 数字搬到真服务宣称）。

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
- [ ] 3b. 真视觉 Claude（AnthropicVision 就绪，待带视觉的兼容端点/官方 key）
- [ ] 3c. 真 embedding DashScope（DashScopeEmbedding 就绪，待 DASHSCOPE_API_KEY）
- [ ] 4. 语义评测另开量表（mock 数值不迁移到真服务宣称；真 embedding 接入后再开）
