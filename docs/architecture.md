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

## 两阶段演进（为什么先"离线可跑"再"Docker 真服务"）

本机硬约束：无 Docker、无 PG/ES/Qdrant/Redis 服务、可用 LLM key 仅 Anthropic。
因此**真服务可切换的代码先落**，真服务留阶段 B —— 同一套 provider 工厂，业务零改动：

| 能力 | 阶段 A（现在，离线默认） | 阶段 B（Docker 装好后） |
|---|---|---|
| Embedding | MockEmbedding（确定性伪向量） | DashScope/text-embedding-v3 或本地 |
| 向量库 | Qdrant `:memory:` / 本地 path | Qdrant server（`deploy/docker-compose`） |
| 视觉 | MockVision（fixture 固定返回） | Claude 视觉（anthropic provider） |
| LLM 决定器 | MockDecision（确定性规则） | 同接口接 Claude 结构化输出 |
| 会话存储 | InMemory / File | Postgres JSONB（同 SessionStore 接口） |
| 缓存 | 无 | Redis（同 Cache 接口，预留） |

切 `config.mode=dev` + 真 key 后，`AgentRuntime.build()` / `Retriever` / 路由代码都不改。

## 关键取舍（面试可展开）

1. **单 Qdrant，不上 ES**：语料量级与语义召回靠向量足够；上双引擎是过度设计，
   面试被问"为什么不用 ES"可以答：候选集合小、真实场景可平滑扩展，工程上先做对再做重。
2. **一份 Product = 一份 Markdown 文档入库**：结构化 spec 仍在 pydantic 层（供比价/筛选），
   入库文本带标题组织——检索命中即拿 SKU 溯源，引用能说清出处。
3. **LangGraph 只做编排，不是"巨型 LLM"**：图里多数节点是确定性代码；"哪步调什么工具"
   由决定器输出（离线=规则，真 LLM=同接口换实现）；预算算术/规格对齐绝不给模型算。
4. **能算的绝不让模型算**：全项目统一的诚实原则，预算、Recall/MRR、grounding 都是代码计数。

## 运行方式

- 离线：`uv run pytest` · `uv run python -m evals.run` · `uv run uvicorn app.main:app --port 8000`
- 测试/评测与业务共用同一批 provider 工厂，注入 `path=":memory:"` 保证隔离、不污染 `./data`。

## 阶段 B 待办（Roadmap）

1. `deploy/docker-compose.yml`：postgres:16 + redis:7 + qdrant
2. SessionStore Postgres JSONB 适配器 + Redis cache + 向量切 qdrant server；`.env.docker`
3. 真 LLM/视觉联调（Claude）+ 真 embedding；集成验证 + 一次真实图文对话冒烟
4. 语义评测另开量表（mock 数值不迁移到真服务宣称）
