# 混合检索：BM25 + Dense → RRF → 重排

## 链路（`core/retriever/service.py::Retriever.search`）

```
query ─▶ tokenize(自研中文分词)
   ├─ Dense：query 向量 → Qdrant search（dense_top_k=20）
   └─ BM25：惰性从库内全量重建 BM25Index（词法 top_k=20）
   RRF 融合（k=60，按"名次"融合，量纲不同的两路不直接相加）
   → MockReranker 词法精排（rerank_top_n=8）
   → 去重到 SKU 粒度，输出 top5 + 命中摘录（溯源用）
```

## 每个子模块为什么要这么写（面试可讲）

- **自研中文分词（tokens.py）而非 jieba**：少一个大依赖、可控。连续 CJK 取 1–2 gram，
  连续英数型号串（如 `Mate60Pro`/`RTX4060`）整段保留——型号必须整段匹配不能被切碎。
  **文档与 query 走同一函数 → token 恒对齐，排名行为确定可测**。
- **BM25（bm25.py）**：经典 `k1=1.5, b=0.75`，IDF 平滑。小语料每次现算可接受，可预建索引优化。
- **RRF（fusion.py）**：用"名次"而非"分数"融合——向量余弦与 BM25 词频量纲/分布不同，
  直接相加无意义；名次在各检索器间可比，且对单路异常高分稳健（TREC 常用）。
- **重排（rerank.py）**：默认词法覆盖度（确定性）；阶段 B 换真 CrossEncoder，同一接口。
- **为什么 Dense 用 mock 还能 demo**：MockEmbedding 只是"同文同向量"的伪向量（无语义），
  离线 demo 的命中主要由 BM25/词法重排撑住 —— 这正是评测"词面可达"口径的前提，README 已声明。

## 数据组织（一份 SKU = 一份文档）

`Product`（pydantic，结构化 spec dict）→ `product_to_markdown` 渲染成带标题的 Markdown →
Ingest 整包重建写进 Qdrant。payload 同时存结构化元数据（name/brand/category/price_cny）
与全文 text——命中即拿 SKU、即拿原文做摘录/引用；BM25 重建与摘录都直接读 payload。

## 检索层测试关注点

- token 原语：型号整段保留 / CJK 单字+双字 / 大小写与空白确定性
- BM25：重复词命中 doc 优先、可回放
- RRF：跨路名次融合的稳定性
- e2e：`test_retriever.py` 覆盖预算/办公/人像三类问法 + 结果按 SKU 去重 + RRF 分随名次严格递减

评测（gold Recall@5/MRR@5 = 1.000）见 [eval.md](eval.md)。
