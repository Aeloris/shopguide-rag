# 导购 Agent：LangGraph 有界工具循环

## 图结构（`core/agent/graph.py`）

```
START → decide ──(tool)──▶ run_tool ──▶ decide   （有界循环，≤ max_tool_rounds=3）
                 ├─(final)──▶ finalize ─▶ END     （白名单收束结构化回执）
                 └─(refuse)─▶ refuse  ─▶ END      （拒绝出口，不浪费轮次）
```

决定器（`MockDecision`，离线=确定性规则）输出 `AgentDecision{kind, tool, args}`：
不可答硬规则最优先 → 未召回先 `search_products` → 再按意图补 `compare_products` /
`filter_products` → 规划用尽则 `final`。接真实 LLM 时换同接口实现，图与工具不动。

## 三工具全部是确定性代码（模型不算算术）

| 工具 | 做什么 | 为什么是代码 |
|---|---|---|
| `search_products(query)` | 走混合检索召回候选 | 召回 = 检索层本职 |
| `filter_products(budget, category, tags)` | `price <= budget` 判价 + 品类/标签过滤，预算内按价升序 | 预算比较可验证，交给模型会算术错 |
| `compare_products(ids, dims)` | 只对齐**两品共有**规格键，行对齐输出 | 缺列不硬补，维度语义由数据决定 |

## 防幻觉的工程形状（面试重点）

1. **引用 = candidates 白名单**：`search/filter` 的产物滚动成 state 里唯一可引用候选；
   `finalize` 只在白名单里取商品 → 回答的 grounding **由构造保证**，不是事后去抓 LLM 引用。
2. **不可答走独立 refuse 出口**：五类硬规则（trade/realtime/future/benchmark/no_match），
   词面命中即拒绝——宁可转引导不硬答；`benchmark` 需要"指向具体型号"才拒，
   避免把"推荐一台能玩3A的笔记本"这种合法导购误杀。
3. **有界循环**：轮次上限在条件边前强制 `final`，不存在无限循环；拒绝不消耗工具轮次。
4. **结构化回执**：`AgentReply{answerable, refused, recommendations, comparison, filter_note,
   tool_trace}`——渲染成人类文本在 app 层做，评测/门禁断言结构而非脆弱措辞。

## 多模态入口（`llm/vision.py` + `core/agent/runtime.py::ask`）

- 图片先进 VisionProvider（mock 读 fixture / 阶段 B anthropic）抽成文本需求；
- 图文同传以文字为主，仅传图用视觉描述兜底；
- `AgentRuntime.build()` 一次自举 Catalog+向量库+图，供多次 `ask` 复用。

## Agent 评测

grounded_rate（引用非空且全在库）/ bad_refusal_rate（好例不误拒）/ 不可答拦截率，
全部代码计数、离线回放，见 [eval.md](eval.md)。
