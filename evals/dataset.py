# -*- coding: utf-8 -*-
"""检索评测集（gold Q&A）与对抗不可答集。

诚实边界声明（README/docs/eval.md 同源，防面试虚标）：
- 本集面向**离线 mock 引擎的回归门禁**，不是"我的检索有多强"的宣称。
  问题刻意带目标 SKU 的区分性词面（型号/参数/品牌/场景词），验证
  BM25 + 向量 → RRF → 重排 全链路没有把"明显该召回"的目标漏掉/挤掉。
- 语义上限是"该用真 embedding 的量表"，离线 mock 的向量无语义，召回主要靠词法 ——
  所以评测集刻意做成**词面可达**，门禁防的是"改坏导致回退"，而非衡量语义天花板。
- 检索层只判 Recall@k / MRR@k（候选召回）；"预算/比价/不可答拒绝"不在检索层，
  由 Agent 工具与坏答防护负责 —— 对抗不可答清单列在这里，供 agent 层复用。

数值口径：metrics 全为确定性代码计数（Recall@k 去重命中 / MRR@k 首中倒排名次）。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalCase(BaseModel):
    """一条"该召回哪些 SKU"的 gold 检索样例。"""

    question: str = Field(description="用户问法/检索 query")
    gold_product_ids: list[str] = Field(
        description="应被召回的目标 SKU id（可多个；空列表则该条无效，不合法）"
    )


class UnanswerableCase(BaseModel):
    """对抗不可答样例：Agent 必须拒绝/转引导，而不是硬答。"""

    query: str
    reason: str = Field(description="为什么不可答（供诊断/README 说明）")


# 词面刻意带区分性 token：型号/品牌/参数都应在目标 SKU 的 Markdown 文档里出现，
# 保证 BM25 + 词法重排在离线确定性下稳定召回 → Recall/MRR 可回归。
RETRIEVAL_CASES: list[RetrievalCase] = [
    RetrievalCase(question="Redmi K70 2K 直屏 快充 性价比", gold_product_ids=["redmi-k70"]),
    RetrievalCase(question="ThinkPad X1 Carbon 商务 轻薄 耐用", gold_product_ids=["thinkpad-x1-carbon"]),
    RetrievalCase(question="MacBook Air M3 无风扇 长续航", gold_product_ids=["macbook-air-m3"]),
    RetrievalCase(question="vivo X100 蔡司 长焦 人像", gold_product_ids=["vivo-x100"]),
    RetrievalCase(question="华硕 天选 RTX 4060 电竞 游戏本", gold_product_ids=["asus-tuf-f15"]),
    RetrievalCase(question="小米平板 6S Pro 骁龙 影音 大电池", gold_product_ids=["xiaomi-pad-6s-pro"]),
    RetrievalCase(question="华为 MatePad Pro 星闪 手写笔 办公", gold_product_ids=["matepad-pro-13"]),
    RetrievalCase(question="小米 14 徕卡 小屏 旗舰", gold_product_ids=["xiaomi-14"]),
    RetrievalCase(question="iPhone 15 灵动岛 iOS 轻薄", gold_product_ids=["iphone-15"]),
    RetrievalCase(question="华为 Mate 60 Pro 卫星通话 商务", gold_product_ids=["mate60-pro"]),
    RetrievalCase(question="联想 小新 Pro 16 大屏 全能本", gold_product_ids=["xiaoxin-pro-16"]),
    RetrievalCase(question="iPad Air 苹果 手写笔 创作 笔记", gold_product_ids=["ipad-air-5"]),
]

# 对抗不可答（agent 层坏答防护的评测输入，检索层不处理）：
UNANSWERABLE_CASES: list[UnanswerableCase] = [
    UnanswerableCase(query="这台华为笔记本能玩 3A 大作吗", reason="库内无华为笔记本；实时体验需真机/评测数据"),
    UnanswerableCase(query="Redmi K70 现在拼多多百亿补贴多少钱", reason="实时渠道价：本项目无价格爬虫，价格是静态样例"),
    UnanswerableCase(query="帮我下单买一台 iPhone 15", reason="交易动作：本项目不做下单/支付"),
    UnanswerableCase(query="苹果下季度会发布什么新机", reason="未来信息：无新闻/预测数据源"),
]
