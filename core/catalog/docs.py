# -*- coding: utf-8 -*-
"""把结构化 Product 渲染成入库用 Markdown（一份 SKU = 一份文档）。

为什么渲染成 Markdown 而不是直接拼字符串：
- 检索/比价工具吃结构化 Product，但**入库进 Qdrant 的是文本块**；
  把规格、卖点、场景组织成带标题的 Markdown，复用标题感知 chunker，
  命中后自带 heading_path → 引用能说清"来自该 SKU 的哪一节"。
- source 用 product.id：检索命中即知 SKU，无需反查文件。
"""
from __future__ import annotations

from core.catalog.schemas import Product

_CATEGORY_LABEL = {"phone": "手机", "laptop": "笔记本", "tablet": "平板"}


def product_to_markdown(p: Product) -> str:
    lines: list[str] = []
    lines.append(f"# {p.name}")
    lines.append("")
    lines.append(
        f"- 品类：{_CATEGORY_LABEL.get(p.category, p.category)}　"
        f"- 品牌：{p.brand}　- 型号：{p.model}　- 价格：{p.display_price}"
    )
    if p.tags:
        lines.append(f"- 标签：{' / '.join(p.tags)}")
    lines.append("")

    lines.append("## 一句话卖点")
    lines.append(p.summary)
    lines.append("")

    if p.highlights:
        lines.append("## 核心卖点")
        for h in p.highlights:
            lines.append(f"- {h}")
        lines.append("")

    if p.specs:
        lines.append("## 规格参数")
        for k, v in p.specs.items():
            lines.append(f"- {k}：{v}")
        lines.append("")

    lines.append("## 适合人群与使用场景")
    if p.tags:
        lines.append("适合" + "、".join(p.tags) + "的用户。")
    else:
        lines.append("通用选择。")
    lines.append("")
    return "\n".join(lines)
