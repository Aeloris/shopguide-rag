# -*- coding: utf-8 -*-
"""Streamlit 演示前端 —— 直接 import 核心（不走 HTTP），离线一键可跑。

用法：
    uv sync                              # 首次（含 streamlit）
    uv run streamlit run scripts/demo_ui.py

三个 Tab：导购对话 / 商品库 / 混合检索调试。
- 默认 offline：mock LLM/视觉/embedding + Qdrant :memory:，无需任何 key/Docker。
- 设 SHOPGUIDE_CONFIG=config/config.live.yaml（且 .env 里有真 key）即自动走 live 真链
  （真 LLM 决定器 / 真语义 embedding / 真 Qwen-VL 视觉），UI 右下角如实显示当前 provider。
诚实口径（与 README 一致）：数字只对 13 SKU 离线语料成立；offline 传图走 Vision mock
固定描述，live 才是真 Qwen-VL；示例按钮只用"词面可达"的域内查询，不搬语义量表金句。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from config.settings import get_settings
from core.agent import AgentRuntime
from core.catalog.loader import load_catalog

_REFUSAL_LABEL = {
    "trade": "交易动作（本项目不做下单/支付）",
    "realtime": "实时渠道价/补贴行情（无爬虫，库内价格是静态样例）",
    "future": "未来发布/爆料（无信息源）",
    "benchmark": "具体型号真机实测（库内规格不能代替体验评测）",
    "no_match": "库内无匹配",
}

DEMO_GROUPS = [
    ("正常推荐（域内）", [
        "预算3000以内的安卓直屏手机",
        "适合办公的轻薄笔记本",
        "512g 拍人像好的手机",
        "小米14和vivo X100哪个拍照好",
        "推荐一台能玩 3A 大作的笔记本",
    ]),
    ("拒答案例（硬规则）", [
        "帮我下单买一台 iPhone 15",
        "Redmi K70 现在拼多多百亿补贴多少钱",
        "华为 Mate 60 Pro 下一代什么时候发布",
        "这台华为笔记本能玩 3A 大作吗",
    ]),
    ("无匹配 no_match", [
        "想买一台 iPhone 17",
        "柠檬洗洁精除油清新",
        "预算2000以内的手机",
    ]),
]


@st.cache_resource(show_spinner="正在构建离线运行时（Qdrant :memory: + mock，一次）…")
def get_runtime() -> AgentRuntime:
    """离线默认：AgentRuntime.build() = get_settings()(config.yaml) + qdrant :memory: + mock。"""
    return asyncio.run(AgentRuntime.build())


def _settings_or_none():
    try:
        return get_settings()
    except Exception:  # noqa: BLE001 - live 缺 key 时只影响 footer 显示，不让 UI 崩
        return None


@st.cache_resource
def get_catalog():
    return load_catalog(get_settings())


def _ask(text: str, images: list[bytes] | None = None):
    return asyncio.run(get_runtime().ask(text, images=images))


def render_recommendation(rec: dict) -> None:
    """一条推荐商品卡；specs/卖点从目录查，纯展示不改引擎。"""
    pid = rec["product_id"]
    prod = get_catalog().get(pid)
    with st.container(border=True):
        head = f"**{rec['name']}**"
        if rec.get("brand") and rec.get("model"):
            head += f"　<small>{rec['brand']} {rec['model']}</small>"
        st.markdown(head, unsafe_allow_html=True)
        st.caption(f"category={rec.get('category')} · price=¥{rec.get('price_cny')} · id={pid}")
        if rec.get("summary"):
            st.write(rec["summary"])
        if rec.get("snippet"):
            st.markdown(f"<small>命中片段：`{rec['snippet']}`</small>", unsafe_allow_html=True)
        if prod is not None and (prod.specs or prod.highlights or prod.tags):
            with st.expander("规格 / 卖点 / 标签"):
                if prod.specs:
                    st.dataframe(
                        [{"规格": k, "值": v} for k, v in prod.specs.items()],
                        hide_index=True, use_container_width=True,
                    )
                if prod.highlights:
                    st.markdown("卖点：\n" + "\n".join(f"- {h}" for h in prod.highlights))
                if prod.tags:
                    st.markdown("标签：`" + "` `".join(prod.tags) + "`")


def render_comparison(cmp) -> None:
    """compare 结构化结果 → 维度×商品 对照表。"""
    if not cmp or not cmp.rows:
        return
    df = pd.DataFrame(index=cmp.dims, columns=[])
    for row in cmp.rows:
        df[row.name] = [row.dims.get(d, "—") for d in cmp.dims]
    price_row = {row.name: f"¥{row.price_cny}" for row in cmp.rows}
    st.markdown("**参数对比（只对齐两品共有维度，缺维度标 —）**")
    st.dataframe(pd.DataFrame([{"维度": "价格", **price_row}]),
                 hide_index=True, use_container_width=True)
    st.dataframe(df.rename_axis("维度"), use_container_width=True)


def render_reply(r) -> None:
    if r.refused:
        kind = r.refusal_kind or "unknown"
        st.error(f"**不可答 · {_REFUSAL_LABEL.get(kind, kind)}**")
        st.write(r.refusal_reason or "")
        if not r.tool_trace:
            st.caption("（未调用任何工具 —— 硬规则先于检索/模型）")
        return
    if not r.answerable:
        st.warning("（未给出可答结果）")
        return
    for rec in r.recommendations:
        render_recommendation(rec)
    if r.comparison:
        render_comparison(r.comparison)
    if r.filter_note:
        st.caption(f"筛选口径：{r.filter_note}")
    with st.expander("grounding / 工具轨迹"):
        st.markdown(
            f"引用全部来自检索命中白名单：`{', '.join(r.cited_product_ids)}`"
            if r.cited_product_ids else "（本次无引用商品）"
        )
        st.markdown("工具轨迹：`" + (" → ".join(r.tool_trace) if r.tool_trace else "无") + "`")


def send_user(text: str, *, image_name: str | None = None, image_bytes: bytes | None = None) -> None:
    if image_bytes is not None:
        content = f"（上传图片 `{image_name}`，空文本 → 由视觉抽取需求）"
    else:
        content = text
    st.session_state.setdefault("messages", []).append({"role": "user", "content": content})
    try:
        reply = _ask(text, images=[image_bytes] if image_bytes is not None else None)
    except Exception as exc:  # noqa: BLE001 - 展示异常不拖垮整个 UI
        st.session_state["messages"].append({
            "role": "assistant", "content": None, "reply": None, "error": str(exc),
        })
        return
    st.session_state["messages"].append({
        "role": "assistant", "content": None, "reply": reply, "error": None,
    })


def sidebar_demo() -> None:
    st.sidebar.title("示例问题")
    st.sidebar.caption("点击即自动发送（词面可达，offline 可复现）")
    for group, queries in DEMO_GROUPS:
        st.sidebar.markdown(f"**{group}**")
        for q in queries:
            if st.sidebar.button(q, key=f"demo:{q}", use_container_width=True):
                send_user(q)


def tab_chat() -> None:
    st.markdown(
        "结构化回执演示：**可答** 给商品卡（引用只来自检索命中白名单）；"
        "**不可答** 给拒答 banner（交易/实时行情/未来/真机实测/库内无匹配五类）。"
    )
    for msg in st.session_state.get("messages", []):
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif msg.get("error"):
                st.error(f"引擎异常：{msg['error']}")
            else:
                render_reply(msg["reply"])
    with st.expander("📷 传图（多模态路径）"):
        up = st.file_uploader("上传一张商品图（png/jpg），离线=Vision mock 固定描述；live 才真 Qwen-VL",
                              type=["png", "jpg", "jpeg"], key="img_up")
        if up is not None:
            if st.button("用这张图问（空文本 → 视觉需求）", key="img_send"):
                send_user("", image_name=up.name, image_bytes=up.getvalue())
    prompt = st.chat_input("例如：预算3000以内的安卓直屏手机 …")
    if prompt:
        send_user(prompt)


def tab_catalog() -> None:
    cat = get_catalog()
    sel = st.selectbox("品类", ["全部", "phone", "laptop", "tablet"])
    prods = [p for p in cat.products if sel == "全部" or p.category == sel]
    st.caption(f"当前目录共 {len(cat.products)} 个 SKU（诚实口径：仅此离线语料，非真电商库存）")
    st.dataframe(
        [{"id": p.id, "name": p.name, "brand": p.brand, "model": p.model,
          "category": p.category, "price": f"¥{p.price_cny}", "summary": p.summary}
         for p in prods],
        hide_index=True, use_container_width=True,
    )
    chosen = st.selectbox(
        "查看某款详情",
        [p.id for p in prods],
        format_func=lambda i: next((p.name for p in prods if p.id == i), i),
    )
    p = cat.get(chosen)
    if p is not None:
        st.markdown(f"**{p.name}**　¥{p.price_cny} · {p.category} · id=`{p.id}`")
        st.write(p.summary)
        st.markdown("卖点：")
        st.markdown("\n".join(f"- {h}" for h in p.highlights) or "（无）")
        st.markdown("规格：")
        st.dataframe([{"规格": k, "值": v} for k, v in p.specs.items()],
                     hide_index=True, use_container_width=True)


def tab_retriever() -> None:
    st.markdown("混合检索调试口（= `POST /api/search`，`retriever.search` 直接调用）。")
    q = st.text_input("query", value="适合办公的轻薄笔记本", key="ret_q")
    if st.button("检索", key="ret_go") or q:
        hits = asyncio.run(get_runtime().retriever.search(q))
        if not hits:
            st.warning("零命中 → Agent 会走 no_match 拒答（域外/库外）")
        else:
            st.dataframe(
                [{"rank": i + 1, "product_id": h.product_id, "name": h.name,
                  "price": f"¥{h.price_cny}", "rrf_score": round(h.rrf_score, 4),
                  "snippet": h.snippet}
                 for i, h in enumerate(hits)],
                hide_index=True, use_container_width=True,
            )
        st.caption("BM25 + Dense → RRF(k=60) → 词法重排，final_top_n=5；offline mock 向量无语义，词法为主。")


def footer() -> None:
    s = _settings_or_none()
    if s is None:
        seg = "mode=?(live 缺 key)"
    else:
        seg = (f"mode={s.mode} · llm={s.llm.provider} · vision={s.vision.provider} · "
               f"embedding={s.embedding.provider}")
    st.divider()
    st.caption(
        f"{seg}　诚实口径：数字只对 13 SKU 离线语料成立；offline 图片=Vision mock，"
        "live(config.live.yaml + key) 才真 Qwen-VL；示例按钮均为词面可达查询，不虚标语义能力。"
    )


def main() -> None:
    st.set_page_config(page_title="shopguide-rag · 多模态导购 Agent 演示", page_icon="🛍️", layout="wide")
    st.title("🛍️ shopguide-rag · 多模态 RAG 电商导购 Agent")
    st.caption("LangGraph 有界工具循环 · BM25+Dense→RRF · grounding=只引用检索命中白名单")
    sidebar_demo()
    t_chat, t_cat, t_ret = st.tabs(["💬 导购对话", "📦 商品库", "🔍 混合检索调试"])
    with t_chat:
        tab_chat()
    with t_cat:
        tab_catalog()
    with t_ret:
        tab_retriever()
    footer()


if __name__ == "__main__":
    main()
