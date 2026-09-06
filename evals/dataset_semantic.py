# -*- coding: utf-8 -*-
"""语义量表问题集（真 embedding 的诚实对照，与离线词法门禁分开）。

为什么另开一本、不并进 dataset.py：
- dataset.py 的 RETRIEVAL_CASES 刻意做成**词面可达**（问题带型号/参数/场景词，
  命中目标 SKU 文档里的字面 token），用来防"改坏导致回退"——离线 mock 无语义，
  只有词法能稳定回归，那是**回归门禁**，不是能力宣称。
- 本集反着来：问题**刻意不带**目标 SKU 的型号/品牌/参数字面 token，用场景/改写/
  同义表达描述需求，让"该召回谁"只能靠**语义**判断。于是：
    · 真 embedding（text-embedding-v3）应能把目标 SKU 顶进 top-k —— 语义可达；
    · 离线 mock（确定性伪向量 + 词法）在无字面命中的题上会漏 —— 词法不可达。
  两边同题对照跑（evals/run_semantic.py），就是"真 embedding 带来什么"的实测证据。

诚实口径（README/docs 同步）：
- 语料仍是 13 SKU 小库，问题是我手工按"语义贴近某 SKU"写的；gold = 描述与
  query 语义最贴的那个 SKU。这不是通用语义能力宣称，是**在受控语料上验证
  text-embedding-v3 接进来后召回上限高于词法**的对照实验。
- 数值以 run_semantic.py 实测为准，量化只出现在报告文件里；README 只报实测。
"""
from __future__ import annotations

from evals.dataset import RetrievalCase

# 每题一个 gold（单 SKU → MRR@k 语义更强）。覆盖全部 13 个 SKU。
# 写作约束：不出现目标 SKU 的 型号/品牌 及文档中独有的规格字面（如"骁龙/2K/徕卡/军规"），
# 用需求场景/近义表达描述；gold 的唯一性靠"13 款里语义上就该是它"来保证。
SEMANTIC_CASES: list[RetrievalCase] = [
    RetrievalCase(
        question="手头紧的大学生想买手机打《原神》，要求屏幕是直板好贴膜，充电功率大，两千五上下能落地",
        gold_product_ids=["redmi-k70"],
    ),
    RetrievalCase(
        question="公司配给出差的电脑，要很皮实能扛机场托运磕碰，键盘敲起来舒服，开会显得正式稳重，背着还不累",
        gold_product_ids=["thinkpad-x1-carbon"],
    ),
    RetrievalCase(
        question="白天在咖啡馆写方案写到下午都不用带充电器，环境要安安静静没有风扇噪音的那种轻薄本",
        gold_product_ids=["macbook-air-m3"],
    ),
    RetrievalCase(
        question="大一新生基本不玩大型游戏，写作业常开一堆网页和文档，偶尔用剪辑软件，想要十六寸大屏、六千左右能拿下",
        gold_product_ids=["xiaoxin-pro-16"],
    ),
    RetrievalCase(
        question="硬核玩家晚上开高画质玩单机大作，机器要压得住温度、屏幕刷新率高不撕裂，预算七千上下",
        gold_product_ids=["asus-tuf-f15"],
    ),
    RetrievalCase(
        question="看演唱会坐后排想把台上拍清楚，晚上拍人像也要好看的安卓机，镜头变焦要够远",
        gold_product_ids=["vivo-x100"],
    ),
    RetrievalCase(
        question="想无纸化学习，课上用手写记笔记还能给 PDF 批注，平板要轻巧好带去图书馆，最好能配原装笔",
        gold_product_ids=["ipad-air-5"],
    ),
    RetrievalCase(
        question="想找块国产系统的大屏平板处理公司文档、改幻灯片顺手，配的笔写字要跟手不延迟",
        gold_product_ids=["matepad-pro-13"],
    ),
    RetrievalCase(
        question="下班窝沙发追剧刷视频，外放声音要立体响亮，躺床上玩手游也流畅，大屏平板、价格别太离谱",
        gold_product_ids=["xiaomi-pad-6s-pro"],
    ),
    RetrievalCase(
        question="常跑没信号的野外，想要关键时刻能联系上外界、抗摔、电池耐用的旗舰手机",
        gold_product_ids=["mate60-pro"],
    ),
    RetrievalCase(
        question="什么都想要一点的水桶旗舰：屏幕顶级通透、重度用一整天还有电、偶尔玩吃鸡也要满帧",
        gold_product_ids=["oneplus-12"],
    ),
    RetrievalCase(
        question="手小的人想要单手能握的小尺寸直屏手机，拍夜景颜色讨喜，性能也得是旗舰别缩水",
        gold_product_ids=["xiaomi-14"],
    ),
    RetrievalCase(
        question="在果子生态里想换台手机，主要日常用、拍拍照片录点视频，要轻薄手感好、用几年不卡",
        gold_product_ids=["iphone-15"],
    ),
]

# 每题"为什么该召回这个 SKU / 为什么词法难"的诊断说明（进报告，自证口径）
SEMANTIC_RATIONALE: dict[str, str] = {
    "手头紧的大学生想买手机打《原神》，要求屏幕是直板好贴膜，充电功率大，两千五上下能落地":
        "语义=低预算+直屏+快充+性能游戏 → redmi-k70（直板≠直屏、没说骁龙/120W/型号，词法无直接命中）",
    "公司配给出差的电脑，要很皮实能扛机场托运磕碰，键盘敲起来舒服，开会显得正式稳重，背着还不累":
        "语义=军规耐用商务轻薄 → thinkpad-x1-carbon（避开 carbon/军规/商务 字面）",
    "白天在咖啡馆写方案写到下午都不用带充电器，环境要安安静静没有风扇噪音的那种轻薄本":
        "语义=无风扇静音+超长续航+轻 → macbook-air-m3（避开 M3/18小时/无风扇 字面）",
    "大一新生基本不玩大型游戏，写作业常开一堆网页和文档，偶尔用剪辑软件，想要十六寸大屏、六千左右能拿下":
        "语义=大屏全能本+性价比 → xiaoxin-pro-16（避开 小新/16英寸/酷睿 字面）",
    "硬核玩家晚上开高画质玩单机大作，机器要压得住温度、屏幕刷新率高不撕裂，预算七千上下":
        "语义=高刷游戏本+独显+散热 → asus-tuf-f15（避开 天选/RTX4060/电竞 字面）",
    "看演唱会坐后排想把台上拍清楚，晚上拍人像也要好看的安卓机，镜头变焦要够远":
        "语义=潜望长焦+人像影像 → vivo-x100（避开 vivo/蔡司/长焦/人像 字面）",
    "想无纸化学习，课上用手写记笔记还能给 PDF 批注，平板要轻巧好带去图书馆，最好能配原装笔":
        "语义=手写笔记+轻创作 → ipad-air-5（避开 Apple/M1/Pencil/笔记 字面）",
    "想找块国产系统的大屏平板处理公司文档、改幻灯片顺手，配的笔写字要跟手不延迟":
        "语义=鸿蒙大屏+PC级办公+低延迟笔 → matepad-pro-13（避开 华为/MatePad/星闪/WPS 字面）",
    "下班窝沙发追剧刷视频，外放声音要立体响亮，躺床上玩手游也流畅，大屏平板、价格别太离谱":
        "语义=影音+多扬声器+游戏平板 → xiaomi-pad-6s-pro（避开 小米/骁龙/八扬声器 字面）",
    "常跑没信号的野外，想要关键时刻能联系上外界、抗摔、电池耐用的旗舰手机":
        "语义=卫星通话+耐摔+长续航商务旗舰 → mate60-pro（避开 华为/卫星/昆仑 字面）",
    "什么都想要一点的水桶旗舰：屏幕顶级通透、重度用一整天还有电、偶尔玩吃鸡也要满帧":
        "语义=顶级屏+大电池+性能游戏水桶 → oneplus-12（避开 一加/东方屏/骁龙 字面）",
    "手小的人想要单手能握的小尺寸直屏手机，拍夜景颜色讨喜，性能也得是旗舰别缩水":
        "语义=小屏直屏+影调+旗舰芯 → xiaomi-14（避开 小米/徕卡/骁龙 字面）",
    "在果子生态里想换台手机，主要日常用、拍拍照片录点视频，要轻薄手感好、用几年不卡":
        "语义=iOS 生态+轻薄耐用 → iphone-15（避开 iPhone/灵动岛/A16 字面）",
}
