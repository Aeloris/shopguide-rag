# -*- coding: utf-8 -*-
"""多模态视觉入口：图片 → 结构化商品特征 / 检索 Query。

真实实现（anthropic/Claude 视觉）在阶段 B（Docker + key 就绪后）接入 —— v1 只留接口与
离线 Mock，避免"以为接了真视觉"的假象。Mock 读 fixtures/vision/sample.json 固定返回，
测试/冒烟确定性。视觉低置信/不可识别 → 降级为文本引导（v1 不做真 OCR，诚实声明）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

from config.settings import Settings
from pydantic import BaseModel, Field


class VisionExtraction(BaseModel):
    """一张图的结构化抽取结果（v1 只取文本语义，供并入检索 query）。"""

    text: str = Field(description="图的自然语言描述/用户想表达的需求")
    keywords: list[str] = Field(default_factory=list)


@runtime_checkable
class VisionProvider(Protocol):
    async def describe(self, image_bytes: bytes) -> VisionExtraction: ...


class MockVision:
    """离线视觉 Mock：读 fixture 固定返回（换图换结果只需改 fixture）。"""

    def __init__(self, settings: Settings) -> None:
        self._fixture_path: Path = settings.fixtures_path / "vision" / "sample.json"
        self.calls: int = 0

    async def describe(self, image_bytes: bytes) -> VisionExtraction:
        self.calls += 1
        payload = json.loads(self._fixture_path.read_text(encoding="utf-8"))
        return VisionExtraction(**payload)


def get_vision_provider(settings: Settings) -> VisionProvider:
    """按 config.vision.provider 产出实现：mock | anthropic | dashscope(Qwen-VL)。"""
    if settings.vision.provider == "mock":
        return MockVision(settings)
    if settings.vision.provider == "anthropic":
        from llm.anthropic import AnthropicVision  # 惰性导入避免 llm 层循环依赖

        return AnthropicVision(settings)
    if settings.vision.provider == "dashscope":
        from llm.dashscope_vision import DashScopeQwenVision

        return DashScopeQwenVision(settings)
    raise ValueError(
        f"未知 vision.provider：{settings.vision.provider}（可选 mock|anthropic|dashscope）"
    )
