# -*- coding: utf-8 -*-
"""配置加载：config/config.yaml（业务参数） + .env / 环境变量（密钥）。

用法：
    from config.settings import get_settings
    s = get_settings()
    s.mode                    # -> "offline"
    s.retrieval.rrf_k         # -> 60

设计要点（与 bid-response-rag-engine 同源，防返工）：
- 全参数在 config.yaml 占位 + 强类型模型；拼错字段启动即报错。
- 密钥只从环境变量 / .env 读，绝不出现在 yaml。
- provider 指向真服务却缺 key 时，启动即抛清晰错误（fail fast）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"

load_dotenv(REPO_ROOT / ".env")


class AppConfig(BaseModel):
    name: str = "shopguide-rag"
    log_level: str = "INFO"
    data_dir: str = "./data"
    fixtures_dir: str = "./fixtures"


class CatalogConfig(BaseModel):
    file: str = "./fixtures/catalog/products.json"


class LLMConfig(BaseModel):
    provider: str = "mock"  # mock | anthropic
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    timeout_sec: float = 60.0
    temperature: float = 0.3

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)

    @model_validator(mode="after")
    def _fail_fast_when_anthropic_without_key(self) -> "LLMConfig":
        if self.provider == "anthropic" and not self.api_key:
            raise ValueError(
                f"llm.provider=anthropic 但环境变量 {self.api_key_env} 未设置。\n"
                "解决办法：复制 .env.example 为 .env 填入真实 key；"
                "或把 config.yaml 的 llm.provider 保持为 mock 离线运行。"
            )
        return self


class VisionConfig(BaseModel):
    provider: str = "mock"  # mock | anthropic
    model: str = "claude-sonnet-5"
    api_key_env: str = "ANTHROPIC_API_KEY"

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env)


class EmbeddingConfig(BaseModel):
    provider: str = "mock"  # mock | dashscope（阶段 B 预留）
    model: str = "text-embedding-v3"
    dimension: int = 1024
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class VectorDBConfig(BaseModel):
    provider: str = "qdrant_local"  # qdrant_local | qdrant_server
    path: str = "./data/qdrant"
    collection: str = "shopguide_products"
    url: str = "http://localhost:6333"


class ChunkingConfig(BaseModel):
    max_chars: int = 800
    overlap_chars: int = 80

    @model_validator(mode="after")
    def _overlap_must_be_below_max(self) -> "ChunkingConfig":
        if self.overlap_chars >= self.max_chars:
            raise ValueError(
                f"chunking.overlap_chars({self.overlap_chars}) 必须小于 "
                f"chunking.max_chars({self.max_chars})，否则超长文本切块死循环"
            )
        return self


class RetrievalConfig(BaseModel):
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_n: int = 8
    final_top_n: int = 5


class AgentConfig(BaseModel):
    max_tool_rounds: int = 3
    min_citations: int = 1
    temperature: float = 0.3


class StoreConfig(BaseModel):
    provider: str = "memory"  # memory | file | postgres
    path: str = "./data/sessions"
    dsn_env: str = "DATABASE_URL"  # provider=postgres 时从该环境变量读 DSN


class CacheConfig(BaseModel):
    provider: str = "off"  # off | memory | redis
    ttl_sec: int = 300
    dsn_env: str = "REDIS_URL"


class EvalThresholds(BaseModel):
    """评测门禁（offline mock 确定性基线，防引擎回退；非"我多好"的宣称）。"""

    hybrid_recall_at_k: float = 0.9
    hybrid_mrr_at_k: float = 0.9
    grounded_rate_min: float = 1.0
    bad_refusal_rate_min: float = 1.0


class EvalConfig(BaseModel):
    output_dir: str = "./data/eval"
    top_k: int = 5
    thresholds: EvalThresholds = Field(default_factory=EvalThresholds)


class Settings(BaseModel):
    mode: str = "offline"  # offline | dev
    app: AppConfig = Field(default_factory=AppConfig)
    catalog: CatalogConfig = Field(default_factory=CatalogConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "Settings":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(**data)

    @property
    def repo_root(self) -> Path:
        return REPO_ROOT

    @property
    def data_path(self) -> Path:
        p = self.repo_root / self.app.data_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def fixtures_path(self) -> Path:
        p = self.repo_root / self.app.fixtures_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_offline(self) -> bool:
        """离线判定：真服务件(LLM/视觉/embedding/PG/Redis)全部为 mock/本地时的安全默认。"""
        return (
            self.llm.provider == "mock"
            and self.vision.provider == "mock"
            and self.embedding.provider == "mock"
            and self.vector_db.provider != "qdrant_server"
            and self.store.provider in ("memory", "file")
            and self.cache.provider in ("off", "memory")
        )


@lru_cache(maxsize=1)
def get_settings(path: str | Path | None = None) -> Settings:
    """进程内共享同一份配置；测试里想用别的 yaml 可传 path。

    默认读 config/config.yaml；设了环境变量 SHOPGUIDE_CONFIG 则读该文件（阶段 B
    dev 模式用它指向 config/config.dev.yaml，一行切换真服务）。
    """
    if path is None:
        path = os.getenv("SHOPGUIDE_CONFIG") or DEFAULT_CONFIG_PATH
    return Settings.from_yaml(path)
