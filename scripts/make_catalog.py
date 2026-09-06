# -*- coding: utf-8 -*-
"""确定性生成 fixtures/catalog/products.json（种子库唯一数据源）。

用法：uv run python scripts/make_catalog.py
产物提交进版本库；重复运行输出不变（无时间戳），供回归比对。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import REPO_ROOT
from core.catalog.seeds import build_seed_dicts


def main() -> None:
    out = REPO_ROOT / "fixtures" / "catalog" / "products.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    products = build_seed_dicts()
    out.write_text(
        json.dumps(products, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(products)} products -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
