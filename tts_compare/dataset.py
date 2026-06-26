"""Load the quick-test sentence set (produced by scripts/select_quick_test.py)."""

from __future__ import annotations

import json

from .config import QUICK_TEST


def load_quick_test() -> list[dict]:
    with QUICK_TEST.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
