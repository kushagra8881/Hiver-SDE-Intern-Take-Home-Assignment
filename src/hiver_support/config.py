from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Settings:
    brand: str = "SpotifyCares"
    random_seed: int = 1729
    chunksize: int = 250_000
    gold_size: int = 200
    gold_test_fraction: float = 0.50
    retrieval_max_pairs: int = 30_000
    retrieval_top_k: int = 3
    min_intent_confidence: float = 0.48
    min_retrieval_similarity: float = 0.16
    gemini_model: str = "gemini-3.5-flash-lite"
    judge_sample_size: int = 40


def load_settings(path: str | Path | None = None) -> Settings:
    if path is None:
        return Settings()
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = Settings.__dataclass_fields__
    unexpected = sorted(set(raw) - set(known))
    if unexpected:
        raise ValueError(f"Unknown config keys: {', '.join(unexpected)}")
    return Settings(**raw)
