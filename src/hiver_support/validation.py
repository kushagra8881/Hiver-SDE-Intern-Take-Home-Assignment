from __future__ import annotations

from typing import Any

import pandas as pd

from .evaluation import parse_bool
from .taxonomy import ESCALATION_REASONS, INTENTS


REQUIRED_GOLD_COLUMNS = {
    "gold_id", "sample_stratum", "challenge_type", "split", "pair_id", "conversation_id",
    "text_fingerprint", "message", "context", "historical_reply", "primary_intent",
    "should_escalate", "escalation_reason", "required_reply_points", "forbidden_claims",
    "annotator", "annotation_notes", "review_status",
}


def validate_gold(
    gold: pd.DataFrame,
    pairs: pd.DataFrame | None = None,
    *,
    require_human_review: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(REQUIRED_GOLD_COLUMNS - set(gold.columns))
    if missing:
        errors.append(f"Missing columns: {', '.join(missing)}")
        return {"valid": False, "errors": errors, "warnings": warnings}
    if not 150 <= len(gold) <= 250:
        errors.append(f"Expected 150-250 rows; found {len(gold)}")
    for column in ("gold_id", "pair_id", "conversation_id", "text_fingerprint"):
        duplicates = int(gold[column].astype(str).duplicated().sum())
        if duplicates:
            errors.append(f"{column} has {duplicates} duplicates")
    invalid_intents = sorted(set(gold["primary_intent"].dropna().astype(str)) - set(INTENTS))
    if invalid_intents:
        errors.append(f"Invalid intents: {', '.join(invalid_intents)}")
    invalid_splits = sorted(set(gold["split"].astype(str)) - {"dev", "test"})
    if invalid_splits:
        errors.append(f"Invalid splits: {', '.join(invalid_splits)}")
    invalid_strata = sorted(set(gold["sample_stratum"].astype(str)) - {"representative", "challenge"})
    if invalid_strata:
        errors.append(f"Invalid strata: {', '.join(invalid_strata)}")
    invalid_reasons = sorted(set(gold["escalation_reason"].dropna().astype(str)) - set(ESCALATION_REASONS))
    if invalid_reasons:
        errors.append(f"Invalid escalation reasons: {', '.join(invalid_reasons)}")
    for row in gold.itertuples(index=False):
        should_escalate = parse_bool(row.should_escalate)
        if should_escalate and row.escalation_reason == "none":
            errors.append(f"{row.gold_id}: escalate=true cannot use reason=none")
        if not should_escalate and row.escalation_reason != "none":
            errors.append(f"{row.gold_id}: auto-eligible row must use reason=none")
    unreviewed = int(gold["review_status"].astype(str).str.lower().ne("reviewed").sum())
    if unreviewed and require_human_review:
        errors.append(f"{unreviewed} rows are not marked reviewed")
    elif unreviewed:
        llm_reviewed = int(gold["review_status"].astype(str).str.lower().eq("llm_reviewed").sum())
        warnings.append(
            f"DEVELOPMENT ONLY: {unreviewed} rows are not human-reviewed"
            + (f" ({llm_reviewed} independently LLM-reviewed)" if llm_reviewed else "")
        )
    if gold["annotator"].astype(str).str.strip().eq("").any():
        errors.append("Every row needs an annotator")
    if pairs is not None:
        missing_ids = set(gold["pair_id"].astype(str)) - set(pairs["pair_id"].astype(str))
        if missing_ids:
            errors.append(f"{len(missing_ids)} gold pair IDs are absent from processed pairs")
    counts = {
        "rows": int(len(gold)),
        "dev": int((gold["split"] == "dev").sum()),
        "test": int((gold["split"] == "test").sum()),
        "representative": int((gold["sample_stratum"] == "representative").sum()),
        "challenge": int((gold["sample_stratum"] == "challenge").sum()),
        "escalate": int(gold["should_escalate"].map(parse_bool).sum()),
        "intent_counts": gold["primary_intent"].value_counts().to_dict(),
    }
    if min(counts["intent_counts"].values(), default=0) < 5:
        warnings.append("At least one intent has fewer than five examples")
    return {"valid": not errors, "errors": errors, "warnings": warnings, "counts": counts}
