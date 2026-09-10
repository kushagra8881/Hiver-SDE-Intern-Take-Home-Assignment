from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LABEL_COLUMNS = [
    "primary_intent",
    "secondary_intent",
    "should_escalate",
    "escalation_reason",
    "required_reply_points",
    "forbidden_claims",
    "annotator",
    "annotation_notes",
    "review_status",
]


def _unique_cases(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["conversation_id", "text_fingerprint", "reply_quality", "pair_id"],
        ascending=[True, True, False, True],
    )
    return ranked.drop_duplicates("conversation_id").drop_duplicates("text_fingerprint")


def sample_gold_candidates(
    pairs: pd.DataFrame,
    output_path: str | Path,
    *,
    size: int = 200,
    seed: int = 1729,
    representative_fraction: float = 0.75,
) -> pd.DataFrame:
    if not 150 <= size <= 250:
        raise ValueError("Golden set size must be between 150 and 250")
    rng = np.random.default_rng(seed)
    candidates = _unique_cases(pairs).copy()
    candidates = candidates[candidates["message"].str.len().between(4, 600)]
    candidates["_created"] = pd.to_datetime(
        candidates["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce"
    )
    candidates = candidates[candidates["_created"].notna()].copy()
    # A prospective evaluation: sample only the latest 30% of eligible cases.
    cutoff = candidates["_created"].quantile(0.70)
    candidates = candidates[candidates["_created"] >= cutoff].copy()

    representative_n = round(size * representative_fraction)
    representative = candidates.sample(n=representative_n, random_state=seed)
    representative = representative.assign(sample_stratum="representative", challenge_type="prevalence")

    remaining = candidates.drop(index=representative.index).copy()
    remaining["_noise"] = rng.uniform(0.0, 0.001, len(remaining))
    selected_parts: list[pd.DataFrame] = []
    selected_indices: set[int] = set()

    def take(name: str, mask: pd.Series, count: int, score: pd.Series | None = None) -> None:
        available = remaining[mask & ~remaining.index.isin(selected_indices)].copy()
        if score is not None:
            available["_rank"] = score.loc[available.index] + available["_noise"]
            part = available.nlargest(min(count, len(available)), "_rank").drop(columns="_rank")
        else:
            part = available.sample(min(count, len(available)), random_state=seed + len(selected_parts) + 1)
        part["challenge_type"] = name
        selected_parts.append(part)
        selected_indices.update(part.index)

    reasons = remaining["weak_risk_reason"].astype(str)
    take("billing_or_refund", reasons.eq("billing_or_refund"), 8)
    take("account_or_security", reasons.eq("account_or_security"), 6)
    take("legal_or_safety", reasons.isin(["legal_or_policy", "abuse_or_safety"]), 4)
    short_mask = remaining["message"].str.split().str.len().le(5) | reasons.eq("missing_context")
    take("short_or_missing_context", short_mask, 10, 1.0 - remaining["reply_quality"].astype(float))
    take("ambiguous_or_multi_intent", pd.Series(True, index=remaining.index), 8, remaining["weak_ambiguity"].astype(float))
    take("weak_historical_evidence", pd.Series(True, index=remaining.index), 8, 1.0 - remaining["reply_quality"].astype(float))
    rare = remaining["weak_intent"].isin(["device_connectivity", "playback_technical", "playlist_library"])
    take("rare_technical_intent", rare, 6)

    challenge = pd.concat(selected_parts, ignore_index=False)
    needed = size - representative_n
    if len(challenge) < needed:
        fill = remaining[~remaining.index.isin(selected_indices)].sample(needed - len(challenge), random_state=seed + 99)
        fill["challenge_type"] = "random_fill"
        challenge = pd.concat([challenge, fill])
    challenge = challenge.drop(columns="_noise", errors="ignore").head(needed)
    challenge = challenge.assign(sample_stratum="challenge")

    selected = pd.concat([representative, challenge], ignore_index=True)
    selected["gold_id"] = [f"SPOT-{idx:03d}" for idx in range(1, len(selected) + 1)]
    # Lock thresholds on the earlier half and report once on the later half.
    time_boundary = selected["_created"].median()
    selected["split"] = np.where(selected["_created"] <= time_boundary, "dev", "test")
    for column in LABEL_COLUMNS:
        selected[column] = ""
    ordered = [
        "gold_id", "sample_stratum", "challenge_type", "split", "created_at", "pair_id", "conversation_id",
        "text_fingerprint", "message", "context", "historical_reply",
        "primary_intent", "secondary_intent", "should_escalate",
        "escalation_reason", "required_reply_points", "forbidden_claims",
        "annotator", "annotation_notes", "review_status",
    ]
    selected = selected[ordered]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    return selected
