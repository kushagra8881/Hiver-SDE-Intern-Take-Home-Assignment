#!/usr/bin/env python3
"""Small resumable terminal labeller. It intentionally has no auto-label button."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hiver_support.taxonomy import ESCALATION_REASONS, INTENT_DESCRIPTIONS


def choose(prompt: str, choices: list[str], allow_blank: bool = False, default: str = "") -> str:
    while True:
        print(prompt)
        for index, value in enumerate(choices, 1):
            print(f"  {index}. {value}")
        suffix = f" [Enter keeps {default}]" if default else ""
        raw = input(f">{suffix} ").strip()
        if not raw and default in choices:
            return default
        if allow_blank and not raw:
            return ""
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        print("Choose a listed number or value.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default="data/gold/spotify_golden_v1.csv")
    parser.add_argument("--annotator", required=True, help="Your initials; AI-assisted drafts are not human labels")
    args = parser.parse_args()
    path = Path(args.csv)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    pending = frame.index[frame["review_status"].str.lower() != "reviewed"]
    for number, index in enumerate(pending, 1):
        row = frame.loc[index]
        print("\n" + "=" * 88)
        print(f"{row['gold_id']} | {row['sample_stratum']} | pending {number}/{len(pending)}")
        print(f"CONTEXT: {row['context'] or '(none)'}")
        print(f"MESSAGE: {row['message']}")
        print(f"HISTORICAL REPLY (context only; do not blindly copy its decision): {row['historical_reply']}")
        print("ASSISTANT DRAFT:")
        print(f"  intent={row['primary_intent'] or '(blank)'} secondary={row['secondary_intent'] or '(none)'}")
        print(f"  escalate={row['should_escalate'] or '(blank)'} reason={row['escalation_reason'] or '(blank)'}")
        print(f"  required={row['required_reply_points'] or '(blank)'}")
        print(f"  forbidden={row['forbidden_claims'] or '(blank)'}")
        intent = choose("Primary intent", list(INTENT_DESCRIPTIONS), default=row["primary_intent"])
        secondary = choose(
            "Secondary intent (Enter for none)",
            list(INTENT_DESCRIPTIONS),
            allow_blank=True,
            default=row["secondary_intent"],
        )
        escalate = choose("Should a human handle this?", ["true", "false"], default=row["should_escalate"])
        reason_choices = list(ESCALATION_REASONS[1:] if escalate == "true" else ("none",))
        reason = choose("Escalation reason", reason_choices, default=row["escalation_reason"])
        required = input("Required reply points [Enter keeps draft]: ").strip() or row["required_reply_points"]
        forbidden = input("Forbidden claims [Enter keeps draft]: ").strip() or row["forbidden_claims"]
        notes = input("Annotation note [Enter keeps draft]: ").strip() or row["annotation_notes"]
        frame.loc[index, [
            "primary_intent", "secondary_intent", "should_escalate", "escalation_reason",
            "required_reply_points", "forbidden_claims", "annotator", "annotation_notes", "review_status",
        ]] = [intent, secondary, escalate, reason, required, forbidden, args.annotator, notes, "reviewed"]
        frame.to_csv(path, index=False)
        print("Saved. Ctrl-C is safe; resume with the same command.")


if __name__ == "__main__":
    main()
