from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from .taxonomy import keyword_intent, keyword_scores, message_risk
from .text import clean_historical_reply, public_text, response_quality


READ_COLUMNS = (
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
)


def _chunks(csv_path: str | Path, chunksize: int) -> Iterator[pd.DataFrame]:
    dtype = {name: "string" for name in READ_COLUMNS if name != "inbound"}
    yield from pd.read_csv(
        csv_path,
        usecols=list(READ_COLUMNS),
        dtype=dtype,
        chunksize=chunksize,
        keep_default_na=False,
        true_values=["True", "true", "1"],
        false_values=["False", "false", "0"],
    )


def parse_id_list(value: object) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
        node = item
        seen: set[str] = set()
        while self.parent[node] != node and node not in seen:
            seen.add(node)
            node = self.parent[node]
        root = node
        for visited in seen:
            self.parent[visited] = root
        return root

    def union(self, left: str, right: str) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            # Numeric tweet IDs make this deterministic; lexical fallback handles fixtures.
            def sort_key(value: str) -> tuple[int, int | str]:
                return (0, int(value)) if value.isdigit() else (1, value)

            root, child = sorted((lroot, rroot), key=sort_key)
            self.parent[child] = root


def _fingerprint(text: str) -> str:
    tokens = " ".join(
        token for token in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
        if token not in {"user", "url"}
    )
    return hashlib.sha1(tokens.encode("utf-8")).hexdigest()[:16]


def _sha256(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def extract_brand_pairs(
    csv_path: str | Path,
    output_path: str | Path,
    *,
    brand: str,
    chunksize: int = 250_000,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Two-pass extraction of inbound tweets that received a direct brand reply.

    IDs remain strings. Multiple direct brand replies are stitched. Connected reply
    components become conversation IDs so train/eval splits cannot share a thread.
    """
    replies_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    brand_reply_by_id: dict[str, dict[str, str]] = {}

    for chunk in _chunks(csv_path, chunksize):
        selected = chunk[(chunk["author_id"] == brand) & (~chunk["inbound"].astype(bool))]
        for row in selected.itertuples(index=False):
            parent = str(row.in_response_to_tweet_id).strip()
            tweet_id = str(row.tweet_id).strip()
            if not parent or not tweet_id:
                continue
            record = {
                "tweet_id": tweet_id,
                "parent_id": parent,
                "created_at": str(row.created_at),
                "text": str(row.text),
            }
            replies_by_parent[parent].append(record)
            brand_reply_by_id[tweet_id] = record

    wanted = set(replies_by_parent)
    inbound_rows: dict[str, dict[str, str]] = {}
    for chunk in _chunks(csv_path, chunksize):
        selected = chunk[chunk["tweet_id"].isin(wanted) & chunk["inbound"].astype(bool)]
        for row in selected.itertuples(index=False):
            tweet_id = str(row.tweet_id).strip()
            inbound_rows[tweet_id] = {
                "tweet_id": tweet_id,
                "created_at": str(row.created_at),
                "text": str(row.text),
                "parent_id": str(row.in_response_to_tweet_id).strip(),
                "response_ids": str(row.response_tweet_id).strip(),
            }

    graph = DisjointSet()
    for parent_id, replies in replies_by_parent.items():
        if parent_id not in inbound_rows:
            continue
        for reply in replies:
            graph.union(parent_id, reply["tweet_id"])
        previous = inbound_rows[parent_id]["parent_id"]
        if previous and previous in brand_reply_by_id:
            graph.union(parent_id, previous)

    rows: list[dict[str, object]] = []
    for inbound_id, inbound in inbound_rows.items():
        replies = sorted(
            replies_by_parent[inbound_id],
            key=lambda item: (item["created_at"], int(item["tweet_id"]) if item["tweet_id"].isdigit() else item["tweet_id"]),
        )
        raw_reply = " ".join(item["text"] for item in replies)
        reply = " ".join(clean_historical_reply(item["text"]) for item in replies).strip()
        message = public_text(inbound["text"])
        previous_reply = brand_reply_by_id.get(inbound["parent_id"], {}).get("text", "")
        context = clean_historical_reply(previous_reply) if previous_reply else ""
        intent, weak_confidence = keyword_intent(message)
        scores = sorted(keyword_scores(message).values(), reverse=True)
        ambiguity = 1.0 if scores[0] == scores[1] else max(0.0, 1.0 - (scores[0] - scores[1]) / max(scores[0], 1.0))
        risk, risk_reason = message_risk(message)
        rows.append(
            {
                "pair_id": inbound_id,
                "conversation_id": graph.find(inbound_id),
                "created_at": inbound["created_at"],
                "message": message,
                "context": context,
                "historical_reply": reply,
                "reply_tweet_ids": ",".join(item["tweet_id"] for item in replies),
                "text_fingerprint": _fingerprint(message),
                "weak_intent": intent,
                "weak_intent_confidence": weak_confidence,
                "weak_ambiguity": round(ambiguity, 4),
                "weak_risk": risk,
                "weak_risk_reason": risk_reason,
                "reply_quality": response_quality(raw_reply),
            }
        )

    frame = pd.DataFrame(rows).sort_values("pair_id", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    if manifest_path:
        manifest = {
            "source_file": Path(csv_path).name,
            "source_sha256": _sha256(csv_path),
            "brand": brand,
            "preprocessing_version": "1.0.0",
            "rows": len(frame),
            "conversation_count": int(frame["conversation_id"].nunique()),
            "fingerprint_count": int(frame["text_fingerprint"].nunique()),
            "columns": list(frame.columns),
        }
        target = Path(manifest_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return frame


def stable_bucket(value: object, seed: int, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def without_gold_leakage(pairs: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    blocked_threads = set(gold["conversation_id"].astype(str))
    blocked_fingerprints = set(gold["text_fingerprint"].astype(str))
    clean = pairs[
        ~pairs["conversation_id"].astype(str).isin(blocked_threads)
        & ~pairs["text_fingerprint"].astype(str).isin(blocked_fingerprints)
    ].copy()
    # Gold is sampled from the latest period. Only evidence that existed before
    # the earliest gold case is eligible, which makes the evaluation prospective.
    if "created_at" in gold and "created_at" in clean:
        date_format = "%a %b %d %H:%M:%S %z %Y"
        gold_times = pd.to_datetime(gold["created_at"], format=date_format, utc=True, errors="coerce")
        pair_times = pd.to_datetime(clean["created_at"], format=date_format, utc=True, errors="coerce")
        if gold_times.notna().any():
            clean = clean[pair_times < gold_times.min()].copy()
    return clean
