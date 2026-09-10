from __future__ import annotations

import html
import re


HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{1,15}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,14}\d(?!\d)")
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
AGENT_SIGNATURE_RE = re.compile(r"\s*/[A-Z]{2,3}\s*$")
SPACE_RE = re.compile(r"\s+")


def normalise_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text).replace("\u200b", " ")
    return SPACE_RE.sub(" ", text).strip()


def public_text(value: object) -> str:
    """Normalise a tweet while replacing handles/links that invite memorisation."""
    text = normalise_text(value)
    text = HANDLE_RE.sub("@user", text)
    text = URL_RE.sub("<URL>", text)
    return SPACE_RE.sub(" ", text).strip()


def redact_sensitive(value: object) -> tuple[str, list[str]]:
    """Redact high-risk strings before retrieval, logging, or model calls."""
    text = public_text(value)
    flags: list[str] = []
    for name, pattern, replacement in (
        ("email", EMAIL_RE, "<EMAIL>"),
        ("payment_card", CARD_RE, "<PAYMENT_CARD>"),
        ("phone", PHONE_RE, "<PHONE>"),
    ):
        if pattern.search(text):
            flags.append(name)
            text = pattern.sub(replacement, text)
    return text, flags


def clean_historical_reply(value: object) -> str:
    text = normalise_text(value)
    text = re.sub(r"^(?:@[A-Za-z0-9_]{1,15}\s*)+", "", text)
    text = AGENT_SIGNATURE_RE.sub("", text)
    text = URL_RE.sub("<historical-link>", text)
    text = HANDLE_RE.sub("@user", text)
    return SPACE_RE.sub(" ", text).strip()


def response_quality(value: object) -> float:
    """Cheap quality prior: concrete historical replies outrank DM-only replies."""
    text = clean_historical_reply(value).lower()
    if not text:
        return 0.0
    score = min(len(text) / 180.0, 1.0) * 0.35
    concrete = (
        "try ", "go to ", "select ", "settings", "reinstall", "restart",
        "available", "because", "follow", "tap ", "click ", "check ",
        "can you", "could you", "what device", "which device",
    )
    score += min(sum(term in text for term in concrete), 3) * 0.18
    if "dm" in text or "private message" in text:
        score -= 0.12
    if len(text.split()) < 5:
        score -= 0.20
    return round(max(0.0, min(score, 1.0)), 4)
