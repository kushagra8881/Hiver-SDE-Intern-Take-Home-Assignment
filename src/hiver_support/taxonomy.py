from __future__ import annotations

import re
from collections import OrderedDict

from .text import normalise_text


INTENT_DESCRIPTIONS: "OrderedDict[str, str]" = OrderedDict(
    [
        ("account_access", "Login, password, Facebook sign-in, hacked or duplicate-account problems."),
        ("billing_subscription", "Payments, charges, Premium status, trials, refunds, cancellation, student/family plans."),
        ("playback_technical", "Playback, downloads, offline mode, crashes, slowness, audio quality or app/web-player faults."),
        ("playlist_library", "Playlists, saved songs, library organisation, queue, shuffle and sync of personal music."),
        ("device_connectivity", "Casting, speakers, consoles, cars, Bluetooth, device visibility and cross-device control."),
        ("content_catalog", "Missing/wrong songs, artists, albums, podcasts, metadata, availability or local files."),
        ("product_feedback", "Feature requests, UI feedback, ads, recommendations, social features or general complaints."),
        ("other", "Greetings, praise, unclear follow-ups, partnerships, artist/business requests and everything else."),
    ]
)

INTENTS = tuple(INTENT_DESCRIPTIONS)

KEYWORDS: dict[str, tuple[str, ...]] = {
    "account_access": (
        "log in", "login", "logged out", "sign in", "signin", "password", "account", "hacked",
        "facebook", "username", "email changed", "can't get in", "cannot get in",
    ),
    "billing_subscription": (
        "premium", "charged", "charge", "payment", "pay ", "paid", "billing", "refund", "cancel",
        "subscription", "student", "family plan", "trial", "receipt", "credit card", "discount",
    ),
    "playback_technical": (
        "won't play", "not playing", "stops playing", "playback", "offline", "download", "crash", "freez",
        "web player", "error", "buffer", "audio quality", "sound", "volume", "app won't", "not working",
    ),
    "playlist_library": (
        "playlist", "saved song", "my songs", "library", "queue", "shuffle", "repeat", "discover weekly",
        "release radar", "liked songs", "daily mix", "wrapped",
    ),
    "device_connectivity": (
        "bluetooth", "chromecast", "cast ", "speaker", "sonos", "alexa", "google home", "xbox", "playstation",
        "carplay", "android auto", "roku", "smart tv", "device", "connect", "ps4", "iphone x",
    ),
    "content_catalog": (
        "song missing", "missing song", "album", "artist", "track", "podcast", "lyrics", "local files",
        "not available", "unavailable", "wrong band", "metadata", "release", "music", "catalog",
    ),
    "product_feedback": (
        "feature", "please add", "wish", "should add", "feedback", "interface", "ui ", "update", "ads",
        "advert", "recommendation", "friend activity", "design", "why don't you", "please make",
    ),
    "other": (),
}


def keyword_scores(text: object) -> dict[str, float]:
    cleaned = normalise_text(text).lower()
    scores = {intent: 0.0 for intent in INTENTS}
    for intent, terms in KEYWORDS.items():
        for term in terms:
            if term in cleaned:
                scores[intent] += 1.0 + min(len(term), 16) / 32.0
    # A few high-precision disambiguators.
    if re.search(r"\b(charge[ds]?|refund|payment|billing)\b", cleaned):
        scores["billing_subscription"] += 2.0
    if re.search(r"\b(password|hacked|log(?:ged)?\s*(?:in|out))\b", cleaned):
        scores["account_access"] += 2.0
    if re.search(r"\b(playlist|queue|shuffle|library)\b", cleaned):
        scores["playlist_library"] += 1.5
    return scores


def keyword_intent(text: object) -> tuple[str, float]:
    scores = keyword_scores(text)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], INTENTS.index(item[0])))
    best, score = ranked[0]
    if score <= 0:
        return "other", 0.35
    second = ranked[1][1]
    confidence = 0.52 + min(0.38, 0.10 * score + 0.05 * max(0.0, score - second))
    return best, round(confidence, 4)


ESCALATION_REASONS = (
    "none",
    "account_or_security",
    "billing_or_refund",
    "privacy_or_pii",
    "legal_or_policy",
    "abuse_or_safety",
    "missing_context",
    "low_confidence",
    "weak_evidence",
)


def message_risk(text: object) -> tuple[bool, str]:
    cleaned = normalise_text(text).lower()
    patterns: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("abuse_or_safety", ("suicide", "kill myself", "self harm", "threat")),
        ("legal_or_policy", ("lawyer", "legal action", "lawsuit", "sue you", "sue spotify", "gdpr", "police", "copyright claim")),
        ("account_or_security", ("hacked", "stolen account", "account stolen", "email changed", "can't log in", "cannot log in", "password reset")),
        ("billing_or_refund", ("refund", "charged", "unauthorised", "unauthorized", "credit card", "payment", "money back", "cancel my")),
    )
    for reason, terms in patterns:
        if any(term in cleaned for term in terms):
            return True, reason
    if re.search(r"\bbomb\b", cleaned):
        return True, "abuse_or_safety"
    # Very short/context-only replies should not be autonomously answered.
    stripped = re.sub(r"@[A-Za-z0-9_]+|https?://\S+", "", cleaned).strip(" .!?,'\"")
    if len(stripped.split()) < 3 or stripped in {"yes", "no", "both", "same", "still", "help", "thanks"}:
        return True, "missing_context"
    return False, "none"
