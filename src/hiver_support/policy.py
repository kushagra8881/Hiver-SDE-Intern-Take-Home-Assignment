from __future__ import annotations

from dataclasses import dataclass

from .taxonomy import message_risk
from .text import redact_sensitive


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str
    explanation: str


SAFE_INTENTS = {
    "playback_technical",
    "playlist_library",
    "device_connectivity",
    "content_catalog",
    "product_feedback",
}


def decide_route(
    message: str,
    *,
    context: str = "",
    intent: str,
    intent_confidence: float,
    top_similarity: float,
    min_intent_confidence: float = 0.48,
    min_retrieval_similarity: float = 0.16,
) -> RouteDecision:
    _, pii_flags = redact_sensitive(f"{message} {context}")
    if pii_flags:
        return RouteDecision("ESCALATE", "privacy_or_pii", f"Public message contains possible {', '.join(pii_flags)} data.")

    risky, reason = message_risk(f"{message} {context}")
    if risky:
        explanations = {
            "account_or_security": "Authentication or account-security work requires identity verification.",
            "billing_or_refund": "Account-specific money, cancellation, or refund work requires a human.",
            "legal_or_policy": "Legal, rights, or policy claims need specialist review.",
            "abuse_or_safety": "Safety-sensitive language must be reviewed immediately.",
            "missing_context": "The current message is too dependent on missing thread context.",
        }
        return RouteDecision("ESCALATE", reason, explanations.get(reason, "A deterministic risk rule fired."))

    lowered = message.lower()
    stripped_words = [word for word in lowered.replace("@user", "").split() if word]
    if any(phrase in lowered for phrase in ("spotify is down", "service is down", "spotify down", "everyone is having")):
        return RouteDecision("ESCALATE", "weak_evidence", "A possible live outage requires current status data that is not available here.")
    if "family" in lowered and any(term in lowered for term in ("invite", "kicked", "undefined", "can't join", "cannot join")):
        return RouteDecision("ESCALATE", "billing_or_refund", "A Family-plan membership failure requires account-specific support.")
    if (
        any(term in lowered for term in ("someone accessing my account", "someone using my account", "stuff on it is not mine"))
        or ("someone" in lowered and "accessing my account" in lowered)
    ):
        return RouteDecision("ESCALATE", "account_or_security", "The message suggests unauthorised account access.")
    if any(symbol in message for symbol in ("$", "£", "€")) and any(term in lowered for term in ("offer", "rate", "price", "charged")):
        return RouteDecision("ESCALATE", "billing_or_refund", "A price, charge, or promotion claim needs current billing verification.")
    if lowered.replace("@user", "").strip().startswith("not working in "):
        return RouteDecision("ESCALATE", "missing_context", "The message reports failure but omits which feature is affected.")
    if len(stripped_words) <= 8 and any(lowered.replace("@user", "").strip().startswith(prefix) for prefix in ("yes ", "no ", "both ", "same ", "still ")):
        return RouteDecision("ESCALATE", "missing_context", "A short follow-up cannot be resolved without the earlier issue.")
    if context and any(term in lowered for term in ("android", "ios", "iphone", "spotify version")) and not any(
        term in lowered for term in ("?", "can't", "cannot", "not working", "error", "issue", "help", "why")
    ):
        return RouteDecision("ESCALATE", "missing_context", "The message supplies device details but the original symptom is missing.")
    if any(term in lowered for term in ("already tried", "still not", "doesn't help", "does not help", "tried everything")):
        return RouteDecision("ESCALATE", "missing_context", "The customer reports failed prior troubleshooting.")
    if intent in {"account_access", "billing_subscription"}:
        return RouteDecision("ESCALATE", "account_or_security" if intent == "account_access" else "billing_or_refund", "This intent commonly requires authenticated account access.")
    if intent not in SAFE_INTENTS or intent_confidence < min_intent_confidence:
        return RouteDecision("ESCALATE", "low_confidence", f"Intent confidence {intent_confidence:.2f} is below the trust gate.")
    if top_similarity < min_retrieval_similarity:
        return RouteDecision("ESCALATE", "weak_evidence", f"Best historical match {top_similarity:.2f} is too weak to ground an autonomous reply.")
    return RouteDecision("AUTO_HANDLE", "none", "Low-risk intent with sufficient classifier confidence and historical evidence.")
