from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .gemini import GeminiClient, GeminiError
from .models import IntentModel, RetrievalIndex, RetrievedCase
from .policy import decide_route
from .text import redact_sensitive


@dataclass(frozen=True)
class AgentResult:
    message: str
    intent: str
    intent_confidence: float
    route: str
    escalation_reason: str
    route_explanation: str
    draft_reply: str
    evidence: list[dict[str, Any]]
    generator: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fallback_draft(intent: str, evidence: list[RetrievedCase], route: str) -> str:
    if route == "ESCALATE":
        if intent in {"account_access", "billing_subscription"}:
            return "I’m sorry you’re dealing with this. A support specialist should review it securely—please don’t post account or payment details publicly."
        return "Thanks for flagging this. I don’t have enough verified context to resolve it safely here, so I’m escalating it to a support specialist."
    templates = {
        "playback_technical": "Sorry about the playback trouble. Which device, operating system, and app version are you using, and does it also happen on another network?",
        "playlist_library": "I can help with that. Which device are you using, and does the same playlist or library issue appear in the web player?",
        "device_connectivity": "Let’s narrow this down: which two devices are involved, and are they on the same network with the latest app version?",
        "content_catalog": "Thanks for reporting this. Please share the artist and release name (not account details) so the catalog issue can be checked.",
        "product_feedback": "Thanks for the thoughtful feedback. I’ve captured the use case clearly so the product team can review it.",
    }
    return templates.get(intent, "Thanks for reaching out. Could you share a little more detail about what you expected and what happened instead?")


def draft_safety_issues(draft: str) -> list[str]:
    lowered = draft.lower()
    issues: list[str] = []
    if re.search(r"\b(dm|direct message|private message)\b", lowered):
        issues.append("moves_to_private_channel")
    if any(term in lowered for term in ("email address", "username", "password", "card number", "account details", "login details")):
        issues.append("requests_sensitive_data")
    if any(term in lowered for term in ("we've refunded", "we have refunded", "we've cancelled", "we have cancelled", "we accessed your account", "we fixed your account")):
        issues.append("claims_unperformed_action")
    if "http://" in lowered or "https://" in lowered or "<historical-link>" in lowered:
        issues.append("unverified_link")
    return sorted(set(issues))


class SupportAgent:
    def __init__(
        self,
        intent_model: IntentModel,
        retrieval_index: RetrievalIndex,
        *,
        gemini_client: GeminiClient | None = None,
        top_k: int = 3,
        min_intent_confidence: float = 0.48,
        min_retrieval_similarity: float = 0.16,
    ) -> None:
        self.intent_model = intent_model
        self.retrieval_index = retrieval_index
        self.gemini_client = gemini_client
        self.top_k = top_k
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def handle(self, message: str, context: str = "", *, exclude_conversation: str | None = None) -> AgentResult:
        redacted_message, pii_flags = redact_sensitive(message)
        redacted_context, context_flags = redact_sensitive(context)
        intent, confidence, _ = self.intent_model.predict(redacted_message, redacted_context)
        evidence = self.retrieval_index.retrieve(
            redacted_message, intent=intent, k=self.top_k, exclude_conversation=exclude_conversation
        )
        top_similarity = evidence[0].similarity if evidence else 0.0
        route = decide_route(
            message,
            context=context,
            intent=intent,
            intent_confidence=confidence,
            top_similarity=top_similarity,
            min_intent_confidence=self.min_intent_confidence,
            min_retrieval_similarity=self.min_retrieval_similarity,
        )
        warnings = [f"redacted_{flag}" for flag in sorted(set(pii_flags + context_flags))]
        generator = "deterministic_fallback"
        draft = fallback_draft(intent, evidence, route.route)
        if self.gemini_client is not None:
            try:
                generated = self.gemini_client.draft(redacted_message, redacted_context, intent, evidence)
                draft = generated.draft_reply
                warnings.extend(generated.warnings)
                generator = f"gemini:{generated.model}"
                safety_issues = draft_safety_issues(draft)
                if safety_issues:
                    warnings.extend(f"rejected_draft:{issue}" for issue in safety_issues)
                    draft = fallback_draft(intent, evidence, route.route)
                    generator = f"gemini_rejected:{generated.model}->deterministic_fallback"
            except GeminiError as exc:
                warnings.append(f"gemini_fallback:{type(exc).__name__}")

        return AgentResult(
            message=redacted_message,
            intent=intent,
            intent_confidence=confidence,
            route=route.route,
            escalation_reason=route.reason,
            route_explanation=route.explanation,
            draft_reply=draft,
            evidence=[asdict(item) for item in evidence],
            generator=generator,
            warnings=warnings,
        )
