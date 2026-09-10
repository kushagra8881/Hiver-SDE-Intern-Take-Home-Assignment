from hiver_support.policy import decide_route
from hiver_support.taxonomy import message_risk
from hiver_support.agent import draft_safety_issues
from hiver_support.text import clean_historical_reply, redact_sensitive


def test_public_pii_is_redacted_and_escalated() -> None:
    text, flags = redact_sensitive("Email me at person@example.com; my card is 4111 1111 1111 1111")
    assert "person@example.com" not in text
    assert "4111" not in text
    assert set(flags) == {"email", "payment_card"}
    decision = decide_route(text, intent="billing_subscription", intent_confidence=0.9, top_similarity=0.8)
    assert decision.route == "ESCALATE"


def test_security_overrides_high_confidence() -> None:
    decision = decide_route("My account was hacked", intent="account_access", intent_confidence=0.99, top_similarity=0.99)
    assert decision.route == "ESCALATE"
    assert decision.reason == "account_or_security"


def test_low_risk_grounded_case_can_auto_handle() -> None:
    decision = decide_route(
        "How do I add songs to a playlist?",
        intent="playlist_library",
        intent_confidence=0.8,
        top_similarity=0.7,
    )
    assert decision.route == "AUTO_HANDLE"


def test_historical_reply_cleanup() -> None:
    cleaned = clean_historical_reply("@123 Hey! Try this https://t.co/example /AB")
    assert "@123" not in cleaned
    assert "/AB" not in cleaned
    assert "https://" not in cleaned


def test_generated_draft_guard_rejects_private_data_transition() -> None:
    assert "moves_to_private_channel" in draft_safety_issues("Please DM us your email address")
    assert "requests_sensitive_data" in draft_safety_issues("Please DM us your email address")
    assert draft_safety_issues("Which device and app version are you using?") == []


def test_risk_keywords_respect_word_boundaries() -> None:
    assert message_risk("I have the same issue again") == (False, "none")
    assert message_risk("I am bombarded with seasonal playlists") == (False, "none")
    assert message_risk("I will sue Spotify") == (True, "legal_or_policy")
    assert message_risk("There is a bomb") == (True, "abuse_or_safety")
