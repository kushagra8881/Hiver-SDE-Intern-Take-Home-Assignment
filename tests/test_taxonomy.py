from hiver_support.taxonomy import keyword_intent


def test_high_precision_keyword_intents() -> None:
    assert keyword_intent("I was charged twice and want a refund")[0] == "billing_subscription"
    assert keyword_intent("My password reset failed and I cannot log in")[0] == "account_access"
    assert keyword_intent("My playlist vanished from the library")[0] == "playlist_library"

