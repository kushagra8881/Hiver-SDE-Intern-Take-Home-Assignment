import pandas as pd

from hiver_support.models import RetrievalIndex


def test_retrieval_excludes_requested_conversation() -> None:
    frame = pd.DataFrame(
        [
            {"pair_id": "1", "conversation_id": "thread-a", "message": "playlist songs missing", "historical_reply": "Check the playlist filter.", "reply_quality": 0.9, "weak_intent": "playlist_library", "text_fingerprint": "a"},
            {"pair_id": "2", "conversation_id": "thread-b", "message": "playlist track disappeared", "historical_reply": "Try checking the web player too.", "reply_quality": 0.8, "weak_intent": "playlist_library", "text_fingerprint": "b"},
            {"pair_id": "3", "conversation_id": "thread-c", "message": "speaker will not connect", "historical_reply": "Check that both devices share a network.", "reply_quality": 0.8, "weak_intent": "device_connectivity", "text_fingerprint": "c"},
        ]
    )
    index = RetrievalIndex(max_features=500).fit(frame, max_pairs=3)
    results = index.retrieve("playlist song missing", intent="playlist_library", k=2, exclude_conversation="thread-a")
    assert all(item.conversation_id != "thread-a" for item in results)
    assert results[0].pair_id == "2"

