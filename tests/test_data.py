from pathlib import Path

import pandas as pd

from hiver_support.data import extract_brand_pairs, parse_id_list, without_gold_leakage


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_twcs.csv"


def test_parse_comma_separated_ids() -> None:
    assert parse_id_list("2, 7") == ["2", "7"]
    assert parse_id_list("") == []


def test_extract_stitches_replies_and_groups_thread(tmp_path: Path) -> None:
    output = tmp_path / "pairs.csv"
    frame = extract_brand_pairs(FIXTURE, output, brand="SpotifyCares", chunksize=2)
    assert set(frame["pair_id"]) == {"1", "3"}
    first = frame.set_index("pair_id").loc["1"]
    assert "Which device" in first["historical_reply"]
    assert "web player" in first["historical_reply"]
    assert first["reply_tweet_ids"] == "2,7"
    assert frame["conversation_id"].nunique() == 1
    assert "Which device" in frame.set_index("pair_id").loc["3", "context"]
    assert output.exists()


def test_gold_thread_and_near_duplicate_are_removed() -> None:
    pairs = pd.DataFrame(
        [
            {"pair_id": "1", "conversation_id": "A", "text_fingerprint": "x", "created_at": "Tue Oct 31 10:00:00 +0000 2017"},
            {"pair_id": "2", "conversation_id": "B", "text_fingerprint": "y", "created_at": "Tue Oct 30 10:00:00 +0000 2017"},
            {"pair_id": "3", "conversation_id": "C", "text_fingerprint": "x", "created_at": "Tue Oct 29 10:00:00 +0000 2017"},
            {"pair_id": "4", "conversation_id": "D", "text_fingerprint": "z", "created_at": "Tue Nov 01 10:00:00 +0000 2017"},
        ]
    )
    gold = pd.DataFrame(
        [{"conversation_id": "A", "text_fingerprint": "x", "created_at": "Tue Oct 31 12:00:00 +0000 2017"}]
    )
    clean = without_gold_leakage(pairs, gold)
    assert clean["pair_id"].tolist() == ["2"]
