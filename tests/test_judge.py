from pathlib import Path

import pandas as pd

from hiver_support.judge import build_audit_sheet, judge_human_agreement


def test_audit_sheet_hides_system_identity_and_has_holdout(tmp_path: Path) -> None:
    gold_rows = []
    prediction_rows = []
    for index in range(1, 5):
        gold_id = f"G-{index}"
        gold_rows.append(
            {
                "gold_id": gold_id,
                "split": "test",
                "sample_stratum": "representative",
                "message": f"message {index}",
                "context": "",
                "required_reply_points": "ask a useful question",
                "forbidden_claims": "no account access",
                "should_escalate": "false",
                "primary_intent": "other",
            }
        )
        for system in ("trivial", "simple", "main"):
            prediction_rows.append(
                {
                    "gold_id": gold_id,
                    "system": system,
                    "message": f"message {index}",
                    "context": "",
                    "draft_reply": f"reply from {system}",
                    "evidence": "[]",
                }
            )
    output = tmp_path / "audit.csv"
    key = tmp_path / "key.csv"
    audit = build_audit_sheet(pd.DataFrame(prediction_rows), pd.DataFrame(gold_rows), output, key_path=key, case_count=4)
    assert "system" not in audit.columns
    assert set(audit["agreement_partition"]) == {"calibration", "holdout"}
    assert key.exists()


def test_empty_judge_results_report_insufficient() -> None:
    audit = pd.DataFrame(
        [{"audit_id": "A", "human_accept": "", "human_groundedness": "", "agreement_partition": "holdout"}]
    )
    judge = pd.DataFrame(columns=["audit_id", "judge_groundedness"])
    result = judge_human_agreement(audit, judge)
    assert result["status"] == "insufficient_human_ratings"
