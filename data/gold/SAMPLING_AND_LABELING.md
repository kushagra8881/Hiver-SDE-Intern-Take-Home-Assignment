# Golden set: sampling and labelling protocol

## Purpose

This set measures intent classification, escalation, and reply requirements on customer messages that received a historical SpotifyCares reply. A historical reply is context—not proof that the issue was resolved and not automatically the correct label.

## Sampling (frozen seed 1729)

1. Scan `twcs.csv` twice, preserve tweet IDs as strings, and select inbound tweets with a direct SpotifyCares child reply.
2. Join multiple direct brand reply fragments and connect reply-parent edges into thread components.
3. Normalise handles/URLs; form an exact normalised fingerprint; keep one case per component and fingerprint.
4. Restrict the candidate pool to the latest 30% by timestamp (2017-11-16 through 2017-12-03 in this run).
5. Uniformly sample 150 representative cases.
6. From the remainder, sample 50 challenge cases in frozen buckets: 8 billing/refund signals, 6 account/security, 4 safety/legal language, 10 short/missing-context, 8 ambiguous/multi-intent, 8 weak historical evidence, and 6 rare technical intents. Bucket membership is a sampling heuristic, not a gold label.
7. Split chronologically at the selected-set median: the earlier 100 rows are development and the later 100 are the untouched test set. Stratum sizes are reported separately.
8. When building retrieval, remove every gold component/fingerprint and keep only evidence older than the earliest gold timestamp.

The raw source hash and preprocessing version are written to `results/data_manifest.json`.

## Label fields

- `primary_intent`: exactly one frozen label from `configs/intents.yaml`.
- `secondary_intent`: optional genuinely separate need; never used to excuse a wrong primary prediction.
- `should_escalate`: whether the incoming message is safe and answerable without authenticated tools or live facts.
- `escalation_reason`: one enum. Use the first/highest-risk reason.
- `required_reply_points`: semicolon-delimited facts/questions/actions a good draft should contain.
- `forbidden_claims`: claims/actions that would make a reply unsafe or ungrounded.
- `annotation_notes`: the boundary decision, ambiguity, or missing context.
- `annotator` and `review_status`: provenance. Only a person who inspected the row may mark it `reviewed`.

## Escalation order

Use the first applicable reason:

1. `abuse_or_safety`
2. `legal_or_policy`
3. `privacy_or_pii`
4. `account_or_security`
5. `billing_or_refund`
6. `missing_context`
7. `low_confidence`
8. `weak_evidence`
9. `none` (auto-eligible)

Account-specific payments, refunds, cancellations, access, compromise, and personally identifying strings always escalate. General how-to plan questions may be auto-eligible only if they do not require current price/policy truth. Short follow-ups without sufficient preceding context escalate. A harsh tone alone is not an escalation reason unless safety/abuse is involved.

## Human-review procedure

Run:

```bash
PYTHONPATH=src python scripts/label_gold.py data/gold/spotify_golden_v1.csv --annotator YOUR_INITIALS
```

The assistant-drafted values are suggestions only. The reviewer must inspect every row, revise disagreements, and mark it reviewed. Record the number of changed rows in the report. Ideally, a second person independently labels 40–50 rows; report intent κ and escalation κ, adjudicate disagreements, and keep the raw audit.
