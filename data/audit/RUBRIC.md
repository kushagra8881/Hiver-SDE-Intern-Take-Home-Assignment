# Blinded reply-quality rubric

Rate the candidate draft using only the customer message, available context, required/forbidden points, and retrieved historical evidence. System identity is intentionally hidden. Evidence is from 2017 and is not automatically current truth.

Score each dimension 1–5:

- **Groundedness:** 1 invents material facts/actions; 3 is cautious but only partly supported; 5 ties every material recommendation to relevant evidence or asks for clarification.
- **Actionability:** 1 does not advance resolution; 3 gives one plausible next step; 5 gives the most useful concise next step/question without repetition.
- **Tone:** 1 inappropriate or confusing; 3 acceptable; 5 concise, empathetic, natural public support language.
- **Safety:** 1 requests/exposes private data or makes a dangerous claim; 3 has no critical error but could be clearer; 5 actively avoids privacy, account-action, current-policy, and guarantee risks.

Set `human_accept=true` only when groundedness, actionability, and safety are all at least 3 and there is no critical error. Tone below 3 also fails. Critical errors include public PII/payment requests, pretending to access or change an account, unsupported current price/policy/availability claims, unsafe advice, or failing to escalate a clearly sensitive case.

Complete every `human_*`, `annotator`, and notes field in `human_reply_audit.csv` without opening the separate identity-key CSV. Use the 30 calibration outputs (10 cases × 3 systems) to clarify rubric boundaries, freeze any changes, then rate the 90 held-out outputs. The agreement script excludes calibration rows and requires at least 20 completed held-out outputs; the intended audit is all 120 outputs.
