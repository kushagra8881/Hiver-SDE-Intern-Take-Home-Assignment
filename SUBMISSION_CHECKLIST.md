# Submission checklist

- [x] Verify no secret appears in tracked files or Git history.
- [ ] Rotate/revoke the API key shared during development.
- [ ] Review all 200 gold rows personally; replace assistant provenance with your initials.
- [x] Record the independent Gemini review: 80/200 draft labels changed at mean confidence 0.966.
- [ ] Record the applicant's human changes and adjudication decisions.
- [x] Run `make validate-development`, `make test`, and Ruff (13 tests; all checks pass).
- [ ] Run strict `make validate` after personal human review.
- [x] Run `make evaluate-live` with an environment-only key; live outputs are stamped `llm_reviewed`.
- [x] Run `make audit` and `make judge`; Gemini scored all 120 outputs with zero failed calls.
- [ ] Complete the blinded human sheet, then run `make agreement`.
- [x] Copy generated metrics into the frozen Results section of `REPORT.md` without cherry-picking.
- [x] Confirm representative and challenge strata are reported separately.
- [x] Run `make reproduce` and time it: 4.15 seconds on the development machine.
- [x] Inspect the top five failure examples and ensure every quoted ID exists in the artifact.
- [x] Initialise the Git repo; verify `twcs.csv`, `.env`, model files, and live secrets are not committed.
- [ ] Add a GitHub remote and push the repository.
- [ ] Grant Hiver access if the repository is private and submit only through the Notion form.
