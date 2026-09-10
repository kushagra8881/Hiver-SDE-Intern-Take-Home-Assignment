# Submission checklist

- [ ] Rotate/revoke the API key shared during development; verify no secret appears in Git history.
- [ ] Review all 200 gold rows personally; replace assistant provenance with your initials.
- [ ] Record how many draft labels you changed and any adjudication decisions.
- [ ] Run `make validate` and `make test`.
- [x] Run `make evaluate-live` with an environment-only key; live outputs are stamped `llm_reviewed`.
- [x] Run `make audit` and `make judge`; Gemini scored all 120 outputs with zero failed calls.
- [ ] Complete the blinded human sheet, then run `make agreement`.
- [x] Copy generated metrics into the frozen Results section of `REPORT.md` without cherry-picking.
- [ ] Confirm representative and challenge strata are reported separately.
- [ ] Run `make reproduce` in a clean environment and time it (<15 minutes).
- [ ] Inspect the top five failure examples and ensure every quoted ID exists in the artifact.
- [ ] Initialise/push the Git repo, verify `twcs.csv`, `.env`, model files, and live secrets are not committed.
- [ ] Grant Hiver access if the repository is private and submit only through the Notion form.
