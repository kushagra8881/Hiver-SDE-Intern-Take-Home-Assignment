# Decision log

1. **SpotifyCares, not the largest brand.** Its 41,585 direct customer→brand cases are ample while the product domain stays more coherent than Amazon or Apple and less real-time/safety-critical than airlines or ride sharing.
2. **Public English Twitter support only.** Multilingual support, private authenticated channels, account actions, and live outage truth are deliberately out of scope.
3. **A connected reply component is the unit of splitting.** Adjacent turns from one conversation must never land on opposite sides of evaluation.
4. **Near-duplicate fingerprints are blocked across retrieval and gold.** Thread splitting alone does not stop templated repeated tweets from leaking.
5. **The taxonomy is frozen before final review.** Eight labels were chosen after inspecting Spotify cases; a secondary label is diagnostic, while metrics score one dominant need.
6. **Representative and challenge sampling stay separate.** The first estimates normal mix; the second exposes rare risks. Combining them would imply a false deployment prevalence.
7. **The test period is later than development and retrieval.** This is closer to a prospective support rollout than a random row split.
8. **Local lexical retrieval beats a downloaded embedding model here.** Word/character TF–IDF is inspectable, CPU-fast, reproducible offline, and unusually strong on short noisy tweets.
9. **Retrieved replies are evidence, not current policy.** 2017 URLs, prices, availability, and device claims can be stale; the generator is explicitly forbidden to restate them as current facts.
10. **Gemini synthesises; it does not own the safety decision.** A deterministic gate makes routing auditable and testable even if model output is malformed.
11. **Money, refunds, account security, PII, legal/safety, repeated failures, and missing context default to a human.** False automation costs much more than a conservative escalation.
12. **Confidence and evidence thresholds are selected on development data, then locked.** The later test set is never used to tune coverage.
13. **The headline is selective quality plus coverage.** Raw accuracy can be improved trivially by escalating everything or by counting acknowledgements.
14. **The judge is blinded and audited against a human.** Judge scores are not accepted as objective truth; below the pre-declared κ threshold they cannot be the headline.
15. **Offline reproduction and live regeneration are different commands.** The former is deterministic and under 15 minutes; the latter depends on external quota, pricing, and model availability.

