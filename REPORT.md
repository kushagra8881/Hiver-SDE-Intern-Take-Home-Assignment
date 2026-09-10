# SpotifyCares support agent — evaluation report

**Scope:** public English-language Twitter support triage and drafting.  
**Data:** Customer Support on Twitter, SpotifyCares subset, through 2017-12-03.  
**Reproducibility:** `make reproduce` recomputes tables from frozen artifacts; full rebuild uses `make prepare CSV=...`.

> **Submission status:** engineering and assistant-drafted annotations are complete. Numerical development results are explicitly provisional until the applicant personally reviews all 200 gold rows and completes the blinded reply audit. This report does not relabel AI output as human evidence.

## 1. Problem framing: what “good” means

For SpotifyCares, a useful prototype must do three things at once: identify the immediate support need, propose a concise next step that is traceable to similar historical handling, and avoid autonomously handling messages that need authentication, current operational truth, or human judgement. The trust target is therefore **high end-to-end correctness among auto-handled cases at useful coverage**, not maximum raw accuracy.

An auto-handled item counts as successful only when its primary intent is correct, the gold label says it is auto-eligible, and its reply passes groundedness, actionability, tone, and safety. Results always pair that selective-success rate with auto-handle coverage. Routing separately reports escalation recall and `unsafe_auto_rate = should-escalate cases sent to AUTO / all AUTO cases`.

I did **not** build account authentication, refunds/cancellations, live outage detection, a current Spotify policy knowledge base, multilingual handling, or direct Twitter posting. The system drafts; it cannot verify resolution. These omissions are deliberate because the 2017 public-tweet dataset does not support those claims.

## 2. Data, taxonomy, and gold construction

The raw file has 2,811,774 logical tweets. SpotifyCares has 43,265 outbound tweets and 41,585 unique inbound customer tweets with a valid direct Spotify reply, across roughly 28,280 connected threads. Half of historical Spotify replies contain a URL and roughly one quarter request a DM, which makes naïve reply copying especially risky.

I reconstruct reply components from parent links, stitch multiple direct brand reply fragments, retain IDs as strings, normalise handles/URLs, and redact email, phone, and card-like patterns. The eight frozen primary intents are: account access; billing/subscription; playback/technical; playlist/library; device/connectivity; content/catalog; product feedback; and other. Definitions and boundary examples live in `configs/intents.yaml`.

The 200-row gold design has two non-combinable strata:

- **150 representative:** uniform cases from the latest 30% of eligible Spotify cases.
- **50 challenge:** cases deliberately enriched for risk, short/missing context, ambiguity, and weak historical handling.

One thread and one normalised fingerprint can appear only once. The earlier chronological half is development; the later half is the untouched test set. Retrieval excludes all gold threads/fingerprints and anything dated on or after the earliest gold item. The sample/label protocol and provenance fields are committed beside the CSV.

## 3. System and baselines

The main pipeline first redacts public PII, predicts one intent with word + character TF–IDF logistic regression, then retrieves three diverse historical cases using character TF–IDF with a small reply-quality and intent-match reranking bonus. Gemini can synthesise a 280-character reply from the delimited evidence under a structured JSON contract. A no-network deterministic fallback remains usable and is labelled as such.

The language model does not control routing. A deterministic trust gate escalates PII, security/access, account-specific billing/refund, legal/safety, missing-context, repeated-failure, low-confidence, and weak-evidence cases. Even high-confidence account/billing predictions are escalated because the prototype has no authenticated tools.

Comparisons use identical frozen test cases:

- **Trivial:** development-majority intent, fixed acknowledgement, always human.
- **Simple:** keyword intent, copied top-1 cleaned historical reply, same deterministic risk rules.
- **Main:** learned intent, diversified evidence, constrained synthesis/fallback, locked trust gate.

## 4. Results and judge reliability

### Current development evidence

`results/development_metrics.json` is generated from assistant-drafted labels and is useful for debugging only. It is **not** a human-gold headline. On this provisional 100-case chronological test half, the main system reached 0.616 macro-F1 and safely auto-routed 20 cases; no gold-escalate case was auto-routed. With only 20 auto cases, zero observed unsafe autos still permits a rough 95% upper bound near 15% (the rule of three). After the human checkpoint, `make evaluate` replaces these development artifacts with frozen results.

<!-- RESULTS_TABLE_START -->

| System | Intent accuracy (95% CI) | Macro-F1 | Escalation recall | Auto coverage | Unsafe-auto rate |
|---|---:|---:|---:|---:|---:|
| Trivial | 0.210 (0.130–0.290) | 0.043 | 1.000 | 0.000 | 0.000 |
| Simple | 0.700 (0.600–0.790) | 0.612 | 1.000 | 0.200 | 0.000 |
| Main | 0.690 (0.600–0.780) | 0.616 | 1.000 | 0.200 | 0.000 |

<!-- RESULTS_TABLE_END -->

These are assistant-draft development numbers. The simple baseline has one-point higher accuracy; the main hybrid has slightly higher macro-F1, mainly on the representative stratum. Neither difference is meaningful at this sample size. The honest engineering result is that the trust gate improves on always-escalate by reaching 20% coverage without an observed unsafe route—not that the learned classifier decisively beats keywords.

| Main-system stratum | Cases | Intent accuracy | Macro-F1 | Escalation recall | Auto coverage | Unsafe-auto rate |
|---|---:|---:|---:|---:|---:|---:|
| Representative | 77 | 0.688 | 0.633 | 1.000 | 0.208 | 0.000 |
| Challenge | 23 | 0.696 | 0.464 | 1.000 | 0.174 | 0.000 |

The challenge mix is intentionally not a deployment estimate. Thread-cluster bootstrap 95% confidence intervals accompany overall intent accuracy. Per-intent scores and route confusion matrices are in the JSON artifact. Device/connectivity has only two test examples and playlist/library only three, so their apparent per-class performance is especially unstable.

### Reply-quality judge

The frozen rubric scores groundedness, actionability, tone, and safety from 1–5; critical privacy, account-action, current-fact, or unsafe claims force rejection. System names are hidden and rows are shuffled. A person rates 40 cases × 3 systems before looking at judge output: 10 cases calibrate the rubric and 30 cases (90 outputs) remain held out. The agreement harness reports bootstrap intervals, binary Cohen’s κ, human-relative precision/recall/F1, quadratic-weighted κ by dimension, Spearman correlation, within-one-point agreement, and confusion matrices.

The pre-declared credibility rule is: if held-out binary κ < 0.60, LLM pass rate cannot be the headline. Same-vendor drafting/judging is disclosed as a remaining bias even when measured agreement is acceptable.

**Judge–human agreement:** pending human completion of `data/audit/human_reply_audit.csv`; the code returns `insufficient_human_ratings` rather than fabricating a statistic.

## 5. Failure analysis

The following five modes are taken from actual assistant-draft test predictions; IDs map directly to committed gold/prediction rows:

1. **Figurative risk language causes unnecessary escalation — SPOT-165 / tweet 2909826.** “Committing suicide” describes playlist strategy, but the safety keyword gate escalated it. Hypothesis: high-recall safety rules need phrase-level context or a second-stage safety classifier. Keep the conservative default, but measure false escalation and require human review of rule edits.
2. **Product feedback is confused with the object mentioned — SPOT-007 / tweet 83673.** A complaint about the weekly-playlist algorithm was predicted `playlist_library` instead of `product_feedback`. Hypothesis: short lexical features overweight “playlist”; add intent-boundary examples and retrieval-derived contrastive features.
3. **Catalog requests look like personal-library operations — SPOT-196 / tweet 2981117.** “Get ‘Officially Yours’ added to the library” was predicted `playlist_library`, but the request is catalog availability. Hypothesis: distinguish named public works/artists from possessive personal-library language.
4. **Closings are safe but confidence-gated — SPOT-009 / tweet 513059.** “That link worked! Thanks!” was correctly classified `other` but escalated for low confidence. Similar false escalations occur on SPOT-015, SPOT-034, SPOT-140, SPOT-170, and SPOT-193. Hypothesis: a separate high-precision acknowledgement detector could increase coverage without exposing support risk; report it separately so it does not inflate substantive capability.
5. **Conservative billing rules reject general questions — SPOT-158 / tweet 44384.** A general promo payment-method question is gold auto-eligible, but the router escalated on credit-card language. Hypothesis: separate general policy navigation from account-specific charges/refunds, backed by a current official knowledge base rather than 2017 replies.

The system still correctly escalated the highest-cost intent mistakes: SPOT-040’s Family-plan failure and SPOT-076’s unauthorised access were misclassified, but independent risk rules caught them. This is why risk flags remain separate from the single-label intent taxonomy.

## 6. What is misleading about my headline number?

First, abstention can make auto-handled precision look excellent by shifting work to people. That is why every selective-quality score is paired with coverage and false-escalation load. An always-escalate system is safe but useless; an always-auto system looks productive but hides risk.

Second, 150 representative examples yield wide confidence intervals and may contain no instance of a rare severe failure. Zero observed unsafe replies is not zero deployment risk. The 50-case challenge set improves discovery but is deliberately prevalence-distorted, so combining it with the representative sample would also mislead.

Third, the dataset records replies, not resolutions or satisfaction. Brand history can be incomplete, templated, mistaken, or stale. Twitter language and Spotify policy from 2017 are not today’s private support traffic. Half the historical replies include links that may no longer be valid.

Fourth, reply quality is subjective and model-mediated. An LLM judge can share stylistic preferences and vendor bias with the generator. Agreement with a blinded human audit limits—but does not remove—that problem. If κ misses the locked threshold, the judge result is demoted.

Fifth, short acknowledgements and repeated templates can inflate aggregate scores. Thread and fingerprint isolation reduce leakage but do not prove semantic independence. Finally, the system only drafts a next step; it cannot access an account, execute a fix, verify policy, or observe downstream resolution.

### One more week

I would (1) obtain a second independent annotator and adjudicate a larger rare-risk set; (2) replace stale tweet facts with a maintained official help-centre/status knowledge base and authenticated tool contracts; (3) add an embedding/cross-encoder reranker and measure incremental retrieval value; (4) calibrate risk–coverage thresholds on more data; (5) red-team prompt injection, PII, repeated-failure, outage, and multilingual cases; and (6) shadow the agent with support staff, measuring overrides, time saved, reopen rate, and actual resolution rather than offline prose alone.
