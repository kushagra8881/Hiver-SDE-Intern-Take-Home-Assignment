# Trustworthy SpotifyCares Support Agent

This repository turns the 2.8M-row Customer Support on Twitter dataset into a scoped support agent for **SpotifyCares**. It classifies the customer’s intent, retrieves traceable historical cases, drafts a reply, and either auto-handles or escalates with an explicit reason.

The central design goal is selective trust, not maximum automation: account/security, money/refund, personal-data, legal/safety, missing-context, low-confidence, and weak-evidence cases go to a human.

> **Integrity checkpoint:** all 200 draft labels received an independent Gemini 3.5 Flash Lite review; 80 were changed and every row is stamped `llm_reviewed`. This is stronger development evidence, but it is not human labelling. Before presenting the set as “human-labelled,” the applicant must personally review every row with `scripts/label_gold.py`. Judge–human agreement likewise remains invalid until a person completes the blinded audit sheet.

The concise assignment report is in [REPORT.md](REPORT.md); the non-obvious choices are in [DECISIONS.md](DECISIONS.md).

## Reproduce the headline results in under 15 minutes

This path uses only committed artifacts: it does **not** need `twcs.csv` or a Gemini key. A clean-clone verification on the development machine completed `make reproduce` in 4.53 seconds (dependency download time depends on the network).

```bash
git clone https://github.com/kushagra8881/Hiver-SDE-Intern-Take-Home-Assignment.git
cd Hiver-SDE-Intern-Take-Home-Assignment

python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps

make reproduce
make test
```

Expected headline output for the main system: intent accuracy `0.710`, macro-F1 `0.633`, escalation recall `1.000`, auto coverage `0.200`, and unsafe-auto rate `0.000`. The frozen Gemini judge accepted 85% of the main replies, compared with 35% simple and 25% trivial. `results/judge_agreement.json` deliberately reports `insufficient_human_ratings` until the blinded sheet is rated by a person.

## Rebuild and run the full pipeline

Python 3.10+ is required. The raw Kaggle file is intentionally ignored by Git.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
```

Place `twcs.csv` in the repository root (or pass its path), then:

```bash
# Rebuild the 41,585 Spotify customer→brand cases and data manifest.
make prepare CSV=/absolute/path/to/twcs.csv

# Check the independently LLM-reviewed labels and regenerate local system results.
make validate-development
make test
make evaluate-development

# Run one deterministic, no-network example.
make demo MESSAGE="My playlist disappeared after the update"
```

`make evaluate-development` also creates the ignored model/retrieval artifacts required by `make demo`. A fresh clone can always run `make reproduce` immediately; running the interactive agent requires `twcs.csv` plus the rebuild commands above.

To repeat the independent review with an environment-only key, run `make review-labels-gemini`. After personal review with `scripts/label_gold.py`, `make validate && make evaluate` becomes the human-gold submission path.

On the supplied 8-core CPU environment, source extraction takes well under two minutes and local evaluation is designed to stay under 15 minutes. `make reproduce` recomputes result tables from committed gold/prediction/judge artifacts without the 493 MiB raw file or an API call.

### Optional Gemini drafting and judging

Rotate any API key that has ever been pasted into a chat. Put only the replacement in your shell environment or an ignored `.env`:

```bash
export GEMINI_API_KEY="..."
make evaluate-live
make audit
make judge
```

These live commands explicitly acknowledge that redacted public tweet text and selected evidence leave the local machine for Gemini. The application uses the key from the environment, never writes it to output, and never searches parent directories for `.env` files. The default drafting model can be changed with `GEMINI_MODEL`; the judge can use a distinct `GEMINI_JUDGE_MODEL`.

## What runs

```text
customer message
      │
      ├─ redact public PII + deterministic risk checks
      ├─ word/character TF–IDF intent classifier
      ├─ retrieve 3 thread-safe historical Spotify cases
      ├─ Gemini evidence-constrained draft (or deterministic fallback)
      └─ deterministic trust gate → AUTO_HANDLE or ESCALATE + reason
```

The generator does **not** decide routing. Historical replies are evidence, not current truth: shortened links, prices, plan rules, availability dates, and device-policy claims from 2017 are blocked from being repeated as current facts.

## Reproducibility paths

| Goal | Command | Raw CSV | API key |
|---|---|---:|---:|
| Recompute reported tables | `make reproduce` | No | No |
| Rebuild cases and local systems | `make prepare CSV=... && make evaluate` | Yes | No |
| Regenerate live replies | `make evaluate-live` | Prepared pairs | Yes |
| Score blinded replies | `make judge` | No | Yes |
| Measure judge agreement | `make agreement` | No | No |

Live regeneration is separated because model availability, rate limits, latency, and cost are not reproducible properties of a repository. Cached outputs must record their model and rubric version.

## Golden-set workflow

The committed sheet contains 200 distinct threads and near-duplicate groups. Its labels are independently LLM-reviewed (`results/gemini_label_review.csv`) pending the required personal human review:

- 150 cases are sampled from the latest 30% of Spotify cases to approximate prevalence.
- 50 challenge cases over-sample risky, short, ambiguous, and poorly answered messages.
- The earlier chronological half is development data; the later half is the one-shot test split.
- Retrieval uses only cases older than the earliest gold item and removes every gold thread and fingerprint.

See [data/gold/SAMPLING_AND_LABELING.md](data/gold/SAMPLING_AND_LABELING.md) for the exact protocol and [configs/intents.yaml](configs/intents.yaml) for frozen definitions.

## Evaluation

The same frozen test cases are used for:

- **Trivial:** development-set majority intent, fixed acknowledgement, always escalate.
- **Simple:** keyword intent, top-1 TF–IDF historical reply, fixed risk rules.
- **Main:** word+character TF–IDF classifier, diversified top-k evidence, constrained Gemini synthesis (or explicit fallback), deterministic trust gate.

Intent accuracy and macro-F1 are accompanied by per-class results. Routing reports escalation recall, unsafe-auto rate, and coverage. Reply scoring covers groundedness, actionability, tone, safety, and critical errors. The primary operational view is quality **at a stated auto-handle coverage**, never reply quality alone.

The blinded audit contains 40 cases × 3 systems: 10 cases calibrate the rubric and 30 cases (90 outputs) remain held out for agreement. Agreement reports bootstrap intervals, binary and weighted Cohen’s κ, Spearman correlation, within-one-point agreement, confusion matrices, and human-relative judge precision/recall. If held-out binary κ is below 0.60, the report treats the LLM judge as unreliable and uses human reply results as the headline.

## Repository map

```text
configs/                 frozen taxonomy and thresholds
data/gold/               200-case gold sheet + labelling protocol
data/audit/              blinded human reply audit + rubric
src/hiver_support/       extraction, retrieval, agent, policy, evaluation, judge
tests/                   offline unit/integration/adversarial checks
results/                 manifests and reproducible outputs
scripts/label_gold.py    resumable manual-review tool
REPORT.md                six-page-equivalent assignment report
DECISIONS.md             15-item decision log
```

## Scope boundaries

This is a public English-language triage-and-drafting prototype. It cannot authenticate users, inspect subscriptions, make refunds, confirm live outages, change an account, or know Spotify’s current policy. The historical dataset ends in 2017 and has no verified resolution or satisfaction labels. See the report’s mandatory “What is misleading about my headline number?” section before interpreting any score.

## Data and borrowed components

The raw dataset is [Customer Support on Twitter by Thought Vector](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter). Core modelling uses scikit-learn’s [TF–IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) and [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). Optional generation uses Google’s [Gemini generate-content API](https://ai.google.dev/gemini-api/docs/text-generation), structured JSON, and environment-based API keys. Agreement metrics use scikit-learn’s [Cohen’s kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html). No third-party prompt, labelled set, or support policy was copied.
