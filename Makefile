PYTHON ?= python
CSV ?= twcs.csv
PREDICTIONS ?= results/live_predictions.csv

.PHONY: install prepare review-labels-gemini validate validate-development evaluate evaluate-development evaluate-live audit judge agreement reproduce test demo

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

prepare:
	$(PYTHON) -m hiver_support.cli prepare --csv "$(CSV)"

review-labels-gemini:
	$(PYTHON) scripts/review_gold_with_gemini.py

validate:
	$(PYTHON) -m hiver_support.cli validate

validate-development:
	$(PYTHON) -m hiver_support.cli validate --allow-draft-labels

evaluate:
	$(PYTHON) -m hiver_support.cli evaluate

evaluate-development:
	$(PYTHON) -m hiver_support.cli evaluate --allow-draft-labels --predictions results/development_predictions.csv --metrics results/development_metrics.json

evaluate-live:
	$(PYTHON) -m hiver_support.cli evaluate --allow-draft-labels --with-gemini --acknowledge-external-data --predictions results/live_predictions.csv --metrics results/live_metrics.json

audit:
	$(PYTHON) -m hiver_support.cli make-audit --predictions "$(PREDICTIONS)"

judge:
	$(PYTHON) -m hiver_support.cli judge --acknowledge-external-data

agreement:
	$(PYTHON) -m hiver_support.cli agreement

reproduce:
	$(PYTHON) -m hiver_support.cli reproduce --predictions "$(PREDICTIONS)"

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) -m hiver_support.cli demo "$(MESSAGE)"
