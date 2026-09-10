from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .agent import SupportAgent
from .config import load_settings
from .data import extract_brand_pairs
from .evaluation import run_systems, score_predictions, write_metrics
from .gemini import GeminiClient
from .judge import build_audit_sheet, judge_human_agreement, judge_predictions
from .models import IntentModel, RetrievalIndex
from .sampling import sample_gold_candidates
from .validation import validate_gold


DEFAULT_CONFIG = "configs/spotify.yaml"
DEFAULT_PAIRS = "data/processed/spotify_pairs.csv"
DEFAULT_GOLD = "data/gold/spotify_golden_v1.csv"


def _settings(args: argparse.Namespace):
    return load_settings(args.config)


def command_prepare(args: argparse.Namespace) -> None:
    settings = _settings(args)
    output = Path(args.output)
    if output.exists() and not args.force:
        print(f"Using existing {output}; pass --force to rebuild")
        return
    frame = extract_brand_pairs(
        args.csv,
        output,
        brand=settings.brand,
        chunksize=settings.chunksize,
        manifest_path=args.manifest,
    )
    print(f"Prepared {len(frame):,} direct {settings.brand} support pairs -> {output}")


def command_sample(args: argparse.Namespace) -> None:
    settings = _settings(args)
    pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False)
    frame = sample_gold_candidates(pairs, args.output, size=settings.gold_size, seed=settings.random_seed)
    print(f"Wrote {len(frame)} unlabelled candidates -> {args.output}")
    print("Now label every row and set review_status=reviewed; do not evaluate draft labels.")


def command_validate(args: argparse.Namespace) -> None:
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False) if args.pairs and Path(args.pairs).exists() else None
    result = validate_gold(gold, pairs, require_human_review=not args.allow_draft_labels)
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


def command_evaluate(args: argparse.Namespace) -> None:
    settings = _settings(args)
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    pairs = pd.read_csv(args.pairs, dtype=str, keep_default_na=False)
    validation = validate_gold(gold, pairs, require_human_review=not args.allow_draft_labels)
    if not validation["valid"]:
        print(json.dumps(validation, indent=2), file=sys.stderr)
        raise SystemExit(1)
    if args.with_gemini and not args.acknowledge_external_data:
        raise SystemExit("--with-gemini sends redacted public tweet text to Gemini; rerun with --acknowledge-external-data")
    gemini = GeminiClient(model=settings.gemini_model) if args.with_gemini else None
    predictions, intent_model, index = run_systems(
        gold,
        pairs,
        seed=settings.random_seed,
        max_pairs=settings.retrieval_max_pairs,
        top_k=settings.retrieval_top_k,
        min_intent_confidence=settings.min_intent_confidence,
        min_retrieval_similarity=settings.min_retrieval_similarity,
        gemini_client=gemini,
        allow_unreviewed=args.allow_draft_labels,
    )
    predictions["label_provenance"] = "assistant_draft" if args.allow_draft_labels else "human_reviewed"
    Path(args.predictions).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.predictions, index=False)
    metrics = score_predictions(gold, predictions, seed=settings.random_seed)
    write_metrics(metrics, args.metrics)
    intent_model.save(args.intent_model)
    index.save(args.index)
    print(json.dumps(metrics, indent=2))


def command_make_audit(args: argparse.Namespace) -> None:
    settings = _settings(args)
    predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    audit = build_audit_sheet(
        predictions,
        gold,
        args.output,
        key_path=args.key,
        case_count=settings.judge_sample_size,
        seed=settings.random_seed,
    )
    print(f"Wrote {len(audit)} blinded system outputs -> {args.output}")
    print(f"Keep the identity key hidden during rating -> {args.key}")


def command_judge(args: argparse.Namespace) -> None:
    if not args.acknowledge_external_data:
        raise SystemExit("Judging sends redacted public tweet text and drafts to Gemini; rerun with --acknowledge-external-data")
    audit = pd.read_csv(args.audit, dtype=str, keep_default_na=False)
    judged = judge_predictions(audit, args.output, model=args.model)
    print(f"Judged {len(judged)} outputs -> {args.output}")


def command_agreement(args: argparse.Namespace) -> None:
    audit = pd.read_csv(args.audit, dtype=str, keep_default_na=False)
    judge = pd.read_csv(args.judge, dtype=str, keep_default_na=False)
    result = judge_human_agreement(audit, judge)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def command_reproduce(args: argparse.Namespace) -> None:
    settings = _settings(args)
    gold = pd.read_csv(args.gold, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(args.predictions, dtype=str, keep_default_na=False)
    result: dict[str, object] = {"prediction_metrics": score_predictions(gold, predictions, seed=settings.random_seed)}
    if Path(args.audit).exists() and Path(args.judge).exists():
        result["judge_human_agreement"] = judge_human_agreement(
            pd.read_csv(args.audit, dtype=str, keep_default_na=False),
            pd.read_csv(args.judge, dtype=str, keep_default_na=False),
        )
    print(json.dumps(result, indent=2))


def command_demo(args: argparse.Namespace) -> None:
    settings = _settings(args)
    model = IntentModel.load(args.intent_model)
    index = RetrievalIndex.load(args.index)
    gemini = GeminiClient(model=settings.gemini_model) if args.with_gemini else None
    agent = SupportAgent(
        model,
        index,
        gemini_client=gemini,
        top_k=settings.retrieval_top_k,
        min_intent_confidence=settings.min_intent_confidence,
        min_retrieval_similarity=settings.min_retrieval_similarity,
    )
    print(json.dumps(agent.handle(args.message, args.context).to_dict(), indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="SpotifyCares evidence-grounded support agent")
    root.add_argument("--config", default=DEFAULT_CONFIG)
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="extract thread-safe Spotify support pairs")
    prepare.add_argument("--csv", default="twcs.csv")
    prepare.add_argument("--output", default=DEFAULT_PAIRS)
    prepare.add_argument("--manifest", default="results/data_manifest.json")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare)

    sample = sub.add_parser("sample-gold", help="create an unlabelled 150+50 gold candidate sheet")
    sample.add_argument("--pairs", default=DEFAULT_PAIRS)
    sample.add_argument("--output", default="data/gold/candidates.csv")
    sample.set_defaults(func=command_sample)

    validate = sub.add_parser("validate", help="validate the frozen golden set")
    validate.add_argument("--gold", default=DEFAULT_GOLD)
    validate.add_argument("--pairs", default=DEFAULT_PAIRS)
    validate.add_argument("--allow-draft-labels", action="store_true")
    validate.set_defaults(func=command_validate)

    evaluate = sub.add_parser("evaluate", help="run baselines and main agent on the frozen test set")
    evaluate.add_argument("--pairs", default=DEFAULT_PAIRS)
    evaluate.add_argument("--gold", default=DEFAULT_GOLD)
    evaluate.add_argument("--predictions", default="results/predictions.csv")
    evaluate.add_argument("--metrics", default="results/metrics.json")
    evaluate.add_argument("--intent-model", default="artifacts/intent_model.joblib")
    evaluate.add_argument("--index", default="artifacts/retrieval_index.joblib")
    evaluate.add_argument("--with-gemini", action="store_true")
    evaluate.add_argument("--acknowledge-external-data", action="store_true")
    evaluate.add_argument(
        "--allow-draft-labels",
        action="store_true",
        help="development only: bypass the human-review gate and stamp outputs assistant_draft",
    )
    evaluate.set_defaults(func=command_evaluate)

    audit = sub.add_parser("make-audit", help="blind and shuffle reply outputs for human rating")
    audit.add_argument("--predictions", default="results/predictions.csv")
    audit.add_argument("--gold", default=DEFAULT_GOLD)
    audit.add_argument("--output", default="data/audit/human_reply_audit.csv")
    audit.add_argument("--key", default="data/audit/human_reply_audit_key.csv")
    audit.set_defaults(func=command_make_audit)

    judge = sub.add_parser("judge", help="score blinded replies with the frozen Gemini rubric")
    judge.add_argument("--audit", default="data/audit/human_reply_audit.csv")
    judge.add_argument("--output", default="results/judge_scores.csv")
    judge.add_argument("--model", default=None)
    judge.add_argument("--acknowledge-external-data", action="store_true")
    judge.set_defaults(func=command_judge)

    agreement = sub.add_parser("agreement", help="measure held-out human/judge agreement")
    agreement.add_argument("--audit", default="data/audit/human_reply_audit.csv")
    agreement.add_argument("--judge", default="results/judge_scores.csv")
    agreement.add_argument("--output", default="results/judge_agreement.json")
    agreement.set_defaults(func=command_agreement)

    reproduce = sub.add_parser("reproduce", help="recompute all metrics from committed artifacts without API calls")
    reproduce.add_argument("--gold", default=DEFAULT_GOLD)
    reproduce.add_argument("--predictions", default="results/predictions.csv")
    reproduce.add_argument("--audit", default="data/audit/human_reply_audit.csv")
    reproduce.add_argument("--judge", default="results/judge_scores.csv")
    reproduce.set_defaults(func=command_reproduce)

    demo = sub.add_parser("demo", help="run one message using built artifacts")
    demo.add_argument("message")
    demo.add_argument("--context", default="")
    demo.add_argument("--intent-model", default="artifacts/intent_model.joblib")
    demo.add_argument("--index", default="artifacts/retrieval_index.joblib")
    demo.add_argument("--with-gemini", action="store_true")
    demo.set_defaults(func=command_demo)
    return root


def main(argv: list[str] | None = None) -> None:
    # Deliberately do not search parent directories for unrelated project secrets.
    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    args = parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
