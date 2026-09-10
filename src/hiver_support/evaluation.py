from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .agent import SupportAgent
from .models import IntentModel, RetrievalIndex
from .policy import decide_route
from .taxonomy import INTENTS, keyword_intent


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "escalate"}


def _simple_reply(evidence: list[Any]) -> str:
    if not evidence:
        return "Thanks for reaching out. Could you share a little more detail?"
    reply = evidence[0].reply.replace("<historical-link>", "").strip()
    return reply[:280] if reply else "Thanks for reaching out. Could you share a little more detail?"


def _calibrate_intent_threshold(
    dev: pd.DataFrame,
    index: RetrievalIndex,
    *,
    seed: int,
    min_retrieval_similarity: float,
    fallback: float,
) -> float:
    """Choose coverage on a temporal dev tail subject to a conservative risk bound."""
    dates = pd.to_datetime(
        dev["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True, errors="coerce"
    )
    ordered = dev.assign(_date=dates).sort_values(["_date", "gold_id"])
    cut = max(2, int(len(ordered) * 0.70))
    train, tune = ordered.iloc[:cut], ordered.iloc[cut:]
    if tune.empty or train["primary_intent"].value_counts().min() < 2:
        return fallback
    model = IntentModel(seed=seed).fit(train)
    cases: list[tuple[Any, str, float, float]] = []
    for row in tune.itertuples(index=False):
        intent, confidence, _ = model.predict(row.message, row.context)
        evidence = index.retrieve(row.message, intent=intent, k=1, exclude_conversation=str(row.conversation_id))
        similarity = evidence[0].similarity if evidence else 0.0
        cases.append((row, intent, confidence, similarity))
    candidates = sorted(set(np.round(np.linspace(0.18, 0.55, 16), 3).tolist() + [fallback]))
    feasible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        routes = [
            decide_route(
                row.message,
                context=row.context,
                intent=intent,
                intent_confidence=confidence,
                top_similarity=similarity,
                min_intent_confidence=threshold,
                min_retrieval_similarity=min_retrieval_similarity,
            ).route
            for row, intent, confidence, similarity in cases
        ]
        auto = np.asarray(routes) == "AUTO_HANDLE"
        truth = np.asarray([parse_bool(row.should_escalate) for row, *_ in cases])
        unsafe_rate = float((auto & truth).sum() / max(int(auto.sum()), 1))
        escalation_recall = float(((~auto) & truth).sum() / max(int(truth.sum()), 1))
        coverage = float(auto.mean())
        if unsafe_rate <= 0.10 and escalation_recall >= 0.90:
            feasible.append((coverage, -unsafe_rate, threshold))
    return round(max(feasible)[2], 4) if feasible else fallback


def run_systems(
    gold: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    seed: int,
    max_pairs: int,
    top_k: int,
    min_intent_confidence: float,
    min_retrieval_similarity: float,
    gemini_client: Any | None = None,
    allow_unreviewed: bool = False,
) -> tuple[pd.DataFrame, IntentModel, RetrievalIndex]:
    from .data import without_gold_leakage

    dev = gold[gold["split"] == "dev"].copy()
    test = gold[gold["split"] == "test"].copy()
    if dev.empty or test.empty:
        raise ValueError("Golden set needs both dev and test rows")
    if not allow_unreviewed and gold["review_status"].astype(str).str.lower().ne("reviewed").any():
        raise ValueError("Every golden row must be marked review_status=reviewed")

    retrieval_frame = without_gold_leakage(pairs, gold)
    index = RetrievalIndex().fit(retrieval_frame, max_pairs=max_pairs, seed=seed)
    calibrated_threshold = _calibrate_intent_threshold(
        dev,
        index,
        seed=seed,
        min_retrieval_similarity=min_retrieval_similarity,
        fallback=min_intent_confidence,
    )
    intent_model = IntentModel(seed=seed).fit(dev)
    agent = SupportAgent(
        intent_model,
        index,
        gemini_client=gemini_client,
        top_k=top_k,
        min_intent_confidence=calibrated_threshold,
        min_retrieval_similarity=min_retrieval_similarity,
    )
    majority = dev["primary_intent"].value_counts().idxmax()
    rows: list[dict[str, Any]] = []
    for case in test.itertuples(index=False):
        base = {
            "gold_id": case.gold_id,
            "conversation_id": str(case.conversation_id),
            "sample_stratum": case.sample_stratum,
            "message": case.message,
            "context": case.context,
        }
        rows.append(
            {
                **base,
                "system": "trivial",
                "predicted_intent": majority,
                "intent_confidence": float(dev["primary_intent"].value_counts(normalize=True).max()),
                "predicted_route": "ESCALATE",
                "escalation_reason": "always_human_baseline",
                "draft_reply": "Thanks for reaching out. A support specialist will take a look.",
                "evidence": "[]",
                "generator": "fixed",
                "routing_threshold": None,
            }
        )

        simple_intent, simple_confidence = keyword_intent(case.message)
        simple_evidence = index.retrieve(case.message, intent=simple_intent, k=1, exclude_conversation=str(case.conversation_id))
        simple_similarity = simple_evidence[0].similarity if simple_evidence else 0.0
        simple_route = decide_route(
            case.message,
            context=case.context,
            intent=simple_intent,
            intent_confidence=simple_confidence,
            top_similarity=simple_similarity,
            min_intent_confidence=min_intent_confidence,
            min_retrieval_similarity=min_retrieval_similarity,
        )
        rows.append(
            {
                **base,
                "system": "simple",
                "predicted_intent": simple_intent,
                "intent_confidence": simple_confidence,
                "predicted_route": simple_route.route,
                "escalation_reason": simple_route.reason,
                "draft_reply": _simple_reply(simple_evidence),
                "evidence": json.dumps([asdict(item) for item in simple_evidence], ensure_ascii=False),
                "generator": "top1_copy",
                "routing_threshold": min_intent_confidence,
            }
        )

        result = agent.handle(case.message, case.context, exclude_conversation=str(case.conversation_id))
        rows.append(
            {
                **base,
                "system": "main",
                "predicted_intent": result.intent,
                "intent_confidence": result.intent_confidence,
                "predicted_route": result.route,
                "escalation_reason": result.escalation_reason,
                "draft_reply": result.draft_reply,
                "evidence": json.dumps(result.evidence, ensure_ascii=False),
                "generator": result.generator,
                "routing_threshold": calibrated_threshold,
            }
        )
    return pd.DataFrame(rows), intent_model, index


def _bootstrap_interval(
    merged: pd.DataFrame,
    value_fn: Any,
    *,
    seed: int = 1729,
    iterations: int = 1_000,
) -> list[float]:
    groups = merged["conversation_id"].astype(str).unique()
    if len(groups) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    group_values = merged["conversation_id"].astype(str).to_numpy()
    group_rows = [np.flatnonzero(group_values == group) for group in groups]
    values: list[float] = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(groups), size=len(groups))
        row_indices = np.concatenate([group_rows[index] for index in sampled])
        sample = merged.iloc[row_indices]
        values.append(float(value_fn(sample)))
    low, high = np.quantile(values, [0.025, 0.975])
    return [round(float(low), 4), round(float(high), 4)]


def score_predictions(gold: pd.DataFrame, predictions: pd.DataFrame, *, seed: int = 1729) -> dict[str, Any]:
    truth_columns = ["gold_id", "conversation_id", "sample_stratum", "primary_intent", "should_escalate"]
    truth = gold[gold["split"] == "test"][truth_columns].copy()
    truth["should_escalate"] = truth["should_escalate"].map(parse_bool)
    report: dict[str, Any] = {"test_cases": int(len(truth)), "systems": {}}
    for system, frame in predictions.groupby("system", sort=False):
        merged = frame.merge(truth, on=["gold_id", "conversation_id", "sample_stratum"], validate="one_to_one")
        predicted_escalate = merged["predicted_route"].eq("ESCALATE")
        intent_accuracy = accuracy_score(merged["primary_intent"], merged["predicted_intent"])
        macro_f1 = f1_score(merged["primary_intent"], merged["predicted_intent"], labels=list(INTENTS), average="macro", zero_division=0)
        precision, recall, route_f1, _ = precision_recall_fscore_support(
            merged["should_escalate"], predicted_escalate, average="binary", zero_division=0
        )
        auto = ~predicted_escalate
        unsafe_auto = auto & merged["should_escalate"]
        unsafe_auto_rate = float(unsafe_auto.sum() / max(int(auto.sum()), 1))
        system_report: dict[str, Any] = {
            "intent_accuracy": round(float(intent_accuracy), 4),
            "intent_accuracy_95ci": _bootstrap_interval(merged, lambda x: accuracy_score(x["primary_intent"], x["predicted_intent"]), seed=seed),
            "intent_macro_f1": round(float(macro_f1), 4),
            "intent_by_class": classification_report(
                merged["primary_intent"], merged["predicted_intent"], labels=list(INTENTS), output_dict=True, zero_division=0
            ),
            "escalation_precision": round(float(precision), 4),
            "escalation_recall": round(float(recall), 4),
            "escalation_f1": round(float(route_f1), 4),
            "auto_coverage": round(float(auto.mean()), 4),
            "unsafe_auto_count": int(unsafe_auto.sum()),
            "unsafe_auto_rate": round(unsafe_auto_rate, 4),
            "route_confusion_matrix": confusion_matrix(merged["should_escalate"], predicted_escalate, labels=[False, True]).tolist(),
        }
        thresholds = pd.to_numeric(merged.get("routing_threshold"), errors="coerce").dropna().unique()
        if len(thresholds) == 1:
            system_report["routing_threshold"] = round(float(thresholds[0]), 4)
        if "judge_accept" in merged and merged["judge_accept"].notna().any():
            accepted = merged["judge_accept"].map(parse_bool)
            eligible_success = auto & ~merged["should_escalate"] & accepted & merged["predicted_intent"].eq(merged["primary_intent"])
            system_report["selective_success"] = round(float(eligible_success.sum() / max(int(auto.sum()), 1)), 4)
        system_report["by_stratum"] = {}
        for stratum, subset in merged.groupby("sample_stratum"):
            subset_escalate = subset["predicted_route"].eq("ESCALATE")
            subset_auto = ~subset_escalate
            subset_truth = subset["should_escalate"]
            _, subset_recall, _, _ = precision_recall_fscore_support(
                subset_truth, subset_escalate, average="binary", zero_division=0
            )
            subset_unsafe = subset_auto & subset_truth
            system_report["by_stratum"][stratum] = {
                "cases": int(len(subset)),
                "intent_accuracy": round(float(accuracy_score(subset["primary_intent"], subset["predicted_intent"])), 4),
                "intent_macro_f1": round(
                    float(f1_score(subset["primary_intent"], subset["predicted_intent"], labels=list(INTENTS), average="macro", zero_division=0)), 4
                ),
                "escalation_recall": round(float(subset_recall), 4),
                "auto_coverage": round(float(subset_auto.mean()), 4),
                "unsafe_auto_rate": round(float(subset_unsafe.sum() / max(int(subset_auto.sum()), 1)), 4),
            }
        report["systems"][system] = system_report

    for stratum in ("representative", "challenge"):
        ids = set(truth.loc[truth["sample_stratum"] == stratum, "gold_id"])
        report[f"{stratum}_test_cases"] = len(ids)
    return report


def write_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
