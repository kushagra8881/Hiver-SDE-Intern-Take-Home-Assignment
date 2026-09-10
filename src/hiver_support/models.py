from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from .taxonomy import INTENTS, keyword_intent
from .text import public_text


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2), min_df=1, max_df=0.995, sublinear_tf=True,
                    strip_accents="unicode", max_features=24_000,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                    sublinear_tf=True, max_features=32_000,
                ),
            ),
        ]
    )


class IntentModel:
    def __init__(self, seed: int = 1729) -> None:
        self.seed = seed
        self.fusion_weight = 0.5
        self.pipeline = Pipeline(
            [
                ("features", _features()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=800,
                        C=3.0,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, frame: pd.DataFrame) -> "IntentModel":
        labelled = frame[frame["primary_intent"].isin(INTENTS)].copy()
        if labelled["primary_intent"].nunique() < 2:
            raise ValueError("Need at least two labelled intents to train")
        text = (labelled["context"].fillna("") + " [CURRENT] " + labelled["message"].fillna(""))
        labels = labelled["primary_intent"].to_numpy()
        min_class = int(labelled["primary_intent"].value_counts().min())
        folds = min(5, min_class)
        if folds >= 2:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
            oof = cross_val_predict(
                clone(self.pipeline), text, labels, cv=splitter, method="predict_proba"
            )
            classes = sorted(set(labels))
            rule = np.vstack([self._rule_distribution(value) for value in labelled["message"]])
            # cross_val_predict columns follow the estimator's sorted classes.
            class_indices = [INTENTS.index(value) for value in classes]
            rule = rule[:, class_indices]
            best: tuple[float, float] = (-1.0, 0.0)
            for weight in np.linspace(0.0, 1.0, 11):
                combined = weight * oof + (1.0 - weight) * rule
                predicted = np.asarray(classes)[combined.argmax(axis=1)]
                score = f1_score(labels, predicted, labels=list(INTENTS), average="macro", zero_division=0)
                # Prefer more learned signal when validation scores tie.
                best = max(best, (float(score), float(weight)))
            self.fusion_weight = best[1]
        self.pipeline.fit(text, labels)
        return self

    @staticmethod
    def _rule_distribution(message: str) -> np.ndarray:
        intent, confidence = keyword_intent(message)
        remainder = (1.0 - confidence) / (len(INTENTS) - 1)
        values = np.full(len(INTENTS), remainder, dtype=float)
        values[INTENTS.index(intent)] = confidence
        return values

    def predict(self, message: str, context: str = "") -> tuple[str, float, dict[str, float]]:
        text = f"{context} [CURRENT] {message}"
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        learned = np.asarray(
            [float(probabilities[classes.index(label)]) if label in classes else 0.0 for label in INTENTS]
        )
        rule = self._rule_distribution(message)
        combined = self.fusion_weight * learned + (1.0 - self.fusion_weight) * rule
        scores = {label: float(combined[index]) for index, label in enumerate(INTENTS)}
        best = max(scores, key=scores.get)
        rule_intent, rule_confidence = keyword_intent(message)
        confidence = max(scores[best], rule_confidence) if best == rule_intent else scores[best]
        return best, round(confidence, 4), {key: round(value, 4) for key, value in scores.items()}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "IntentModel":
        return joblib.load(path)


@dataclass(frozen=True)
class RetrievedCase:
    pair_id: str
    conversation_id: str
    message: str
    reply: str
    similarity: float
    quality: float
    weak_intent: str


class RetrievalIndex:
    def __init__(self, max_features: int = 60_000) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.998,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.matrix: sparse.csr_matrix | None = None
        self.cases: pd.DataFrame | None = None

    def fit(self, frame: pd.DataFrame, *, max_pairs: int = 30_000, seed: int = 1729) -> "RetrievalIndex":
        usable = frame[
            frame["message"].fillna("").str.len().between(5, 600)
            & frame["historical_reply"].fillna("").str.len().between(8, 600)
        ].copy()
        usable = usable.sort_values(["reply_quality", "pair_id"], ascending=[False, True])
        usable = usable.drop_duplicates("text_fingerprint").drop_duplicates("conversation_id")
        if len(usable) > max_pairs:
            # Preserve half of the best replies and use a stable random half for coverage.
            best_n = max_pairs // 2
            best = usable.head(best_n)
            rest = usable.drop(index=best.index).sample(n=max_pairs - best_n, random_state=seed)
            usable = pd.concat([best, rest], ignore_index=True)
        self.cases = usable[
            ["pair_id", "conversation_id", "message", "historical_reply", "reply_quality", "weak_intent"]
        ].reset_index(drop=True)
        self.matrix = self.vectorizer.fit_transform(self.cases["message"].map(public_text)).tocsr()
        return self

    def retrieve(
        self,
        message: str,
        *,
        intent: str | None = None,
        k: int = 3,
        exclude_conversation: str | None = None,
    ) -> list[RetrievedCase]:
        if self.matrix is None or self.cases is None:
            raise RuntimeError("Retrieval index is not fitted")
        query = self.vectorizer.transform([public_text(message)])
        similarities = (query @ self.matrix.T).toarray()[0]
        quality = self.cases["reply_quality"].astype(float).to_numpy()
        intent_bonus = (
            (self.cases["weak_intent"].astype(str).to_numpy() == intent).astype(float) * 0.035
            if intent else 0.0
        )
        ranking_score = similarities + 0.06 * quality + intent_bonus
        order = np.argsort(-ranking_score, kind="stable")
        results: list[RetrievedCase] = []
        seen_replies: set[str] = set()
        for index in order:
            row = self.cases.iloc[int(index)]
            if exclude_conversation and str(row["conversation_id"]) == str(exclude_conversation):
                continue
            reply_key = " ".join(str(row["historical_reply"]).lower().split())[:120]
            if reply_key in seen_replies:
                continue
            seen_replies.add(reply_key)
            results.append(
                RetrievedCase(
                    pair_id=str(row["pair_id"]),
                    conversation_id=str(row["conversation_id"]),
                    message=str(row["message"]),
                    reply=str(row["historical_reply"]),
                    similarity=round(float(similarities[int(index)]), 4),
                    quality=round(float(row["reply_quality"]), 4),
                    weak_intent=str(row["weak_intent"]),
                )
            )
            if len(results) >= k:
                break
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target, compress=3)

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalIndex":
        return joblib.load(path)
