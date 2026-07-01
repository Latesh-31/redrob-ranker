"""Optional XGBoost reranker for shortlist filtering and refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.utils.types import CandidateFeatures, ScoreBreakdown

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - fallback path is covered instead
    XGBClassifier = None


RERANK_FEATURE_NAMES = [
    "semantic_score",
    "heuristic_score",
    "skill_score",
    "experience_score",
    "title_score",
    "education_score",
    "behavior_score",
    "consistency_multiplier",
    "honeypot_penalty",
    "matched_skill_count",
    "skill_count",
    "total_years",
    "relevant_years",
    "max_education_level",
    "cert_count",
    "language_count",
    "profile_completeness",
    "career_count",
    "signal_profile_views",
    "signal_response_rate",
    "signal_avg_tenure_months",
    "signal_job_hop_score",
    "signal_engagement_score",
    "signal_profile_completeness",
]


@dataclass
class RerankResult:
    candidate: CandidateFeatures
    breakdown: ScoreBreakdown
    semantic_score: float
    fit_probability: float
    rerank_score: float


class XGBReranker:
    def __init__(
        self,
        positive_quantile: float = 0.75,
        negative_quantile: float = 0.50,
        filter_threshold: float = 0.45,
        blend_weight: float = 0.35,
        random_state: int = 42,
    ) -> None:
        self.positive_quantile = positive_quantile
        self.negative_quantile = negative_quantile
        self.filter_threshold = filter_threshold
        self.blend_weight = blend_weight
        self.random_state = random_state
        self.model: XGBClassifier | None = None

    def _signal(self, candidate: CandidateFeatures, name: str) -> float:
        value = candidate.signals.get(name, 0.5)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.5

    def _feature_vector(
        self,
        candidate: CandidateFeatures,
        breakdown: ScoreBreakdown,
        semantic_score: float,
    ) -> np.ndarray:
        return np.array(
            [
                semantic_score,
                breakdown.final_score,
                breakdown.skill,
                breakdown.experience,
                breakdown.title,
                breakdown.education,
                breakdown.behavior,
                breakdown.consistency,
                breakdown.honeypot_penalty,
                float(len(breakdown.matched_skills)),
                float(candidate.skill_count),
                float(candidate.total_years),
                float(candidate.relevant_years),
                float(candidate.max_education_level),
                float(candidate.cert_count),
                float(candidate.language_count),
                float(candidate.profile_completeness),
                float(candidate.career_count),
                self._signal(candidate, "profile_views"),
                self._signal(candidate, "response_rate"),
                self._signal(candidate, "avg_tenure_months"),
                self._signal(candidate, "job_hop_score"),
                self._signal(candidate, "engagement_score"),
                self._signal(candidate, "profile_completeness"),
            ],
            dtype=np.float32,
        )

    def _build_training_data(
        self,
        records: Sequence[tuple[CandidateFeatures, ScoreBreakdown, float]],
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if not records:
            return None, None

        scores = np.array([breakdown.final_score for _, breakdown, _ in records], dtype=np.float32)
        if len(scores) < 8:
            threshold = float(np.median(scores)) if len(scores) else 0.5
            features = np.stack([self._feature_vector(candidate, breakdown, semantic) for candidate, breakdown, semantic in records])
            labels = (scores >= threshold).astype(np.int32)
            if len(np.unique(labels)) < 2:
                return None, None
            return features, labels

        upper = float(np.quantile(scores, self.positive_quantile))
        lower = float(np.quantile(scores, self.negative_quantile))

        keep_mask = (scores >= upper) | (scores <= lower)
        if not np.any(keep_mask):
            return None, None

        labels = np.where(scores >= upper, 1, 0).astype(np.int32)
        selected_records = [record for record, keep in zip(records, keep_mask, strict=False) if keep]
        selected_labels = labels[keep_mask]

        if len(np.unique(selected_labels)) < 2:
            threshold = float(np.median(scores))
            features = np.stack([self._feature_vector(candidate, breakdown, semantic) for candidate, breakdown, semantic in records])
            labels = (scores >= threshold).astype(np.int32)
            if len(np.unique(labels)) < 2:
                return None, None
            return features, labels

        features = np.stack(
            [self._feature_vector(candidate, breakdown, semantic) for candidate, breakdown, semantic in selected_records]
        )
        return features, selected_labels

    def fit(self, records: Sequence[tuple[CandidateFeatures, ScoreBreakdown, float]]) -> bool:
        if XGBClassifier is None:
            self.model = None
            return False

        features, labels = self._build_training_data(records)
        if features is None or labels is None or len(features) < 8 or len(np.unique(labels)) < 2:
            self.model = None
            return False

        self.model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            tree_method="hist",
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(features, labels)
        return True

    def predict_proba(
        self,
        records: Sequence[tuple[CandidateFeatures, ScoreBreakdown, float]],
    ) -> np.ndarray:
        if not records:
            return np.zeros(0, dtype=np.float32)

        features = np.stack([self._feature_vector(candidate, breakdown, semantic) for candidate, breakdown, semantic in records])
        if self.model is None:
            return np.array([breakdown.final_score for _, breakdown, _ in records], dtype=np.float32)

        probabilities = self.model.predict_proba(features)
        if probabilities.ndim != 2 or probabilities.shape[1] < 2:
            return np.array([breakdown.final_score for _, breakdown, _ in records], dtype=np.float32)
        return probabilities[:, 1].astype(np.float32)

    def rerank(
        self,
        records: Sequence[tuple[CandidateFeatures, ScoreBreakdown, float]],
        minimum_keep: int = 1,
    ) -> list[RerankResult]:
        if not records:
            return []

        probabilities = self.predict_proba(records)
        enriched = [
            RerankResult(
                candidate=candidate,
                breakdown=breakdown,
                semantic_score=semantic,
                fit_probability=float(prob),
                rerank_score=float((1.0 - self.blend_weight) * breakdown.final_score + self.blend_weight * prob),
            )
            for (candidate, breakdown, semantic), prob in zip(records, probabilities, strict=False)
        ]

        eligible = [item for item in enriched if item.fit_probability >= self.filter_threshold]
        if len(eligible) < minimum_keep:
            eligible = enriched

        eligible.sort(key=lambda item: (-item.rerank_score, -item.fit_probability, item.candidate.candidate_id))
        return eligible