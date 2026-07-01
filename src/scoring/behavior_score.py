"""Score behavioral signals from candidate profile."""

from __future__ import annotations

from src.utils.types import CandidateFeatures, JDRequirements


def behavior_score(candidate: CandidateFeatures, requirements: JDRequirements) -> float:
    signals = candidate.signals
    if not signals:
        return 0.5

    weights: dict[str, float] = {
        "response_rate": 0.25,
        "engagement_score": 0.25,
        "profile_completeness": 0.20,
        "profile_views": 0.10,
        "avg_tenure_months": 0.10,
        "job_hop_score": 0.10,
    }

    if requirements.seniority == "senior":
        weights["job_hop_score"] = 0.15
        weights["avg_tenure_months"] = 0.15
        weights["response_rate"] = 0.15

    total_weight = 0.0
    total_score = 0.0

    for name, weight in weights.items():
        if name not in signals:
            continue
        value = float(signals[name])
        if name == "job_hop_score":
            value = 1.0 - value
        elif name == "avg_tenure_months":
            value = min(1.0, value / 36.0)
        total_score += weight * value
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return total_score / total_weight
