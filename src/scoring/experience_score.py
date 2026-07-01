"""Score candidate experience against job requirements."""

from __future__ import annotations

from src.utils.types import CandidateFeatures, JDRequirements


def experience_score(candidate: CandidateFeatures, requirements: JDRequirements) -> float:
    years = candidate.relevant_years or candidate.total_years
    min_years = requirements.min_years

    if min_years <= 0:
        return min(1.0, years / 10.0) if years else 0.3

    if years >= min_years:
        excess = years - min_years
        if excess <= 5:
            return 1.0
        return max(0.7, 1.0 - (excess - 5) * 0.03)

    ratio = years / min_years
    return max(0.0, min(0.85, ratio * 0.85))
