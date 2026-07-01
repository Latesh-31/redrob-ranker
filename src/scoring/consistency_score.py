"""Score profile consistency and apply honeypot penalties."""

from __future__ import annotations

from src.utils.types import CandidateFeatures


def consistency_score(candidate: CandidateFeatures) -> float:
    """Return consistency multiplier base in [0, 1] before honeypot penalty."""
    score = 1.0

    if candidate.career_count == 0 and candidate.skill_count > 20:
        score -= 0.3
    if candidate.profile_completeness < 0.3:
        score -= 0.2
    if candidate.total_years <= 0 and candidate.career_count > 0:
        score -= 0.15

    return max(0.0, min(1.0, score))


def consistency_multiplier(candidate: CandidateFeatures) -> tuple[float, float]:
    """
    Return (multiplier, honeypot_penalty).
    Hard-reject honeypots via near-zero multiplier.
    """
    base = consistency_score(candidate)
    penalty = candidate.honeypot_penalty

    from src.utils.constants import HONEYPOT_HARD_REJECT

    if penalty >= HONEYPOT_HARD_REJECT:
        return 0.0, penalty

    multiplier = max(0.0, base * (1.0 - penalty))
    return multiplier, penalty
