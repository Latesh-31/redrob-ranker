"""Combine individual signal scores into a final ranking score."""

from __future__ import annotations

from src.scoring.behavior_score import behavior_score
from src.scoring.consistency_score import consistency_multiplier
from src.scoring.education_score import education_score
from src.scoring.experience_score import experience_score
from src.scoring.skill_score import skill_score
from src.scoring.title_score import title_score
from src.utils.constants import SCORE_WEIGHTS
from src.utils.types import CandidateFeatures, JDRequirements, ScoreBreakdown


def compute_breakdown(
    candidate: CandidateFeatures,
    requirements: JDRequirements,
    semantic_score: float = 0.0,
) -> ScoreBreakdown:
    skill, matched = skill_score(candidate, requirements)
    exp = experience_score(candidate, requirements)
    title = title_score(candidate, requirements)
    edu = education_score(candidate, requirements)
    behavior = behavior_score(candidate, requirements)
    consistency_mult, honeypot_penalty = consistency_multiplier(candidate)

    weighted = (
        SCORE_WEIGHTS["semantic"] * semantic_score
        + SCORE_WEIGHTS["skill"] * skill
        + SCORE_WEIGHTS["experience"] * exp
        + SCORE_WEIGHTS["title"] * title
        + SCORE_WEIGHTS["education"] * edu
        + SCORE_WEIGHTS["behavior"] * behavior
    )

    final = min(1.0, max(0.0, weighted * consistency_mult))

    return ScoreBreakdown(
        candidate_id=candidate.candidate_id,
        semantic=semantic_score,
        skill=skill,
        experience=exp,
        title=title,
        education=edu,
        behavior=behavior,
        consistency=consistency_mult,
        honeypot_penalty=honeypot_penalty,
        final_score=final,
        matched_skills=matched,
        honeypot_flags=list(candidate.honeypot_flags),
    )


def final_score(
    candidate: CandidateFeatures,
    requirements: JDRequirements,
    semantic_score: float = 0.0,
) -> tuple[float, ScoreBreakdown]:
    breakdown = compute_breakdown(candidate, requirements, semantic_score)
    return breakdown.final_score, breakdown
