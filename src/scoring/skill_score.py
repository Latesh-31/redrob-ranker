"""Score candidate skill overlap against job requirements."""

from __future__ import annotations

from src.utils.helpers import fuzzy_skill_match
from src.utils.types import CandidateFeatures, JDRequirements


def skill_score(candidate: CandidateFeatures, requirements: JDRequirements) -> tuple[float, list[str]]:
    if not requirements.required_skills and not requirements.preferred_skills:
        return 0.5, []

    matched: list[str] = []
    req_hits = 0
    for req in requirements.required_skills:
        for cand_skill in candidate.skills:
            if fuzzy_skill_match(cand_skill, req):
                matched.append(req)
                req_hits += 1
                break

    req_total = len(requirements.required_skills) or 1
    required_ratio = req_hits / req_total

    pref_hits = 0
    for pref in requirements.preferred_skills:
        for cand_skill in candidate.skills:
            if fuzzy_skill_match(cand_skill, pref):
                if pref not in matched:
                    matched.append(pref)
                pref_hits += 1
                break

    pref_total = len(requirements.preferred_skills) or 1
    preferred_ratio = pref_hits / pref_total if requirements.preferred_skills else 0.0

    score = 0.85 * required_ratio + 0.15 * preferred_ratio
    return min(1.0, score), matched
