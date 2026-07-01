"""Score candidate job title relevance."""

from __future__ import annotations

from src.utils.helpers import tokenize
from src.utils.types import CandidateFeatures, JDRequirements


def title_score(candidate: CandidateFeatures, requirements: JDRequirements) -> float:
    jd_title = requirements.title.lower()
    if not jd_title:
        return 0.5

    jd_tokens = tokenize(jd_title)
    if not jd_tokens:
        return 0.5

    best = 0.0
    titles = [candidate.current_title] + candidate.title_history
    for title in titles:
        if not title:
            continue
        title_tokens = tokenize(title.lower())
        if not title_tokens:
            continue
        overlap = len(jd_tokens & title_tokens) / len(jd_tokens | title_tokens)
        substring_bonus = 0.15 if jd_title in title.lower() or title.lower() in jd_title else 0.0
        best = max(best, min(1.0, overlap + substring_bonus))

    seniority_bonus = 0.0
    if requirements.seniority == "senior":
        for title in titles:
            if any(k in title.lower() for k in ("senior", "lead", "staff", "principal")):
                seniority_bonus = 0.1
                break

    return min(1.0, best + seniority_bonus)
