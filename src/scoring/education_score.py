"""Score candidate education against job requirements."""

from __future__ import annotations

from src.utils.helpers import tokenize
from src.utils.types import CandidateFeatures, JDRequirements


def education_score(candidate: CandidateFeatures, requirements: JDRequirements) -> float:
    level = candidate.max_education_level
    required = requirements.required_education_level

    if required <= 0:
        level_score = min(1.0, level / 3.0) if level else 0.4
    elif level >= required:
        level_score = 1.0
    elif level == required - 1:
        level_score = 0.6
    else:
        level_score = max(0.2, level / max(required, 1))

    field_score = 0.5
    if requirements.education_fields and candidate.education_field:
        cand_tokens = tokenize(candidate.education_field.lower())
        for field in requirements.education_fields:
            if tokenize(field) & cand_tokens:
                field_score = 1.0
                break

    return 0.7 * level_score + 0.3 * field_score
