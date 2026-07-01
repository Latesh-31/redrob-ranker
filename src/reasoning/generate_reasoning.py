"""Generate human-readable reasoning for candidate rankings."""

from __future__ import annotations

from src.utils.constants import SCORE_WEIGHTS
from src.utils.types import CandidateFeatures, ScoreBreakdown

REASON_TEMPLATES = {
    "skill": "Strong skill match ({skills})",
    "experience": "Relevant experience ({years:.0f} years)",
    "title": "Title alignment with role ({title})",
    "semantic": "High overall profile relevance to job description",
    "education": "Education meets role requirements",
    "behavior": "Strong platform engagement and behavioral signals",
    "education_gap": "Education below preferred level",
    "experience_gap": "Experience below required minimum",
    "skill_gap": "Partial skill overlap with requirements",
    "title_gap": "Limited title match with target role",
    "behavior_gap": "Behavioral signals below average",
    "honeypot": "Profile consistency concerns noted.",
    "only_consulting_firms": "Career limited to IT consulting firms.",
    "title_chaser": "Frequent job changes with rapid title inflation.",
    "pure_research": "Experience limited to research/academic roles.",
    "location_mismatch": "Location mismatch for hybrid role and not willing to relocate.",
    "inactive_unresponsive": "Inactive on platform and unresponsive to recruiters.",
}


def _strength_phrase(name: str, breakdown: ScoreBreakdown, candidate: CandidateFeatures) -> str:
    if name == "skill" and breakdown.matched_skills:
        skills = ", ".join(breakdown.matched_skills[:5])
        return REASON_TEMPLATES["skill"].format(skills=skills)
    if name == "experience":
        years = candidate.relevant_years or candidate.total_years
        return REASON_TEMPLATES["experience"].format(years=years)
    if name == "title":
        title = candidate.current_title or (candidate.title_history[0] if candidate.title_history else "N/A")
        return REASON_TEMPLATES["title"].format(title=title)
    return REASON_TEMPLATES.get(name, f"Strong {name} match")


def _gap_phrase(name: str, score: float) -> str:
    gap_map = {
        "education": REASON_TEMPLATES["education_gap"],
        "experience": REASON_TEMPLATES["experience_gap"],
        "skill": REASON_TEMPLATES["skill_gap"],
        "title": REASON_TEMPLATES["title_gap"],
        "behavior": REASON_TEMPLATES["behavior_gap"],
    }
    return gap_map.get(name, f"Moderate gap in {name}")


def generate_reasoning(
    breakdown: ScoreBreakdown,
    candidate: CandidateFeatures,
    rank: int,
) -> str:
    contribs = breakdown.weighted_contributions(SCORE_WEIGHTS)
    strengths: list[str] = []

    component_scores = {
        "semantic": breakdown.semantic,
        "skill": breakdown.skill,
        "experience": breakdown.experience,
        "title": breakdown.title,
        "education": breakdown.education,
        "behavior": breakdown.behavior,
    }

    # Prioritize specific concrete strengths (skills and titles) over generic semantic relevance
    # to make the reasoning highly diverse and candidate-specific
    for name in ["skill", "title", "experience", "education", "behavior"]:
        score = component_scores.get(name, 0)
        if score >= 0.35:
            if name == "skill" and not breakdown.matched_skills:
                continue
            strengths.append(_strength_phrase(name, breakdown, candidate))
            if len(strengths) >= 2:
                break

    if len(strengths) < 2:
        score = component_scores.get("semantic", 0)
        if score >= 0.45:
            strengths.append(_strength_phrase("semantic", breakdown, candidate))

    gap = ""
    # Map strength prefix to score category name to avoid gap contradiction
    strength_categories = set()
    for s in strengths:
        s_lower = s.lower()
        if "skill" in s_lower:
            strength_categories.add("skill")
        elif "title" in s_lower:
            strength_categories.add("title")
        elif "experience" in s_lower:
            strength_categories.add("experience")
        elif "education" in s_lower:
            strength_categories.add("education")
        elif "platform" in s_lower or "behavioral" in s_lower:
            strength_categories.add("behavior")
        elif "relevance" in s_lower:
            strength_categories.add("semantic")

    # Sort components by score ascending, pick the weakest that is not already a strength
    for name, score in sorted(component_scores.items(), key=lambda x: x[1]):
        if name in strength_categories:
            continue
        if score < 0.50:
            gap = _gap_phrase(name, score)
            break

    parts = [
        f"Candidate {breakdown.candidate_id} ranks #{rank} (score {breakdown.final_score:.2f}).",
    ]
    if strengths:
        parts.append("Strengths: " + "; ".join(strengths[:2]) + ".")
    if gap:
        parts.append(f"Gap: {gap}.")
    if breakdown.honeypot_penalty > 0.2:
        triggered = []
        for flag in breakdown.honeypot_flags:
            if flag in REASON_TEMPLATES:
                triggered.append(REASON_TEMPLATES[flag])
        if triggered:
            parts.append(" ".join(triggered))
        else:
            parts.append(REASON_TEMPLATES["honeypot"])

    return " ".join(parts)
